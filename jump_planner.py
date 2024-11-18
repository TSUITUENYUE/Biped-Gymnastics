import numpy as np
from typing import List, Tuple
from scipy.linalg import expm

from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import Context, LeafSystem, BasicVector, DiscreteValues, EventStatus
from pydrake.trajectories import Trajectory, PiecewisePolynomial
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.multibody.all import JacobianWrtVariable
import importlib
import jump_utils as fsm_utils
importlib.reload(fsm_utils)

from jump_utils import get_fsm, CROUCH, TAKEOFF, FLIGHT, LANDING, time_since_switch, time_until_switch
from osc_tracking_objective import PointOnFrame
from footstep_planner import LipTrajPlanner

class JumpPlanner(LipTrajPlanner):
    def __init__(self):
        super().__init__()

        # Declare input port for robot state
        self.robot_state_input_port_index = self.DeclareVectorInputPort(
            "robot_state", self.plant.num_positions() + self.plant.num_velocities()
        ).get_index()

        # Declare output ports
        self.DeclareVectorOutputPort("com_position", 3, self.CalcComPosition)
        self.DeclareVectorOutputPort("swing_foot_position", 3, self.CalcSwingFootPosition)

        # Jump parameters
        self.crouch_depth = 0.1
        self.jump_height = 0.2
        initial_com_height = 0.5  # Set this to your robot's initial CoM height
        self.initial_com_position = np.array([0.0, 0.0, initial_com_height])

        # Discrete state indices
        self.com_pos_vel_at_takeoff_idx = self.DeclareDiscreteState(np.zeros(6))
        self.prev_fsm_state_idx = self.DeclareDiscreteState(np.array([CROUCH], dtype=int))

        # Declare a periodic event to update discrete states at phase transitions
        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=0.001,  # Adjust period as needed
            offset_sec=0.0,
            update=self.UpdateAtDiscreteEvents
        )

    def CalcYdot(self, context) -> np.ndarray:
        plant_context = self.plant.GetMyContextFromRoot(context)
        return (self.CalcJ(plant_context) @ self.plant.GetVelocities(plant_context)).ravel()

    def CalcJ(self, plant_context) -> np.ndarray:
        return self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
            plant_context, JacobianWrtVariable.kV, self.plant.world_frame(), self.plant.world_frame()
        )

    def GetFootPositionAtTime(self, t, context):
        # Obtain the plant context
        plant_context = self.plant.GetMyContextFromRoot(context)
        
        # Retrieve the robot state
        state = self.EvalVectorInput(context, self.robot_state_input_port_index).value()
        self.plant.SetPositions(plant_context, state[:self.plant.num_positions()])
    
        # Assuming both feet have the same position for simplicity
        left_foot_frame = self.plant.GetBodyByName("left_lower_leg").body_frame()
        foot_pos = self.plant.CalcPointsPositions(
            plant_context, left_foot_frame, np.zeros(3), self.plant.world_frame()
        ).ravel()
        return foot_pos

    def CalcSwingFootTraj(self, context: Context, output: Trajectory) -> None:
        t = context.get_time()
        fsm = get_fsm(t)

        if fsm == CROUCH or fsm == TAKEOFF:
            # Feet remain stationary relative to the robot during crouch and takeoff
            foot_pos = self.GetFootPositionAtTime(t, context)
            foot_traj = PiecewisePolynomial.FirstOrderHold(
                [t, t + time_until_switch(t)], [foot_pos, foot_pos]
            )
            output.set_value(foot_traj)
        elif fsm == FLIGHT:
            # Feet follow a trajectory upward during flight
            start_time = t - time_since_switch(t)
            end_time = t + time_until_switch(t)
        
            foot_pos_start = self.GetFootPositionAtTime(start_time, context)
            foot_pos_peak = foot_pos_start + np.array([0, 0, self.jump_height])
            foot_pos_end = foot_pos_start  # Landing at the same position
        
            times = [start_time, (start_time + end_time) / 2, end_time]
            positions = np.column_stack([foot_pos_start, foot_pos_peak, foot_pos_end])
        
            foot_traj = PiecewisePolynomial.CubicShapePreserving(times, positions)
            output.set_value(foot_traj)
        elif fsm == LANDING:
            # Feet return to the ground and stay there
            foot_pos = self.GetFootPositionAtTime(t, context)
            foot_traj = PiecewisePolynomial.FirstOrderHold(
                [t, t + time_until_switch(t)], [foot_pos, foot_pos]
            )
            output.set_value(foot_traj)

    def UpdateAtDiscreteEvents(self, context, state):
        # Get the current time and FSM state
        t = context.get_time()
        fsm = get_fsm(t)
        # Get the previous FSM state from the discrete state
        prev_fsm_state = int(context.get_discrete_state(self.prev_fsm_state_idx).get_value()[0])

        # Detect transition from TAKEOFF to FLIGHT
        if prev_fsm_state != FLIGHT and fsm == FLIGHT:
            # Transition to FLIGHT phase detected
            self.CalcComPosVelAtTakeoff(context, state)

        # Update the previous FSM state
        state.get_mutable_discrete_state(self.prev_fsm_state_idx).set_value([fsm])

    def CalcComPosVelAtTakeoff(self, context, state):
        # Obtain the plant context
        plant_context = self.plant.GetMyContextFromRoot(context)

        # Retrieve the robot state
        state_vector = self.EvalVectorInput(context, self.robot_state_input_port_index).value()
        nq = self.plant.num_positions()
        nv = self.plant.num_velocities()
        q = state_vector[:nq]
        v = state_vector[nq:nq + nv]

        # Set the positions and velocities in the plant context
        self.plant.SetPositions(plant_context, q)
        self.plant.SetVelocities(plant_context, v)

        # Calculate CoM position and velocity
        com_pos = self.plant.CalcCenterOfMassPositionInWorld(plant_context)
        com_vel = self.CalcYdot(context)

        # Store in the discrete state
        com_pos_vel = np.hstack([com_pos, com_vel])
        state.get_mutable_discrete_state(self.com_pos_vel_at_takeoff_idx).set_value(com_pos_vel)

    def CalcComTraj(self, context: Context, output) -> None:
        t = context.get_time()
        fsm = get_fsm(t)
        start_time = t - time_since_switch(t)
        end_time = t + time_until_switch(t)

        if fsm == CROUCH:
            # CoM lowers during the crouch phase
            initial_com_pos = self.initial_com_position
            crouch_depth = self.crouch_depth
            com_pos_start = initial_com_pos
            com_pos_end = initial_com_pos - np.array([0, 0, crouch_depth])

            # Use a smooth trajectory (e.g., cubic) for CoM descent
            times = [start_time, end_time]
            positions = np.vstack([com_pos_start, com_pos_end])
            velocities = np.zeros((2, 3))  # Assuming start and end velocities are zero

            com_traj = PiecewisePolynomial.CubicHermite(times, positions.T, velocities.T)
            output.set_value(com_traj)

        elif fsm == TAKEOFF:
            # CoM rises during the takeoff phase
            crouch_depth = self.crouch_depth
            jump_height = self.jump_height

            com_pos_start = self.initial_com_position - np.array([0, 0, crouch_depth])
            com_pos_end = com_pos_start + np.array([0, 0, jump_height])

            # Estimate initial velocity needed to reach jump height
            g = 9.81  # Gravitational acceleration
            v0_z = np.sqrt(2 * g * jump_height)
            initial_velocity = np.array([0, 0, v0_z])
            final_velocity = np.array([0, 0, 0])  # At peak, vertical velocity is zero

            times = [start_time, end_time]
            positions = np.vstack([com_pos_start, com_pos_end])
            velocities = np.vstack([initial_velocity, final_velocity])

            com_traj = PiecewisePolynomial.CubicHermite(times, positions.T, velocities.T)
            output.set_value(com_traj)

        elif fsm == FLIGHT:
            # Flight phase
            com_pos_vel_at_takeoff = context.get_discrete_state(self.com_pos_vel_at_takeoff_idx).get_value()
            com_pos = com_pos_vel_at_takeoff[:3]
            com_vel = com_pos_vel_at_takeoff[3:]

            duration = end_time - start_time
            times = [start_time, end_time]

            # Positions and velocities at start and end of flight
            positions = np.vstack([
                com_pos,
                com_pos + com_vel * duration + 0.5 * np.array([0, 0, -9.81]) * duration ** 2
            ])
            velocities = np.vstack([
                com_vel,
                com_vel + np.array([0, 0, -9.81]) * duration
            ])

            # Create the trajectory
            com_traj = PiecewisePolynomial.CubicHermite(times, positions.T, velocities.T)
            output.set_value(com_traj)

        elif fsm == LANDING:
            # CoM descends during landing phase back to initial standing height
            # Obtain the plant context
            plant_context = self.plant.GetMyContextFromRoot(context)

            # Get current CoM position and velocity
            state_vector = self.EvalVectorInput(context, self.robot_state_input_port_index).value()
            nq = self.plant.num_positions()
            nv = self.plant.num_velocities()
            q = state_vector[:nq]
            v = state_vector[nq:nq + nv]
            self.plant.SetPositions(plant_context, q)
            self.plant.SetVelocities(plant_context, v)
            com_pos_start = self.plant.CalcCenterOfMassPositionInWorld(plant_context)
            com_vel_start = self.CalcYdot(context)

            com_pos_end = self.initial_com_position
            com_vel_end = np.zeros(3)  # Assume we come to rest at the end

            times = [start_time, end_time]
            positions = np.vstack([com_pos_start, com_pos_end])
            velocities = np.vstack([com_vel_start, com_vel_end])

            # Create the trajectory
            com_traj = PiecewisePolynomial.CubicHermite(times, positions.T, velocities.T)
            output.set_value(com_traj)

        else:
            raise ValueError(f"Unknown FSM state: {fsm}")