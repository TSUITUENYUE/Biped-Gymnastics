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
import run_utils as fsm_utils
importlib.reload(fsm_utils)

from run_utils import get_fsm,LEFT_STANCE, FLIGHT_AFTER_LEFT, RIGHT_STANCE, FLIGHT_AFTER_RIGHT,STANCE_DURATION,FLIGHT_DURATION,CYCLE_DURATION,time_until_switch, time_since_switch
from osc_tracking_objective import PointOnFrame
from footstep_planner import LipTrajPlanner

LEFT_STANCE_IDENTIFIER = 0
FLIGHT_AFTER_LEFT_IDENTIFIER = 1
RIGHT_STANCE_IDENTIFIER = 2
FLIGHT_AFTER_RIGHT_IDENTIFIER = 3

    
class RunStepPlanner(LipTrajPlanner):
    def __init__(self):
        super().__init__()
        self.running_speed_input_port_index = self.DeclareVectorInputPort("vdes_run", 1).get_index()
        self.DeclareVectorOutputPort("com_position", 3, self.CalcComPosition)
        self.DeclareVectorOutputPort("swing_foot_position", 3, self.CalcSwingFootPosition)


        self.swing_foot_points = {
            LEFT_STANCE: PointOnFrame(
                self.plant.GetBodyByName("right_lower_leg").body_frame(),
                np.array([0, 0, -0.5])
            ),
            FLIGHT_AFTER_LEFT: PointOnFrame(
                self.plant.GetBodyByName("right_lower_leg").body_frame(),
                np.array([0, 0, -0.5])
            ),
            RIGHT_STANCE: PointOnFrame(
                self.plant.GetBodyByName("left_lower_leg").body_frame(),
                np.array([0, 0, -0.5])
            ),
            FLIGHT_AFTER_RIGHT: PointOnFrame(
                self.plant.GetBodyByName("left_lower_leg").body_frame(),
                np.array([0, 0, -0.5])
            )
        }

        self.stance_foot_points = {
            LEFT_STANCE: PointOnFrame(
                self.plant.GetBodyByName("left_lower_leg").body_frame(),
                np.array([0, 0, -0.5])
            ),
            RIGHT_STANCE: PointOnFrame(
                self.plant.GetBodyByName("right_lower_leg").body_frame(),
                np.array([0, 0, -0.5])
            ),
            FLIGHT_AFTER_LEFT: None,
            FLIGHT_AFTER_RIGHT: None
        }

    def get_running_speed_input_port(self):
        return self.get_input_port(self.running_speed_input_port_index)

    def CalcYdot(self) -> np.ndarray:
        return (self.CalcJ() @ self.plant.GetVelocities(self.plant_context)).ravel()

    def CalcJ(self) -> np.ndarray:
        return self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
            self.plant_context, JacobianWrtVariable.kV, self.plant.world_frame(), self.plant.world_frame()
        )
        
    def CalcSwingFootVelocityAtLiftoff(self, context, swing_foot_frame):
        # Get the current velocities
        v = self.plant.GetVelocities(self.plant_context)

        # Calculate the Jacobian of the swing foot
        J = self.plant.CalcJacobianTranslationalVelocity(
            self.plant_context,
            JacobianWrtVariable.kV,
            swing_foot_frame,
            np.zeros(3),  # Point of interest at the origin of the foot frame
            self.plant.world_frame(),
            self.plant.world_frame()
        )

        # Calculate the swing foot velocity
        swing_foot_velocity = J @ v
        return swing_foot_velocity.ravel()
    def CalcComPosVel(self, state):
        nq = self.plant.num_positions()
        nv = self.plant.num_velocities()
        q = state[:nq]
        v = state[nq:nq + nv]

        # Set the positions and velocities in the plant context
        self.plant.SetPositions(self.plant_context, q)
        self.plant.SetVelocities(self.plant_context, v)

        # Calculate CoM position in the world frame
        com_pos = self.plant.CalcCenterOfMassPositionInWorld(self.plant_context)

        # Calculate CoM translational velocity in the world frame
        com_vel = self.CalcYdot()

        return com_pos, com_vel

        
    def ComputeDesiredFootPlacement(self, context, state, t, end_time):
        # Get the CoM position and velocity at the start of the flight phase
        com_pos, com_vel = self.CalcComPosVel(state)
    
        # Flight duration
        flight_duration = end_time - t
    
        # Predict CoM position at landing
        com_pos_end_x = com_pos[0] + com_vel[0] * flight_duration
    
        # Get the desired running speed from input port
        vdes_run = self.EvalVectorInput(context, self.running_speed_input_port_index).get_value()
        v_desired = vdes_run[0]
    
        # LIPM parameter
        H = self.H  # Height of CoM during stance phase
        omega = np.sqrt(9.81 / H)
    
        # Compute desired foot placement using LIPM dynamics
        desired_foot_placement_x = com_pos_end_x - (v_desired / omega)
    
        # Set desired foot placement y coordinate (assuming straight-line running)
        desired_foot_placement_y = com_pos[1] 
    
        # Foot should land on the ground (z = 0)
        desired_foot_placement_z = -0.01  # Or slightly negative to ensure ground contact
    
        desired_foot_placement = np.array([
            desired_foot_placement_x,
            desired_foot_placement_y,
            desired_foot_placement_z
        ])
    
        return desired_foot_placement



    def CalcSwingFootTraj(self, context: Context, output: Trajectory) -> None:
        state = self.EvalVectorInput(context, self.robot_state_input_port_index).value()
        t = context.get_time()
        fsm = get_fsm(t)
        start_time = t - time_since_switch(t)
        end_time = t + time_until_switch(t)
        is_flight_phase = (fsm == FLIGHT_AFTER_LEFT_IDENTIFIER or fsm == FLIGHT_AFTER_RIGHT_IDENTIFIER)
        foot_clearance_height = 0.1  # Adjust as appropriate

        if is_flight_phase:
            # Swing foot trajectory during flight phase
            swing_pos_at_liftoff = context.get_discrete_state(self.foot_position_at_liftoff_idx).get_value()
            desired_foot_placement = self.ComputeDesiredFootPlacement(context, state, t, end_time)

            Y0 = swing_pos_at_liftoff
            Y2 = desired_foot_placement
            Y1 = (Y0 + Y2) / 2
            Y1[2] = max(Y0[2], Y2[2]) + foot_clearance_height
            # Y1[2] = foot_clearance_height
            
            # Compute the swing foot velocity at liftoff
            swing_foot = self.swing_foot_points[fsm]
            v0 = self.CalcSwingFootVelocityAtLiftoff(context, swing_foot.frame)

            # Set final velocity at landing
            vf_z = -0.1  # Gentle landing
            vf_x = v0[0]  # Maintain horizontal velocity
            vf_y = v0[1]  # Maintain lateral velocity (if any)
            vf = np.array([vf_x, vf_y, vf_z])

            # Compute midpoint derivative
            vmid = (v0 + vf) / 2

            # Derivatives at knot points
            derivatives = [v0, vmid, vf]

            # Time samples and positions
            time_samples = [start_time, (start_time + end_time) / 2, end_time]
            positions = [Y0, Y1, Y2]

            # Use CubicHermite for trajectory generation
            swing_traj = PiecewisePolynomial.CubicHermite(
                time_samples,
                np.hstack([pos.reshape(-1, 1) for pos in positions]),
                np.hstack([der.reshape(-1, 1) for der in derivatives])
            )
            output.set_value(swing_traj)
        else:
            # print(fsm,"run_swing")
            alip_state = self.CalcAlipState(fsm, state)
            A = np.fliplr(np.diag([1.0 / (self.m * self.H), self.m * 9.81]))
            alip_pred = expm(time_until_switch(t) * A) @ alip_state

            stance_foot = self.stance_foot_points[fsm]
            stance_foot_pos = self.plant.CalcPointsPositions(
                self.plant_context, stance_foot.frame, stance_foot.pt, self.plant.world_frame()).ravel()
            com_pred = self.ConstructAlipComTraj(t, end_time, alip_state, stance_foot_pos).value(end_time).ravel()
            vdes_run = self.EvalVectorInput(context, self.running_speed_input_port_index).get_value()
            Ly_des = vdes_run[0] * self.H * self.m
            omega = np.sqrt(9.81/self.H)
            T = time_until_switch(t)
            Ly_0 = alip_pred[-1]

            p_x_foot_to_com = (Ly_des - np.cosh(omega * T) * Ly_0) / (self.m * self.H * omega * np.sinh(omega * T))

            swing_pos_at_liftoff = context.get_discrete_state(self.foot_position_at_liftoff_idx).get_value()

            Y0 = swing_pos_at_liftoff
            Y2 = np.zeros((3,))
            Y2[0] = com_pred[0] - p_x_foot_to_com
            Y1 = 0.5 * (Y0 + Y2)
            Y1[-1] = 0.05  # Higher foot lift for running
            output.set_value(
                PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                    [start_time, 0.5*(start_time + end_time), end_time],
                    [np.expand_dims(y, axis=1) for y in [Y0, Y1, Y2]],
                    np.zeros((3,)),
                    np.array([0, 0, -0.5])
                )
            )

    def CalcComTraj(self, context: Context, output) -> None:
        state = self.EvalVectorInput(context, self.robot_state_input_port_index).value()
        t = context.get_time()
        fsm = get_fsm(t)
        start_time = t - time_since_switch(t)
        end_time = t + time_until_switch(t)
        is_flight_phase = (fsm == FLIGHT_AFTER_LEFT_IDENTIFIER or fsm == FLIGHT_AFTER_RIGHT_IDENTIFIER)

        if is_flight_phase:
            # print(fsm,"com")
            # Ballistic CoM trajectory during flight
            com_pos, com_vel = self.CalcComPosVel(state)
            x0, y0, z0 = com_pos[0],com_pos[1],com_pos[2]
            vx0,vy0, vz0 = com_vel[0],com_vel[1],com_vel[2]
            duration = end_time - start_time

            # Positions at start and end
            pos_start = np.array([x0,y0,z0])
            pos_end = np.array([
                x0 + vx0 * duration,
                y0 + vy0 * duration,
                z0 + vz0 * duration - 0.5 * 9.81 * duration ** 2
            ])

            # Derivatives at start and end
            vel_start = np.array([vx0, vy0, vz0])
            vel_end = np.array([vx0, vy0, vz0 - 9.81 * duration])

            # Create the trajectory
            time_samples = np.array([start_time, end_time])
            positions = np.vstack([pos_start, pos_end])
            derivatives = np.vstack([vel_start, vel_end])

            com_traj = PiecewisePolynomial.CubicHermite(time_samples, positions.T, derivatives.T)
            output.set_value(com_traj)
        else:
            # print(fsm,"com")
            # Stance phase using ALIP model
            alip_state = self.CalcAlipState(fsm, state)
            stance_foot = self.stance_foot_points[fsm]
            assert stance_foot is not None, f"Stance foot is None during stance phase {fsm}"
            stance_foot_pos = self.plant.CalcPointsPositions(
                self.plant_context, stance_foot.frame, stance_foot.pt, self.plant.world_frame()
            ).ravel()
            com_traj = self.ConstructAlipComTraj(t, end_time, alip_state, stance_foot_pos)
            output.set_value(com_traj)


    def CalcComPosition(self, context, output):
        state = self.EvalVectorInput(context, self.robot_state_input_port_index).value()
        com_pos, _ = self.CalcComPosVel(state)
        output.SetFromVector(com_pos)

    def CalcSwingFootPosition(self, context, output):
        t = context.get_time()
        fsm = get_fsm(t)
        swing_foot = self.swing_foot_points[fsm]

        # Set positions in plant context
        self.plant.SetPositions(self.plant_context, self.EvalVectorInput(context, self.robot_state_input_port_index).value()[:self.plant.num_positions()])

        # Calculate swing foot position
        swing_foot_pos = self.plant.CalcPointsPositions(
            self.plant_context, swing_foot.frame, swing_foot.pt, self.plant.world_frame()
        ).ravel()
        output.SetFromVector(swing_foot_pos)