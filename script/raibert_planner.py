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

from run_utils import get_fsm,LEFT_STANCE, FLIGHT_AFTER_LEFT, RIGHT_STANCE, FLIGHT_AFTER_RIGHT,IMPACT_AFTER_LEFT,IMPACT_AFTER_RIGHT,STANCE_DURATION,FLIGHT_DURATION,CYCLE_DURATION,time_until_switch, time_since_switch
from osc_tracking_objective import PointOnFrame
from footstep_planner import LipTrajPlanner

LEFT_STANCE_IDENTIFIER = 0
FLIGHT_AFTER_LEFT_IDENTIFIER = 1
IMPACT_AFTER_LEFT_IDENTIFIER = 2
RIGHT_STANCE_IDENTIFIER = 3
FLIGHT_AFTER_RIGHT_IDENTIFIER = 4
IMPACT_AFTER_RIGHT_IDENTIFIER = 5


    
class RunStepPlanner(LipTrajPlanner):
    def __init__(self):
        super().__init__()
        self.running_speed_input_port_index = self.DeclareVectorInputPort("vdes_run", 1).get_index()
        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=0.001,  
            offset_sec=0.0,
            update=self.UpdateAtDiscreteEvents
        )

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
            ),
            IMPACT_AFTER_LEFT_IDENTIFIER: None,
            IMPACT_AFTER_RIGHT_IDENTIFIER: None
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
            FLIGHT_AFTER_RIGHT: None,
            IMPACT_AFTER_LEFT_IDENTIFIER: PointOnFrame(
                self.plant.GetBodyByName("right_lower_leg").body_frame(),  # The foot that impacts
                np.array([0, 0, -0.5])
            ),
            IMPACT_AFTER_RIGHT_IDENTIFIER: PointOnFrame(
                self.plant.GetBodyByName("left_lower_leg").body_frame(),
                np.array([0, 0, -0.5])
            )
        }
        
        self.liftoff_position = np.zeros(3)

    def StoreSwingFootLiftoffPosition(self, context, fsm):
        swing_foot = self.swing_foot_points[fsm]
        state_vector = self.EvalVectorInput(context, self.robot_state_input_port_index).get_value()
        nq = self.plant.num_positions()
        self.plant.SetPositions(self.plant_context, state_vector[:nq])

        swing_foot_pos = self.plant.CalcPointsPositions(
            self.plant_context, swing_foot.frame, swing_foot.pt, self.plant.world_frame()
        ).ravel()

        self.liftoff_position = swing_foot_pos 

    def GetSwingFootLiftoffPosition(self):
        return self.liftoff_position

    def get_running_speed_input_port(self):
        return self.get_input_port(self.running_speed_input_port_index)

    def CalcYdot(self) -> np.ndarray:
        return (self.CalcJ() @ self.plant.GetVelocities(self.plant_context)).ravel()

    def CalcJ(self) -> np.ndarray:
        return self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
            self.plant_context, JacobianWrtVariable.kV, self.plant.world_frame(), self.plant.world_frame()
        )

    def CalculateImpactForce(self, context, fsm):
        # Parameters
        k_spring = 10000.0  # Spring constant
        d_damper = 100.0    # Damping coefficient

        stance_foot = self.stance_foot_points[fsm]
        plant_context = self.plant_context
        
        v = self.plant.GetVelocities(plant_context)
        stance_foot_pos = self.plant.CalcPointsPositions(
            plant_context, stance_foot.frame, stance_foot.pt, self.plant.world_frame()
        ).ravel()
        penetration_depth = max(0.0, -stance_foot_pos[2])

        Jv = self.plant.CalcJacobianTranslationalVelocity(
            plant_context,
            JacobianWrtVariable.kV,
            stance_foot.frame,
            stance_foot.pt,
            self.plant.world_frame(),
            self.plant.world_frame()
        )
        stance_foot_vel = Jv @ v
        relative_velocity = -stance_foot_vel[2]

        impact_force = k_spring * penetration_depth + d_damper * relative_velocity
        return impact_force

    def ApplySpringDamperImpact(self, fsm):
        plant_context = self.plant_context

        q = self.plant.GetPositions(plant_context)
        v = self.plant.GetVelocities(plant_context)

        stance_foot = self.stance_foot_points[fsm]
        stance_foot_pos = self.plant.CalcPointsPositions(
            plant_context, stance_foot.frame, stance_foot.pt, self.plant.world_frame()
        ).ravel()
        penetration_depth = max(0.0, -stance_foot_pos[2])

        Jv = self.plant.CalcJacobianTranslationalVelocity(
            plant_context,
            JacobianWrtVariable.kV,
            stance_foot.frame,
            stance_foot.pt,
            self.plant.world_frame(),
            self.plant.world_frame()
        )
        stance_foot_vel = Jv @ v
        relative_velocity = stance_foot_vel[2]

        impact_force = self.CalculateImpactForce(plant_context,fsm)
        M = self.plant.CalcMassMatrixViaInverseDynamics(plant_context)
        delta_v = np.linalg.pinv(M) @ (Jv.T @ np.array([0, 0, impact_force]))
        new_velocities = v + delta_v

        self.plant.SetVelocities(plant_context, new_velocities)

    def UpdateAtDiscreteEvents(self, context, state):
        # Get the current time and FSM state
        t = context.get_time()
        fsm = get_fsm(t)
        # Get the previous FSM state from the discrete state
        prev_fsm_state = int(context.get_discrete_state(self.prev_fsm_state_idx).get_value()[0])

        # Detect transition to flight phase
        if prev_fsm_state in [LEFT_STANCE_IDENTIFIER, RIGHT_STANCE_IDENTIFIER] and \
            fsm in [FLIGHT_AFTER_LEFT_IDENTIFIER, FLIGHT_AFTER_RIGHT_IDENTIFIER]:
            # Transition to flight phase detected
            self.StoreFootPositionAtLiftoff(context, state, fsm)
            self.StoreSwingFootLiftoffPosition(context, prev_fsm_state)
        if fsm in [IMPACT_AFTER_LEFT_IDENTIFIER, IMPACT_AFTER_RIGHT_IDENTIFIER]:
            self.ApplySpringDamperImpact(fsm)

        # Update the previous FSM state
        state.get_mutable_discrete_state(self.prev_fsm_state_idx).set_value([fsm])

    def StoreFootPositionAtLiftoff(self, context, state, fsm):
        swing_foot = self.swing_foot_points[fsm]

        # Get the current robot state
        state_vector = self.EvalVectorInput(context, self.robot_state_input_port_index).value()
        nq = self.plant.num_positions()
        self.plant.SetPositions(self.plant_context, state_vector[:nq])

        # Get the foot position
        foot_pos = self.plant.CalcPointsPositions(
            self.plant_context, swing_foot.frame, swing_foot.pt, self.plant.world_frame()
        ).ravel()
        state.get_mutable_discrete_state(self.foot_position_at_liftoff_idx).set_value(foot_pos)
    
    def CalcSwingFootVelocityAtLiftoff(self, context, swing_foot_frame):
        swing_foot_velocity = self.CalcYdot()
        return swing_foot_velocity
        
    def CalcComPosVel(self, state):
        nq = self.plant.num_positions()
        nv = self.plant.num_velocities()
        q = state[:nq]
        v = state[nq:nq + nv]
        self.plant.SetPositions(self.plant_context, q)
        self.plant.SetVelocities(self.plant_context, v)


        # Calculate CoM position in the world frame
        com_pos = self.plant.CalcCenterOfMassPositionInWorld(self.plant_context)

        # Calculate CoM translational velocity in the world frame
        com_vel = self.CalcYdot()

        return com_pos, com_vel

        
    def ComputeDesiredFootPlacement(self, context, state):
        com_pos, com_vel = self.CalcComPosVel(state)
        vdes_run = self.EvalVectorInput(context, self.running_speed_input_port_index).get_value()
        v_desired = vdes_run[0]

        # Raibert gain
        k_raibert = 0.06
        desired_foot_placement_x = com_pos[0] + k_raibert * (com_vel[0] - v_desired)
        desired_foot_placement_y = 0
        desired_foot_placement_z = 0

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
        is_impact_phase = (fsm == IMPACT_AFTER_LEFT_IDENTIFIER or fsm == IMPACT_AFTER_RIGHT_IDENTIFIER)
        foot_clearance_height = 0.01  # Adjust as appropriate

        # print(fsm,"swing")
        if is_flight_phase:
            swing_pos_at_liftoff = context.get_discrete_state(self.foot_position_at_liftoff_idx).get_value()
            desired_foot_placement = self.ComputeDesiredFootPlacement(context, state)

            Y0 = swing_pos_at_liftoff
            Y2 = desired_foot_placement


            Y1 = (Y0 + Y2) / 2
            Y1[2] = Y0[2] + foot_clearance_height
            # Y1[2] = foot_clearance_height

            swing_foot = self.swing_foot_points[fsm]
            v0 = self.CalcSwingFootVelocityAtLiftoff(context, swing_foot.frame)

            # Set final velocity at landing
            vf_z = -0.2
            vf_x = v0[0]
            vf_y = v0[1]  
            vf = np.array([vf_x, vf_y, vf_z])

            vmid = (v0 + vf) / 2
            derivatives = [v0, vmid, vf]
    
            time_samples = [start_time, (start_time + end_time) / 2, end_time]
            positions = [Y0, Y1, Y2]

            swing_traj = PiecewisePolynomial.CubicHermite(
                time_samples,
                np.hstack([pos.reshape(-1, 1) for pos in positions]),
                np.hstack([der.reshape(-1, 1) for der in derivatives])
            )
            output.set_value(swing_traj)
        elif is_impact_phase:
            # Hold the swing foot at the landing position
            desired_foot_placement = self.ComputeDesiredFootPlacement(context, state)
            swing_traj = PiecewisePolynomial.ZeroOrderHold(
                [start_time, end_time],
                np.vstack([desired_foot_placement, desired_foot_placement]).T
            )
            output.set_value(swing_traj)
        else:
            # Plan swing foot trajectory during stance phase
            swing_foot = self.swing_foot_points[fsm]
            state_vector = self.EvalVectorInput(context, self.robot_state_input_port_index).value()
            nq = self.plant.num_positions()
            self.plant.SetPositions(self.plant_context, state_vector[:nq])
            swing_foot_pos = self.plant.CalcPointsPositions(
                self.plant_context, swing_foot.frame, swing_foot.pt, self.plant.world_frame()
            ).ravel()

    
            desired_foot_placement = self.ComputeDesiredFootPlacement(context, state)
    
            Y0 = swing_foot_pos
            Y2 = desired_foot_placement
            Y1 = (Y0 + Y2) / 2
            Y1[2] = Y0[2] + foot_clearance_height

            time_samples = [start_time, (start_time + end_time) / 2, end_time]
            positions = [Y0, Y1, Y2]

            v0 = np.zeros(3)
            vf = np.zeros(3)
            vmid = (Y2 - Y0) / (end_time - start_time)

            derivatives = [v0, vmid, vf]

            swing_traj = PiecewisePolynomial.CubicHermite(
                time_samples,
                np.hstack([pos.reshape(-1, 1) for pos in positions]),
                np.hstack([der.reshape(-1, 1) for der in derivatives])
            )
            output.set_value(swing_traj)
            
    def CalcComTraj(self, context: Context, output) -> None:
        state = self.EvalVectorInput(context, self.robot_state_input_port_index).value()
        t = context.get_time()
        fsm = get_fsm(t)
        start_time = t - time_since_switch(t)
        end_time = t + time_until_switch(t)
        is_flight_phase = (fsm == FLIGHT_AFTER_LEFT_IDENTIFIER or fsm == FLIGHT_AFTER_RIGHT_IDENTIFIER)
        is_impact_phase = (fsm == IMPACT_AFTER_LEFT_IDENTIFIER or fsm == IMPACT_AFTER_RIGHT_IDENTIFIER)
        
        # print(fsm,"com")
        
        if is_flight_phase:
            # Ballistic CoM trajectory during flight
            com_pos, com_vel = self.CalcComPosVel(state)
            x0, y0, z0 = com_pos[0],0,com_pos[2]
            vx0,vy0, vz0 = com_vel[0],0,com_vel[2]
            duration = end_time - start_time
            # print("z_before",z0)
            # Positions at start and end
            pos_start = np.array([x0,y0,z0])
            pos_end = np.array([
                x0 + vx0 * duration,
                y0 + vy0 * duration,
                z0 + vz0 * duration - 0.5 * 9.81 * (duration ** 2)
            ])

            vel_start = np.array([vx0, vy0, vz0])
            vel_end = np.array([vx0, vy0, vz0 - 9.81 * duration])

            time_samples = np.array([start_time, end_time])
            positions = np.vstack([pos_start, pos_end])
            derivatives = np.vstack([vel_start, vel_end])

            com_traj = PiecewisePolynomial.CubicHermite(time_samples, positions.T, derivatives.T)
            # print("z_after",pos_end[2])
            output.set_value(com_traj)
        elif is_impact_phase:
            com_pos_before,com_vel_before = self.CalcComPosVel(state)
            print("impact_before",com_vel_before)
            impact_force = self.CalculateImpactForce(context, fsm)

            M = self.plant.CalcMassMatrixViaInverseDynamics(self.plant_context)

            # Calculate stance foot Jacobian
            stance_foot = self.stance_foot_points[fsm]
            Jv = self.plant.CalcJacobianTranslationalVelocity(
                self.plant_context,
                JacobianWrtVariable.kV,
                stance_foot.frame,
                stance_foot.pt,
                self.plant.world_frame(),
                self.plant.world_frame()
            )

  
            delta_v = np.linalg.pinv(M) @ (Jv.T @ np.array([0, 0, impact_force]))

            J_com = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
                self.plant_context, JacobianWrtVariable.kV, self.plant.world_frame(), self.plant.world_frame()
            )

            delta_v_com = J_com @ delta_v
            com_vel_after = com_vel_before + delta_v_com
            times = np.array([t, end_time])
            positions = np.vstack([com_pos_before, com_pos_before]).T  # Position remains the same at impact
            velocities = np.vstack([com_vel_before, com_vel_after]).T

            com_traj = PiecewisePolynomial.CubicHermite(times, positions, velocities)
            print("impact_after",com_vel_after)
            output.set_value(com_traj)
        else:
            
            com_pos, com_vel = self.CalcComPosVel(state)
            duration = end_time - t
            times = np.linspace(t, end_time, num=100)

            m = self.m 
            l0 = 0.4
            l0 = 1
            g = 9.81 
            delta = 0.05 * l0 # Leg compression
            damping_ratio = 0.7
            # SLIP parameters
            k = m*g/delta # Leg stiffness
            c = 2 * damping_ratio * np.sqrt(k * m) # Leg damping
            # Initialize arrays
            x_positions = [com_pos[0]]
            z_positions = [com_pos[2]]
            x_velocities = [com_vel[0]]
            z_velocities = [com_vel[2]]
            dt = times[1] - times[0]
            #print("before_z",com_pos[2])
            #print("before_z_dot",com_vel[2])

            stance_foot = self.stance_foot_points[fsm]
            stance_foot_pos = self.plant.CalcPointsPositions(
                self.plant_context, stance_foot.frame, stance_foot.pt, self.plant.world_frame()
            ).ravel()


            dx = com_pos[0] - stance_foot_pos[0]
            dz = com_pos[2] - stance_foot_pos[2]
            r = np.sqrt(dx**2 + dz**2)
            theta = np.arctan2(dx, dz)
            r_dot = (dx * com_vel[0] + dz * com_vel[2]) / r
            theta_dot = (dx * com_vel[2] - dz * com_vel[0]) / (r**2)

            # Simulate SLIP dynamics over the stance phase
            for i in range(1, len(times)):
                # Radial acceleration
                r_ddot = r * theta_dot**2 - (k / m) * (r - l0) - (c / m) * r_dot - g * np.cos(theta)
                # Angular acceleration
                theta_ddot = (-2 * r_dot * theta_dot) / r

                # Update velocities
                r_dot += r_ddot * dt
                theta_dot += theta_ddot * dt

                # Update positions
                r += r_dot * dt
                theta += theta_dot * dt

                # Convert back to Cartesian coordinates
                x = stance_foot_pos[0] + r * np.sin(theta)
                z = stance_foot_pos[2] + r * np.cos(theta)
                x_dot = r_dot * np.sin(theta) + r * theta_dot * np.cos(theta)
                z_dot = r_dot * np.cos(theta) - r * theta_dot * np.sin(theta)

                x_positions.append(x)
                z_positions.append(z)
                x_velocities.append(x_dot)
                z_velocities.append(z_dot)

            # Build full position and velocity arrays
            positions = np.vstack([
                x_positions,
                np.full(len(times), com_pos[1]),
                z_positions
            ])
            velocities = np.vstack([
                x_velocities,
                np.full(len(times), 0),
                z_velocities
            ])

            com_traj = PiecewisePolynomial.CubicHermite(
                times,
                positions,
                velocities
            )
            #print("after_z",z)
            #print("after_z_dot",z_dot)
            output.set_value(com_traj)


