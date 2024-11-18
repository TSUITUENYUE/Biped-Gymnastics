import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import Context, LeafSystem, BasicVector
from pydrake.trajectories import Trajectory, PiecewisePolynomial
from pydrake.solvers import MathematicalProgram, OsqpSolver
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.multibody.all import JacobianWrtVariable

import importlib
import osc_tracking_objective
importlib.reload(osc_tracking_objective)

from osc_tracking_objective import *

import run_utils
importlib.reload(run_utils)

from run_utils import LEFT_STANCE, RIGHT_STANCE, FLIGHT_AFTER_LEFT, FLIGHT_AFTER_RIGHT, get_fsm

@dataclass
class OscGains:
    kp_com: np.ndarray
    kd_com: np.ndarray
    w_com: np.ndarray
    kp_swing_foot: np.ndarray
    kd_swing_foot: np.ndarray
    w_swing_foot: np.ndarray
    kp_base: np.ndarray
    kd_base: np.ndarray
    w_base: np.ndarray
    w_vdot: float

class OperationalSpaceRunningController(LeafSystem):
    def __init__(self, gains: OscGains):
        """
        Constructor for the operational space controller for running.
        """
        super().__init__()
        self.gains = gains

        ''' Load the MultibodyPlant '''
        self.plant = MultibodyPlant(0.0)
        self.parser = Parser(self.plant)
        self.parser.AddModels("humanoid_walker2.urdf")  # Adjust if needed
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetBodyByName("base").body_frame(),
            RigidTransform.Identity()
        )
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        ''' Assign contact frames '''
        self.contact_points = {
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
        self.swing_foot_points = {
            LEFT_STANCE: self.contact_points[RIGHT_STANCE],
            RIGHT_STANCE: self.contact_points[LEFT_STANCE],
            FLIGHT_AFTER_LEFT: self.contact_points[LEFT_STANCE],
            FLIGHT_AFTER_RIGHT: self.contact_points[RIGHT_STANCE]
        }

        ''' Initialize tracking objectives '''
        self.tracking_objectives = {
            "com_traj": CenterOfMassPositionTrackingObjective(
                self.plant, self.plant_context, [LEFT_STANCE, RIGHT_STANCE, FLIGHT_AFTER_LEFT, FLIGHT_AFTER_RIGHT],
                self.gains.kp_com, self.gains.kd_com
            ),
            "swing_foot_traj": PointPositionTrackingObjective(
                self.plant, self.plant_context, [LEFT_STANCE, RIGHT_STANCE, FLIGHT_AFTER_LEFT, FLIGHT_AFTER_RIGHT],
                self.gains.kp_swing_foot, self.gains.kd_swing_foot, self.swing_foot_points
            ),
            "base_joint_traj": JointAngleTrackingObjective(
                self.plant, self.plant_context, [LEFT_STANCE, RIGHT_STANCE, FLIGHT_AFTER_LEFT, FLIGHT_AFTER_RIGHT],
                self.gains.kp_base, self.gains.kd_base, "planar_roty"
            )
        }
        self.tracking_costs = {
            "com_traj": self.gains.w_com,
            "swing_foot_traj": self.gains.w_swing_foot,
            "base_joint_traj": self.gains.w_base
        }
        self.trajs = self.tracking_objectives.keys()

        ''' Declare Input Ports '''
        # State input port
        self.robot_state_input_port_index = self.DeclareVectorInputPort(
            "x", self.plant.num_positions() + self.plant.num_velocities()
        ).get_index()

        # Trajectory Input Ports
        trj = PiecewisePolynomial()
        self.traj_input_ports = {
            "com_traj": self.DeclareAbstractInputPort("com_traj", AbstractValue.Make(trj)).get_index(),
            "swing_foot_traj": self.DeclareAbstractInputPort("swing_foot_traj", AbstractValue.Make(trj)).get_index(),
            "base_joint_traj": self.DeclareAbstractInputPort("base_joint_traj", AbstractValue.Make(trj)).get_index()
        }

        # Define the output ports
        self.torque_output_port = self.DeclareVectorOutputPort(
            "u", self.plant.num_actuators(), self.CalcTorques
        )

        self.u = np.zeros((self.plant.num_actuators()))

    def get_traj_input_port(self, traj_name):
        return self.get_input_port(self.traj_input_ports[traj_name])

    def get_state_input_port(self):
        return self.get_input_port(self.robot_state_input_port_index)

    def CalculateContactJacobian(self, fsm: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the Jacobian terms for the contact constraint, J and JdotV.
        Handle flight phases where there is no contact.
        """
        if fsm in [FLIGHT_AFTER_LEFT, FLIGHT_AFTER_RIGHT]:
            # No contact during flight phases
            J = np.zeros((0, self.plant.num_velocities()))
            JdotV = np.zeros((0,))
        else:
            # Contact phase
            pt_to_track = self.contact_points[fsm]
            J = self.plant.CalcJacobianTranslationalVelocity(
                self.plant_context, JacobianWrtVariable.kV, pt_to_track.frame,
                pt_to_track.pt, self.plant.world_frame(), self.plant.world_frame()
            )
            JdotV = self.plant.CalcBiasTranslationalAcceleration(
                self.plant_context, JacobianWrtVariable.kV, pt_to_track.frame,
                pt_to_track.pt, self.plant.world_frame(), self.plant.world_frame()
            ).ravel()

        return J, JdotV

    def SetupAndSolveQP(self, context: Context) -> Tuple[np.ndarray, MathematicalProgram]:
        # First get the state, time, and fsm state
        x = self.EvalVectorInput(context, self.robot_state_input_port_index).get_value()
        t = context.get_time()
        fsm = get_fsm(t)

        # Update the plant context with the current position and velocity
        self.plant.SetPositionsAndVelocities(self.plant_context, x)

        # Update tracking objectives
        for traj_name in self.trajs:
            traj = self.EvalAbstractInput(context, self.traj_input_ports[traj_name]).get_value()
            self.tracking_objectives[traj_name].Update(t, traj, fsm)

        '''Set up and solve the QP '''
        prog = MathematicalProgram()

        # Make decision variables
        u = prog.NewContinuousVariables(self.plant.num_actuators(), "u")
        vdot = prog.NewContinuousVariables(self.plant.num_velocities(), "vdot")
        if fsm in [LEFT_STANCE, RIGHT_STANCE]:
            lambda_c = prog.NewContinuousVariables(3, "lambda_c")
        else:
            lambda_c = np.zeros(0)  # No contact force during flight

        # Add Quadratic Cost on Desired Acceleration
        for traj_name in self.trajs:
            obj = self.tracking_objectives[traj_name]
            yddot_cmd_i = obj.yddot_cmd
            J_i = obj.J
            JdotV_i = obj.JdotV
            W_i = self.tracking_costs[traj_name]

            # TODO: Add Cost per tracking objective
            Q_i = J_i.T @ W_i @ J_i
            b_i = -J_i.T @ W_i @ (yddot_cmd_i - JdotV_i)
            c_i = 0.5 * (yddot_cmd_i - JdotV_i).T @ W_i @ (yddot_cmd_i - JdotV_i)

            prog.AddQuadraticCost(2 * Q_i, 2 * b_i, vdot)

        # Add Quadratic Cost on vdot using self.gains.w_vdot
        Q_vdot = self.gains.w_vdot * np.eye(self.plant.num_velocities())
        prog.AddQuadraticCost(Q_vdot, 2 * np.zeros(self.plant.num_velocities()), vdot)

        # Calculate terms in the manipulator equation
        J_c, J_c_dot_v = self.CalculateContactJacobian(fsm)
        M = self.plant.CalcMassMatrix(self.plant_context)
        Cv = self.plant.CalcBiasTerm(self.plant_context)
        G = -self.plant.CalcGravityGeneralizedForces(self.plant_context)
        B = self.plant.MakeActuationMatrix()

        # Dynamics constraint
        if fsm in [LEFT_STANCE, RIGHT_STANCE]:
            # TODO: Add the dynamics constraint
            prog.AddLinearEqualityConstraint(M @ vdot + Cv + G - B @ u - J_c.T @ lambda_c, np.zeros(len(Cv)))

            # TODO: Add Contact Constraint
            prog.AddLinearEqualityConstraint(J_c, -J_c_dot_v, vdot.flatten())
            A = np.array([[1, 0, -1], [-1, 0, -1]])
            prog.AddLinearConstraint(lambda_c[0] - lambda_c[2] <= 0)
            prog.AddLinearConstraint(-lambda_c[0] - lambda_c[2] <= 0)
            # Vertical contact force must be positive
            prog.AddLinearConstraint(lambda_c[2] >= 0)
            # Lateral force is zero (assuming planar)
            prog.AddLinearEqualityConstraint(lambda_c[1] == 0)
        else:
            # Flight phases (no contact forces)
            prog.AddLinearEqualityConstraint(M @ vdot + Cv + G - B @ u,np.zeros(len(Cv)))

        # Solve the QP
        solver = OsqpSolver()
        prog.SetSolverOption(solver.id(), "max_iter", 2000)

        result = solver.Solve(prog)

        # If the solver fails, use previous solution
        if not result.is_success():
            usol = self.u
        else:
            usol = result.GetSolution(u)
            self.u = usol

        return usol, prog

    def CalcTorques(self, context: Context, output: BasicVector) -> None:
        usol, _ = self.SetupAndSolveQP(context)
        output.SetFromVector(usol)

if __name__ == "__main__":
    gains = OscGains(
        kp_com=np.eye(3), kd_com=np.eye(3), w_com=np.eye(3),
        kp_swing_foot=np.eye(3), kd_swing_foot=np.eye(3), w_swing_foot=np.eye(3),
        kp_base=np.eye(1), kd_base=np.eye(1), w_base=np.eye(1),
        w_vdot=0.001
    )
    osc = OperationalSpaceRunningController(gains)