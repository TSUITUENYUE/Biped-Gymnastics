import numpy as np
from typing import List, Tuple
from scipy.linalg import expm

from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import Context, LeafSystem, BasicVector, DiscreteValues, EventStatus
from pydrake.trajectories import Trajectory, PiecewisePolynomial
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform

import importlib
import run_utils as fsm_utils
importlib.reload(fsm_utils)

from run_utils import get_fsm,\
                      time_until_switch, time_since_switch
from osc_tracking_objective import PointOnFrame
from footstep_planner import LipTrajPlanner

class RunStepPlanner(LipTrajPlanner):
    def __init__(self):
        super().__init__()
        self.running_speed_input_port_index = self.DeclareVectorInputPort("vdes_run", 1).get_index()

    def get_running_speed_input_port(self):
        return self.get_input_port(self.running_speed_input_port_index)

    def CalcSwingFootTraj(self, context: Context, output: Trajectory) -> None:
        state = self.EvalVectorInput(context, self.robot_state_input_port_index).value()
        t = context.get_time()
        fsm = get_fsm(t)
        start_time = t - time_since_switch(t)
        end_time = t + time_until_switch(t)

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
        Y2[0] =  com_pred[0] - p_x_foot_to_com
        Y1 = 0.5 * (Y0 + Y2)
        Y1[-1] = 0.05  # Higher foot lift for running
        output.set_value(
            PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                [start_time, 0.5*(start_time + end_time),  end_time],
                [np.expand_dims(y, axis=1) for y in [Y0, Y1, Y2]],
                np.zeros((3,)),
                np.array([0, 0, -0.5])
            )
        )

    def CalcComTraj(self, context: Context, output) -> None:
        state = self.EvalVectorInput(context, self.robot_state_input_port_index).value()
        t = context.get_time()
        fsm = get_fsm(t)
        end_time = t + time_until_switch(t)

        alip_state = self.CalcAlipState(fsm, state)
        stance_foot = self.stance_foot_points[fsm]
        stance_foot_pos = self.plant.CalcPointsPositions(
            self.plant_context, stance_foot.frame, stance_foot.pt, self.plant.world_frame()).ravel()

        com_traj = self.ConstructAlipComTraj(t, end_time, alip_state, stance_foot_pos)
        output.set_value(com_traj)