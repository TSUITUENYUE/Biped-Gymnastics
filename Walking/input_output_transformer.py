import numpy as np
from pydrake.all import LeafSystem, BasicVector

# Decompose Full Plant (22) states to Lower Body (14), Left Arm (4), and Right Arm (4) states
class DecomposeStates(LeafSystem):
    def __init__(self):
        super().__init__()
        
        # Declare an input port for the full plant state (22 states)
        self.input_port_index = self.DeclareVectorInputPort("full_state", BasicVector(22)).get_index()
        
        # Declare an output port for the lower body states
        self.lower_body_states_output_port_index = self.DeclareVectorOutputPort(
            "lower_body_states", BasicVector(14),  # 14 states in total
            lambda context, output: output.SetFromVector(
                np.hstack([
                    self.get_input_port(self.input_port_index).Eval(context)[:3],         # 1st to 3rd states
                    self.get_input_port(self.input_port_index).Eval(context)[7:14],       # 8th to 14th states
                    self.get_input_port(self.input_port_index).Eval(context)[18:22]      # 19th to 22nd states
                ])
            )
        ).get_index()

        # Declare an output port for the left arm states 
        self.left_arm_states_output_port_index = self.DeclareVectorOutputPort(
            "left_arm_states", BasicVector(4),  # 4 states in total
            lambda context, output: output.SetFromVector(
                np.hstack([
                    self.get_input_port(self.input_port_index).Eval(context)[3:5],         # 4th to 5th states
                    self.get_input_port(self.input_port_index).Eval(context)[14:16],       # 15th to 16th states
                ])
            )
        ).get_index()

        # Declare an output port for the right arm states 
        self.right_arm_states_output_port_index = self.DeclareVectorOutputPort(
            "right_arm_states", BasicVector(4),  # 4 states in total
            lambda context, output: output.SetFromVector(
                np.hstack([
                    self.get_input_port(self.input_port_index).Eval(context)[5:7],         # 6th to 7th states
                    self.get_input_port(self.input_port_index).Eval(context)[16:18],       # 17th to 18th states
                ])
            )
        ).get_index()

class CombineControlInputs(LeafSystem):
    def __init__(self):
        super().__init__()
        
        self.left_arm_control_input_port_index = self.DeclareVectorInputPort(
            "left_arm_control_input", BasicVector(2)
            ).get_index()
        self.right_arm_control_input_port_index = self.DeclareVectorInputPort(
            "right_arm_control_input", BasicVector(2)
            ).get_index()
        
        self.osc_input_port_index = self.DeclareVectorInputPort(
            "osc_input", BasicVector(4)
            ).get_index()
        
        # Declare an output port with 8 elements
        self.combined_control_input_output_port_index = self.DeclareVectorOutputPort(
            "combined_control_input", BasicVector(8),
            lambda context, output: output.SetFromVector(
                [
                self.get_input_port(self.osc_input_port_index).Eval(context)[0], 
                self.get_input_port(self.osc_input_port_index).Eval(context)[1], 
                self.get_input_port(self.osc_input_port_index).Eval(context)[2], 
                self.get_input_port(self.osc_input_port_index).Eval(context)[3], 
                self.get_input_port(self.left_arm_control_input_port_index).Eval(context)[0], 
                self.get_input_port(self.right_arm_control_input_port_index).Eval(context)[0], 
                self.get_input_port(self.left_arm_control_input_port_index).Eval(context)[1], 
                self.get_input_port(self.right_arm_control_input_port_index).Eval(context)[1]
                ]
            )
        ).get_index()