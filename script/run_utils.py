import numpy as np

# Global variables for gait timing

# Define phase identifiers
LEFT_STANCE = 0
FLIGHT_AFTER_LEFT = 1
IMPACT_AFTER_LEFT = 2
RIGHT_STANCE = 3
FLIGHT_AFTER_RIGHT = 4
IMPACT_AFTER_RIGHT = 5

# Gait parameters
STANCE_DURATION = 0.6
FLIGHT_DURATION = 0.8
IMPACT_DURATION = 0.001
CYCLE_DURATION = 2 * (STANCE_DURATION + FLIGHT_DURATION + IMPACT_DURATION)

def get_fsm(t: float) -> int:
    t_in_cycle = t % CYCLE_DURATION
    T1 = STANCE_DURATION
    T2 = T1 + FLIGHT_DURATION
    T3 = T2 + IMPACT_DURATION
    T4 = T3 + STANCE_DURATION
    T5 = T4 + FLIGHT_DURATION
    T6 = T5 + IMPACT_DURATION 

    if t_in_cycle < T1:
        return LEFT_STANCE 
    elif t_in_cycle < T2:
        return FLIGHT_AFTER_LEFT 
    elif t_in_cycle < T3:
        return IMPACT_AFTER_LEFT
    elif t_in_cycle < T4:
        return RIGHT_STANCE 
    elif t_in_cycle < T5:
        return FLIGHT_AFTER_RIGHT
    else:
        return IMPACT_AFTER_RIGHT

def time_until_switch(t: float) -> float:
    t_in_cycle = t % CYCLE_DURATION
    T1 = STANCE_DURATION
    T2 = T1 + FLIGHT_DURATION
    T3 = T2 + IMPACT_DURATION
    T4 = T3 + STANCE_DURATION
    T5 = T4 + FLIGHT_DURATION
    T6 = T5 + IMPACT_DURATION

    if t_in_cycle < T1:
        return T1 - t_in_cycle
    elif t_in_cycle < T2:
        return T2 - t_in_cycle
    elif t_in_cycle < T3:
        return T3 - t_in_cycle
    elif t_in_cycle < T4:
        return T4 - t_in_cycle
    elif t_in_cycle < T5:
        return T5 - t_in_cycle
    else:
        return T6 - t_in_cycle

def time_since_switch(t: float) -> float:
    t_in_cycle = t % CYCLE_DURATION
    T1 = STANCE_DURATION
    T2 = T1 + FLIGHT_DURATION
    T3 = T2 + IMPACT_DURATION
    T4 = T3 + STANCE_DURATION
    T5 = T4 + FLIGHT_DURATION
    T6 = T5 + IMPACT_DURATION

    if t_in_cycle < T1:
        return t_in_cycle - 0  
    elif t_in_cycle < T2:
        return t_in_cycle - T1
    elif t_in_cycle < T3:
        return t_in_cycle - T2
    elif t_in_cycle < T4:
        return t_in_cycle - T3
    elif t_in_cycle < T5:
        return t_in_cycle - T4
    else:
        return t_in_cycle - T5