import numpy as np

# Global variables for gait timing
# Define phase identifiers
LEFT_STANCE = 0
FLIGHT_AFTER_LEFT = 1
RIGHT_STANCE = 2
FLIGHT_AFTER_RIGHT = 3

# Gait parameters
STANCE_DURATION = 0.2
FLIGHT_DURATION = 0.2
CYCLE_DURATION = 2 * (STANCE_DURATION + FLIGHT_DURATION)


def get_fsm(t: float) -> int:
    t_in_cycle = t % CYCLE_DURATION
    if t_in_cycle < STANCE_DURATION:
        return LEFT_STANCE
    elif t_in_cycle < STANCE_DURATION + FLIGHT_DURATION:
        return FLIGHT_AFTER_LEFT
    elif t_in_cycle < 2 * STANCE_DURATION + FLIGHT_DURATION:
        return RIGHT_STANCE
    else:
        return FLIGHT_AFTER_RIGHT

def time_until_switch(t: float) -> float:
    t_in_cycle = t % CYCLE_DURATION
    if t_in_cycle < STANCE_DURATION:
        return STANCE_DURATION - t_in_cycle
    elif t_in_cycle < STANCE_DURATION + FLIGHT_DURATION:
        return (STANCE_DURATION + FLIGHT_DURATION) - t_in_cycle
    elif t_in_cycle < 2 * STANCE_DURATION + FLIGHT_DURATION:
        return (2 * STANCE_DURATION + FLIGHT_DURATION) - t_in_cycle
    else:
        return CYCLE_DURATION - t_in_cycle

def time_since_switch(t: float) -> float:
    t_in_cycle = t % CYCLE_DURATION
    if t_in_cycle < STANCE_DURATION:
        return t_in_cycle
    elif t_in_cycle < STANCE_DURATION + FLIGHT_DURATION:
        return t_in_cycle - STANCE_DURATION
    elif t_in_cycle < 2 * STANCE_DURATION + FLIGHT_DURATION:
        return t_in_cycle - (STANCE_DURATION + FLIGHT_DURATION)
    else:
        return t_in_cycle - (2 * STANCE_DURATION + FLIGHT_DURATION)

