import numpy as np

# Global variables for gait timing
LEFT_STANCE = 0
RIGHT_STANCE = 1
FLYING = 2
STANCE_DURATION = 0.2


def get_fsm(t: float) -> int:
    '''TO DO:Adjust HARD CODED VALUES'''
    if t % 2 < 0.5:
        return LEFT_STANCE
    elif t % 2 < 1.5:
        return RIGHT_STANCE
    else:
        return FLYING

def time_until_switch(t: float) -> float:
    '''TO DO:Adjust HARD CODED VALUES'''
    if t % 2 < 0.5:
        return 0.5 - (t % 0.5)
    elif t % 2 < 1.5:
        return 1.5 - (t % 1.5)
    else:
        return 2.0 - (t % 2.0)

def time_since_switch(t: float) -> float:
    '''TO DO:Adjust HARD CODED VALUES'''
    if t % 2 < 0.5:
        return t % 0.5
    elif t % 2 < 1.5:
        return t % 1.5 - 0.5
    else:
        return t % 2.0 - 1.5

