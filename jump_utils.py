import numpy as np

CROUCH = 0
TAKEOFF = 1
FLIGHT = 2
LANDING = 3

def get_fsm(t: float) -> int:
    t_in_cycle = t % JUMP_CYCLE_DURATION
    if t_in_cycle < CROUCH_DURATION:
        return CROUCH
    elif t_in_cycle < CROUCH_DURATION + TAKEOFF_DURATION:
        return TAKEOFF
    elif t_in_cycle < CROUCH_DURATION + TAKEOFF_DURATION + FLIGHT_DURATION:
        return FLIGHT
    else:
        return LANDING

def time_until_switch(t: float) -> float:
    t_in_cycle = t % JUMP_CYCLE_DURATION
    if t_in_cycle < CROUCH_DURATION:
        # In CROUCH phase, time until TAKEOFF
        return CROUCH_DURATION - t_in_cycle
    elif t_in_cycle < CROUCH_DURATION + TAKEOFF_DURATION:
        # In TAKEOFF phase, time until FLIGHT
        return (CROUCH_DURATION + TAKEOFF_DURATION) - t_in_cycle
    elif t_in_cycle < CROUCH_DURATION + TAKEOFF_DURATION + FLIGHT_DURATION:
        # In FLIGHT phase, time until LANDING
        return (CROUCH_DURATION + TAKEOFF_DURATION + FLIGHT_DURATION) - t_in_cycle
    else:
        # In LANDING phase, time until next cycle starts (CROUCH)
        return JUMP_CYCLE_DURATION - t_in_cycle

def time_since_switch(t: float) -> float:
    t_in_cycle = t % JUMP_CYCLE_DURATION
    if t_in_cycle < CROUCH_DURATION:
        # In CROUCH phase, time since CROUCH started
        return t_in_cycle
    elif t_in_cycle < CROUCH_DURATION + TAKEOFF_DURATION:
        # In TAKEOFF phase, time since TAKEOFF started
        return t_in_cycle - CROUCH_DURATION
    elif t_in_cycle < CROUCH_DURATION + TAKEOFF_DURATION + FLIGHT_DURATION:
        # In FLIGHT phase, time since FLIGHT started
        return t_in_cycle - (CROUCH_DURATION + TAKEOFF_DURATION)
    else:
        # In LANDING phase, time since LANDING started
        return t_in_cycle - (CROUCH_DURATION + TAKEOFF_DURATION + FLIGHT_DURATION)


CROUCH_DURATION = 0.5  # seconds
TAKEOFF_DURATION = 0.1
FLIGHT_DURATION = 0.5
LANDING_DURATION = 0.2
JUMP_CYCLE_DURATION = CROUCH_DURATION + TAKEOFF_DURATION + FLIGHT_DURATION + LANDING_DURATION