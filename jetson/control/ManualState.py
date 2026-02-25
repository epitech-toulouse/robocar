
from .State import State
from .Motor import Motor
from .Gamepad import Gamepad

class ManualState(State):
    def __init__(self):
        pass

    def run_single(self, motor : Motor, gamepad : Gamepad):
        steering = gamepad.getAxis("steering")
        forward = gamepad.getAxis("forward")
        backward = gamepad.getAxis("backward")
        speed = forward - backward

        motor.set_steering_objective(steering)
        motor.set_speed_objective(speed)
    
    def stop(self):
        pass
