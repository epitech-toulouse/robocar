
from Gamepad import Gamepad
from Motor import Motor
from Logger import Logger
from State import State
from ManualState import ManualState

class Manager:
    def __init__(self, State state = None):
        self.logger = Logger("Manager")
        self.logger.log("Starting initialisation.")
        self.manual_state = ManualState
        self.other_state = state
        self.state = self.manual_state
        self.gamepad = Gamepad()
        self.motor = Motor("/dev/ttyACM0", 0.4, -0.2)
        self.logger.log("Init done.")
        self.running = True
        self.take_manual_control()
        self.loop()

    def take_manual_control(self):
        self.gamepad.setLedsOn()
        self.state = self.manual_state

    def switch_other_state(self):
        if (self.other_state == None)
            return
        self.gamepad.setLedsOff()
        self.state = self.other_state

    def urgent_stop(self):
        self.motor.urgent_stop()
        self.gamepad.setLedsOn()
        self.running = False

    def stop(self):
        self.motor.stop()
        self.take_manual_control()

    def loop(self):
        while (self.running):
            self.gamepad.updateEvents()
            if self.gamepad.getButton("B"):
                self.urgent_stop()
                continue
            if self.gamepad.getButton("X"):
                self.stop()
                continue
            if self.gamepad.getButton("Y"):
                self.switch_other_state()
                continue
            if self.gamepad.getButton("A"):
                self.take_manual_control()
                continue
            self.state.run_single(self.gamepad)
