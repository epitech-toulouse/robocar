
import time
from .Gamepad import Gamepad
from .Motor import Motor
from .Logger import Logger
from .State import State
from .ManualState import ManualState

class Manager:
    def __init__(self, state = None):
        self.logger = Logger("Manager")
        self.logger.log("Starting initialisation.")
        self.manual_state = ManualState()
        self.other_state = state
        self.state = self.manual_state
        self.gamepad = Gamepad()
        self.motor = Motor(["/dev/ttyACM0", "/dev/ttyACM1"], 0.4, -0.2)
        self.logger.log("Init done.")
        self.running = True
        self.take_manual_control()

    def take_manual_control(self):
        self.logger.log("Took manual control.")
        self.gamepad.setLedsOn()
        self.state = self.manual_state

    def switch_other_state(self):
        self.logger.log("Try to switch to other state.")
        if (self.other_state == None):
            return
        self.logger.log("Switch to other state.")
        self.gamepad.setLedsOff()
        self.state = self.other_state

    def urgent_stop(self):
        self.logger.log("Stopping urgently.")
        self.gamepad.setLedsOn()
        self.motor.urgent_stop()
        self.running = False

    def stop(self):
        self.logger.log("Stopping.")
        self.motor.stop()
        self.take_manual_control()

    def loop(self):
        self.logger.log("Waiting for Start call.")
        self.gamepad.updateEvents()
        while not self.gamepad.getButton("Start"):
            time.sleep(0.1)
            self.gamepad.updateEvents()
        self.logger.log("Starting !")
        while (self.running):
            self.gamepad.updateEvents()
            if self.gamepad.getButton("B"):
                self.urgent_stop()
                continue
            if self.gamepad.getButton("RB"):
                self.stop()
                continue
            if self.gamepad.getButton("X"):
                self.switch_other_state()
                continue
            if self.gamepad.getButton("A"):
                self.take_manual_control()
                continue
            self.state.run_single(self.motor, self.gamepad)
        self.logger.log("End of loop.")
        self.safe_stop()
        # self.motor.join()
    
    def safe_stop(self):
        self.motor.stop()
        self.manual_state.stop()
        if self.other_state != None:
            self.other_state.stop()
