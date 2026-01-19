import evdev
import time
from .Logger import Logger

class Gamepad:
    def __init__(self):
        self.logger : Logger = Logger("Gamepad")
        self.logger.log("Starting initialisation.")
        self.device = None
        while (self.device is None):
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            for device in devices:
                if "F710" in device.name:
                    self.device = evdev.InputDevice(device.path)
            if (self.device is None):
                self.logger.log("Waiting for devices...")
                time.sleep(1)
        self.logger.log("Found device.", self.device)

        self.leds = self.device.leds()

        self.axis = {
            "steering": 0.0,
            "forward": 0.0,
            "backward": 0.0
        }

        self.buttons = {
            "A": False,
            "X": False,
            "Y": False,
            "B": False,
        }

        self.logger.log("Init done.")

    def setLedsOn(self):
        for led in self.leds:
            self.device.set_led(led, 1)

    def setLedsOff(self):
        for led in self.leds:
            self.device.set_led(led, 0)

    def getAxis(self, axis: str) -> float:
        return self.axis[axis]

    def getButton(self, button: str) -> bool:
        return self.buttons[button]

    def updateEvents(self):
        for button in self.buttons:
            self.buttons[button] = False
        value = self.device.read_one()
        while (value != None):
            # If axis
            if value.type == evdev.ecodes.EV_ABS:
                abs_event = evdev.categorize(value)
                if value.code == 0:
                    self.axis["steering"] = abs_event.event.value / 32767
                elif value.code == 5:
                    self.axis["forward"] = abs_event.event.value / 255 / 3
                elif value.code == 2:
                    self.axis["backward"] = abs_event.event.value / 255 / 5
            # If button
            elif value.type == evdev.ecodes.BTN_SOUTH:
                self.buttons["A"] = True
            elif value.type == evdev.ecodes.BTN_WEST:
                self.buttons["X"] = True
            elif value.type == evdev.ecodes.BTN_NORTH:
                self.buttons["Y"] = True
            elif value.type == evdev.ecodes.BTN_EAST:
                self.buttons["B"] = True
            value = self.device.read_one()
