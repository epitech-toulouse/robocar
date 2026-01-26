import evdev
import time
from .Logger import Logger

NORMAL_SPEED : float = 0.2
MANI_SPEED : float = 0.4

class Gamepad:
    def __init__(self):
        self.logger : Logger = Logger("Gamepad")
        self.logger.log("Starting initialisation.")
        self.device = None
        self.max_speed_ratio : float = NORMAL_SPEED
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
            "Select": False,
            "Start": False,
            "LB": False,
            "RB": False,
        }

        self.logger.log("Init done.")

    def setManiSpeed(self):
        self.max_speed_ratio = MANI_SPEED

    def setNormalSpeed(self):
        self.max_speed_ratio = NORMAL_SPEED

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
                    self.axis["forward"] = abs_event.event.value / 255 * self.max_speed_ratio
                elif value.code == 2:
                    self.axis["backward"] = abs_event.event.value / 255 * self.max_speed_ratio / 2
            # If button
            elif value.code == evdev.ecodes.BTN_A:
                self.buttons["A"] = True
            elif value.code == evdev.ecodes.BTN_X:
                self.buttons["X"] = True
            elif value.code == evdev.ecodes.BTN_Y:
                self.buttons["Y"] = True
            elif value.code == evdev.ecodes.BTN_B:
                self.buttons["B"] = True
            elif value.code == evdev.ecodes.BTN_SELECT:
                self.buttons["Select"] = True
            elif value.code == evdev.ecodes.BTN_START:
                self.buttons["Start"] = True
            elif value.code == evdev.ecodes.BTN_TL:
                self.buttons["LB"] = True
            elif value.code == evdev.ecodes.BTN_TR:
                self.buttons["RB"] = True
            for key, item in evdev.ecodes.__dict__.items():
                if not "BTN_" in key:
                    continue
                if item == value.code:
                    print("PRESSED: ", key)
            value = self.device.read_one()
