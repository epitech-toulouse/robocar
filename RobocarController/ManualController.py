import evdev.ecodes
import evdev
import time
from enum import Enum

class RequestChangeGamemode(Enum):
    CONTROLLER = 1
    AI = 2

class ManualController:
    gamepad = None

    @classmethod
    def lookForDevice(cls):
        print("Looking for device ...")
        devices = []
        while not devices:
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            if not devices:
                print("Waiting for devices...")
                time.sleep(1)
        for device in devices:
            if "F710" in device.name:
                cls.gamepad = evdev.InputDevice(device.path)
                break
        if cls.gamepad is None:
            print("F710 controller not found.")
            return

    @classmethod
    def runManualControlLoop(cls):
        R2_value = 0.0
        L2_value = 0.0

        if cls.gamepad is None:
            print("No gamepad available. Call lookForDevice() first.")
            return

        for event in cls.gamepad.read_loop():
            if event.code == evdev.ecodes.BTN_SOUTH and event.value == 1: #TODO better control mode change
                return RequestChangeGamemode.AI
            if event.type == evdev.ecodes.EV_ABS:
                abs_event = evdev.categorize(event)

                if event.code == 0:  # Left stick horizontal
                    Motor.setSteeringObjective(abs_event.event.value / 32767)
                elif event.code == 5:  # R2
                    R2_value = abs_event.event.value / 255 / 3
                elif event.code == 2:  # L2
                    L2_value = abs_event.event.value / 255 / 5

                Motor.setSpeedObjective(R2_value - L2_value)