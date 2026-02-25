import evdev.ecodes
import evdev
from pyvesc import VESC
import time
import argparse
from enum import Enum
import numpy as np
import subprocess
import re
# import cv2
import gc
from src.motor.Motor import Motor

serial_port = '/dev/ttyACM0'
interval = 0.5

# Main
if __name__ == "__main__":
    devices = []
    while not devices:
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        if not devices:
            print("Waiting for devices...")
            time.sleep(1)
    for device in devices:
        if "F710" in device.name:
            gamepad = evdev.InputDevice(device.path)
            break
    if gamepad is None:
        print("F710 controller not found.")
        exit(1)

    motor = Motor(serial_port)
    running = True
    
    while running:
        R2_value = 0.0
        L2_value = 0.0

        for event in gamepad.read_loop():
            if event.type == evdev.ecodes.EV_ABS:
                abs_event = evdev.categorize(event)
                if event.code == 0:  # Left stick horizontal
                    steering = abs_event.event.value / 32767
                    motor.set_steering_objective(steering)
                elif event.code == 5:  # R2
                    R2_value = abs_event.event.value / 255 / 3
                    valueChanged = True
                elif event.code == 2:  # L2
                    L2_value = abs_event.event.value / 255 / 5
                    valueChanged = True
            speed = R2_value - L2_value
            motor.set_speed_objective(speed)
