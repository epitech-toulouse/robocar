"""Main autonomous driving script for Robocar with Manual Control support"""

import time
import signal
import sys
import evdev
from enum import Enum
from LidarParser import LidarParser
from MotorController import MotorController

# Configuration
SAFE_DISTANCE = 1.5  # meters
SLOW_DISTANCE = 1.5   # meters
STEER_ANGLE = 0.8    # Max steering
FORWARD_SPEED = 0.1   # Conservative speed
SCAN_FRONT_DEG = 30   # Degrees to scan in front of car

class DriveMode(Enum):
    AUTO = 1
    MANUAL = 2

class AutoDrive:
    def __init__(self):
        print("Initializing AutoDrive...")
        self.lidar = LidarParser()
        self.motor = MotorController()
        self.running = True
        self.mode = DriveMode.MANUAL  # Start in manual mode for safety
        self.gamepad = self.setup_gamepad()
        
        # Manual control values
        self.manual_speed = 0.0
        self.manual_steering = 0.0
        self.r2_value = 0.0
        self.l2_value = 0.0
        
        # Light support (attempt to import Jetson.GPIO)
        self.gpio = None
        self.light_toggled = False
        self.light_channels = [36, 38]
        try:
            import Jetson.GPIO as GPIO
            self.gpio = GPIO
            self.gpio.setmode(self.gpio.BOARD)
            self.gpio.setup(self.light_channels, self.gpio.OUT)
            print("Jetson.GPIO initialized for lights.")
        except (ImportError, RuntimeError):
            print("Jetson.GPIO not available. Lights disabled.")

        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def setup_gamepad(self):
        print("Searching for F710 controller...")
        try:
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            for device in devices:
                if "F710" in device.name:
                    print(f"Found controller: {device.name} at {device.path}")
                    return evdev.InputDevice(device.path)
        except Exception as e:
            print(f"Error searching for gamepad: {e}")
        
        print("F710 controller not found. Running in AUTO mode only.")
        self.mode = DriveMode.AUTO
        return None

    def toggle_light(self):
        if not self.gpio:
            return
        self.light_toggled = not self.light_toggled
        state = self.gpio.HIGH if self.light_toggled else self.gpio.LOW
        self.gpio.output(self.light_channels, state)
        print(f"Lights {'ON' if self.light_toggled else 'OFF'}")

    def shutdown(self, signum, frame):
        print("\nShutting down...")
        self.running = False
        self.motor.stop()
        self.lidar.stop()
        if self.gpio:
            self.gpio.cleanup()
        sys.exit(0)

    def process_gamepad_events(self):
        if not self.gamepad:
            return

        try:
            # Read all available events
            while True:
                event = self.gamepad.read_one()
                if event is None:
                    break
                
                # Mode toggle: South button (A on Xbox layout)
                if event.type == evdev.ecodes.EV_KEY:
                    if event.code == evdev.ecodes.BTN_SOUTH and event.value == 1:
                        if self.mode == DriveMode.AUTO:
                            self.mode = DriveMode.MANUAL
                            print("Switched to MANUAL mode")
                            self.motor.set_speed(0)
                        else:
                            self.mode = DriveMode.AUTO
                            print("Switched to AUTO mode")
                    
                    # Light toggle: North button (Y on Xbox layout)
                    elif event.code == evdev.ecodes.BTN_NORTH and event.value == 1:
                        self.toggle_light()
                
                # Manual control inputs (only process if in MANUAL mode)
                if self.mode == DriveMode.MANUAL:
                    if event.type == evdev.ecodes.EV_ABS:
                        abs_event = evdev.categorize(event)
                        if event.code == 0:  # Left stick horizontal (Steering)
                            self.manual_steering = abs_event.event.value / 32767
                            self.motor.set_steering(self.manual_steering)
                        elif event.code == 5:  # R2 (Forward)
                            self.r2_value = abs_event.event.value / 255 / 3
                            self.update_manual_speed()
                        elif event.code == 2:  # L2 (Backward)
                            self.l2_value = abs_event.event.value / 255 / 5
                            self.update_manual_speed()
        except Exception as e:
            print(f"Error reading gamepad: {e}")

    def update_manual_speed(self):
        self.manual_speed = self.r2_value - self.l2_value
        self.motor.set_speed(self.manual_speed)

    def get_obstacles_in_range(self, points, min_angle, max_angle):
        obstacles = []
        for p in points:
            angle = p['angle']
            if angle > 180:
                angle -= 360
            if min_angle <= angle <= max_angle:
                obstacles.append(p)
        return obstacles

    def run(self):
        print(f"AutoDrive started in {self.mode.name} mode.")
        print("Controls: 'A' to toggle AUTO/MANUAL, 'Y' to toggle LIGHTS (if available).")
        print("In MANUAL: Left Stick = Steering, R2 = Forward, L2 = Backward.")
        print("Press Ctrl+C to stop.")
        
        while self.running:
            self.process_gamepad_events()
            
            if self.mode == DriveMode.AUTO:
                points = self.lidar.get_points()
                if not points:
                    time.sleep(0.05)
                    continue
                    
                # Check front sector
                front_obstacles = self.get_obstacles_in_range(points, -SCAN_FRONT_DEG, SCAN_FRONT_DEG)
                
                min_dist = 100.0
                for ob in front_obstacles:
                    if ob['distance'] < min_dist:
                        min_dist = ob['distance']
                
                if min_dist < SAFE_DISTANCE:
                    print(f"Obstacle detected! Min dist: {min_dist:.2f}m. Avoiding...")
                    left_obs = self.get_obstacles_in_range(points, -90, -SCAN_FRONT_DEG)
                    right_obs = self.get_obstacles_in_range(points, SCAN_FRONT_DEG, 90)
                    
                    min_left = min([ob['distance'] for ob in left_obs]) if left_obs else 100.0
                    min_right = min([ob['distance'] for ob in right_obs]) if right_obs else 100.0
                    
                    if min_left > min_right:
                        self.motor.set_steering(-STEER_ANGLE)
                    else:
                        self.motor.set_steering(STEER_ANGLE)
                    
                    if min_dist < SAFE_DISTANCE / 2:
                        self.motor.set_speed(0.0)
                    else:
                        self.motor.set_speed(FORWARD_SPEED / 2)
                        
                elif min_dist < SLOW_DISTANCE:
                    self.motor.set_speed(FORWARD_SPEED / 2)
                    self.motor.set_steering(0.0)
                else:
                    self.motor.set_speed(FORWARD_SPEED)
                    self.motor.set_steering(0.0)
            
            time.sleep(0.02)

if __name__ == "__main__":
    auto = AutoDrive()
    auto.run()
