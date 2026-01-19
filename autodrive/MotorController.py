"""Motor controller for Robocar using pyvesc"""

import threading
import time
from pyvesc import VESC

class MotorController:
    def __init__(self, serial_port='/dev/ttyACM0') -> None:
        self.vesc = VESC(serial_port=serial_port)
        self.target_speed = 0.0
        self.speed = 0.0
        self.steering = 0.0
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.__loop__, daemon=True)
        self.thread.start()

    def __loop__(self) -> None:
        while self.running:
            with self.lock:
                # Smooth speed transitions
                if self.speed < self.target_speed:
                    self.speed += 0.02
                    if self.speed > self.target_speed:
                        self.speed = self.target_speed
                elif self.speed > self.target_speed:
                    self.speed -= 0.05
                    if self.speed < self.target_speed:
                        self.speed = self.target_speed
                
                current_speed = self.speed
                current_steering = self.steering
            
            self.vesc.set_duty_cycle(current_speed)
            # Steering mapping: -1.0 (left) to 1.0 (right) -> 1.0 to 0.0 for servo
            self.vesc.set_servo((-current_steering + 1) / 2)
            time.sleep(0.02)

    def set_steering(self, steering: float) -> None:
        """Set steering between -1.0 (left) and 1.0 (right)."""
        with self.lock:
            self.steering = max(-1.0, min(1.0, steering))

    def set_speed(self, speed: float) -> None:
        """Set speed between -1.0 and 1.0. Recommendation: keep below 0.3."""
        with self.lock:
            self.target_speed = max(-1.0, min(1.0, speed))

    def stop(self) -> None:
        """Stop the motor and the control thread."""
        with self.lock:
            self.target_speed = 0.0
            self.speed = 0.0
            self.running = False
        # Give it a moment to send the stop signal
        time.sleep(0.1)
        self.vesc.set_duty_cycle(0)
