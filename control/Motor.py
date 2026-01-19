"""The motor class"""

import threading
from pyvesc import VESC
from Motor import Motor
import time

class Motor:
    def __init__(self, serial_port : str, max_speed = 0.4, max_speed = -0.2) -> None:
        self.logger : Logger = Logger("Motor")
        self.logger.log("Starting initialisation.")
        self.vesc = None
        while (self.vesc == None):
            try:
                self.vesc : VESC = VESC(serial_port=serial_port)
            except:
                self.logger.log("Waiting for VESC...")
                time.sleep(1)
        self.logger.log("Motor controler connected.")
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.target_speed : float = 0.0
        self.speed : float = 0.0
        self.running : bool = True
        self.steering : float = 0.0
        self.logger.log("Init done.")
        self.thread = threading.Thread(target=self.__loop__)
        self.lock = threading.Lock()
        self.thread.start()

    def __loop__(self) -> None:
        self.target_speed = 0.0
        self.speed = 0.0
        self.running = True
        while self.running or self.speed != 0.0:
            self.lock.acquire()
            if self.target_speed > self.max_speed:
                self.target_speed = self.max_speed
            if self.target_speed < self.min_speed:
                self.target_speed = self.min_speed
            if not self.running:
                self.target_speed = 0.0
            if self.speed < self.target_speed:
                self.speed += 0.02
                if self.speed > self.target_speed:
                    self.speed = self.target_speed
            elif self.speed > self.target_speed:
                self.speed -= 0.05
                if self.speed < self.target_speed:
                    self.speed = self.target_speed
            self.lock.release()
            self.vesc.set_duty_cycle(self.speed)
            self.vesc.set_servo((self.steering + 1) / 2)

    def set_steering_objective(self, steering : float) -> None:
        """steering is a number between -1 and 1."""
        self.lock.acquire()
        self.steering = steering
        self.lock.release()

    def set_speed_objective(self, speed_obj : float) -> None:
        """speed_obj is a number between -1 and 1. Please do not go passed 0.3 !"""
        self.lock.acquire()
        self.target_speed = speed_obj
        self.lock.release()

    def urgent_stop(self) -> None:
        self.lock.acquire()
        self.target_speed = 0.0
        self.speed = 0.0
        self.running = False
        self.lock.release()

    def stop(self) -> None:
        self.lock.acquire()
        self.running = False
        self.lock.release()
