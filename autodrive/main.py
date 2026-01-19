"""Main autonomous driving script for Robocar"""

import time
import signal
import sys
from LidarParser import LidarParser
from MotorController import MotorController

# Configuration
SAFE_DISTANCE = 0.8  # meters
SLOW_DISTANCE = 1.5   # meters
STEER_ANGLE = 0.8    # Max steering
FORWARD_SPEED = 0.1   # Conservative speed
SCAN_FRONT_DEG = 30   # Degrees to scan in front of car

class AutoDrive:
    def __init__(self):
        print("Initializing AutoDrive...")
        self.lidar = LidarParser()
        self.motor = MotorController()
        self.running = True
        
        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def shutdown(self, signum, frame):
        print("\nShutting down...")
        self.running = False
        self.motor.stop()
        self.lidar.stop()
        sys.exit(0)

    def get_obstacles_in_range(self, points, min_angle, max_angle):
        obstacles = []
        for p in points:
            # Handle angle wrap around if necessary
            angle = p['angle']
            if angle > 180:
                angle -= 360
            
            if min_angle <= angle <= max_angle:
                obstacles.append(p)
        return obstacles

    def run(self):
        print("AutoDrive started. Press Ctrl+C to stop.")
        while self.running:
            points = self.lidar.get_points()
            if not points:
                time.sleep(0.1)
                continue
                
            # Check front sector
            front_obstacles = self.get_obstacles_in_range(points, -SCAN_FRONT_DEG, SCAN_FRONT_DEG)
            
            min_dist = 100.0
            for ob in front_obstacles:
                if ob['distance'] < min_dist:
                    min_dist = ob['distance']
            
            if min_dist < SAFE_DISTANCE:
                print(f"Obstacle detected! Min dist: {min_dist:.2f}m. Avoiding...")
                # Simple avoidance: check left and right sectors
                left_obs = self.get_obstacles_in_range(points, -90, -SCAN_FRONT_DEG)
                right_obs = self.get_obstacles_in_range(points, SCAN_FRONT_DEG, 90)
                
                min_left = min([ob['distance'] for ob in left_obs]) if left_obs else 100.0
                min_right = min([ob['distance'] for ob in right_obs]) if right_obs else 100.0
                
                if min_left > min_right:
                    print("Turning Left")
                    self.motor.set_steering(-STEER_ANGLE)
                else:
                    print("Turning Right")
                    self.motor.set_steering(STEER_ANGLE)
                
                # Slow down or stop if too close
                if min_dist < SAFE_DISTANCE / 2:
                    self.motor.set_speed(0.0)
                else:
                    self.motor.set_speed(FORWARD_SPEED / 2)
                    
            elif min_dist < SLOW_DISTANCE:
                print(f"Approaching obstacle ({min_dist:.2f}m). Slowing down.")
                self.motor.set_speed(FORWARD_SPEED / 2)
                self.motor.set_steering(0.0)
            else:
                # Path clear
                self.motor.set_speed(FORWARD_SPEED)
                self.motor.set_steering(0.0)
                
            time.sleep(0.05)

if __name__ == "__main__":
    auto = AutoDrive()
    auto.run()
