
from control.State import State
from control.Motor import Motor
from control.Gamepad import Gamepad
from .LidarParser import LidarParser

SAFE_DISTANCE = 1.3  # meters
SLOW_DISTANCE = 0.4   # meters
STEER_ANGLE = 0.8    # Max steering
FORWARD_SPEED = 0.06   # Conservative speed
SLOW_SPEED = 0.04
SCAN_FRONT_DEG = 20   # Degrees to scan in front of car
ANGLE_OFFSET = 15 # degrees to offset from center

class AutoDriveState(State):
    def __init__(self):
        print("Initializing AutoDrive...")
        self.lidar = LidarParser()

    def run_single(motor : Motor, gamepad : Gamepad):
        points = self.lidar.get_points()
        if not points:
            time.sleep(0.05)
            return
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
                self.motor.set_steering_objective(-STEER_ANGLE)
            else:
                self.motor.set_steering_objective(STEER_ANGLE)
            
            if min_dist < SAFE_DISTANCE / 2:
                self.motor.set_speed_objective(0.0)
            else:
                self.motor.set_speed_objective(SLOW_SPEED)
                        
        elif min_dist < SLOW_DISTANCE:
            self.motor.set_speed_objective(SLOW_SPEED)
            self.motor.set_steering_objective(0.0)
        else:
            self.motor.set_speed_objective(FORWARD_SPEED)
            self.motor.set_steering_objective(0.0)
        

    def get_obstacles_in_range(self, points, min_angle, max_angle):
        obstacles = []
        for p in points:
            angle = (p['angle'] + angle_offset) % 360
            if angle > 180:
                angle -= 360
            if min_angle <= angle <= max_angle:
                obstacles.append(p)
        return obstacles
