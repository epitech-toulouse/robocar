
from control.State import State
from control.Motor import Motor
from control.Gamepad import Gamepad
from .LidarParser import LidarParser

SAFE_DISTANCE = 1.3  # meters
SLOW_DISTANCE = 0.4   # meters
STEER_ANGLE = 0.8    # Max steering
STEER_SMOOTHING = 0.8  # Reduce steering aggressiveness (0.0 to 1.0)
FORWARD_SPEED = 0.06   # Conservative speed
SLOW_SPEED = 0.04
SCAN_FRONT_DEG = 20   # Degrees to scan in front of car
ANGLE_OFFSET = 15 # degrees to offset from center

class AutoDriveState(State):
    def __init__(self):
        print("Initializing AutoDrive...")
        self.lidar = LidarParser()
    
    def stop(self):
        self.lidar.stop()

    def run_single(self, motor : Motor, gamepad : Gamepad):
        points = self.lidar.get_points()
        if not points:
            time.sleep(0.05)
            return
        front_obstacles = self.get_obstacles_in_range(points, -SCAN_FRONT_DEG, SCAN_FRONT_DEG)

        min_dist = 100.0
        closest_obstacle = None
        for ob in front_obstacles:
            if ob['distance'] < min_dist:
                min_dist = ob['distance']
                closest_obstacle = ob
        if min_dist < SAFE_DISTANCE:
            print(f"Obstacle detected! Min dist: {min_dist:.2f}m. Avoiding...")
            
            # Adjust steering based on obstacle angle
            if closest_obstacle:
                obstacle_angle = (closest_obstacle['angle'] + ANGLE_OFFSET) % 360
                if obstacle_angle > 180:
                    obstacle_angle -= 360
                
                steering = -obstacle_angle / SCAN_FRONT_DEG * STEER_ANGLE * STEER_SMOOTHING
                print(f"Steering away from obstacle at angle {closest_obstacle['angle']:.2f}°: Steering set to {steering:.2f}")
                motor.set_steering_objective(steering)
                motor.set_speed_objective(self.get_speed_from_angle(closest_obstacle['angle']))
            else:
                motor.set_steering_objective(0.0)
                motor.set_speed_objective(SLOW_SPEED)
                        
        elif min_dist < SLOW_DISTANCE:
            motor.set_speed_objective(SLOW_SPEED)
            motor.set_steering_objective(0.0)
        else:
            motor.set_speed_objective(FORWARD_SPEED)
            motor.set_steering_objective(0.0)
        

    def get_obstacles_in_range(self, points, min_angle, max_angle):
        obstacles = []
        for p in points:
            angle = (p['angle'] + ANGLE_OFFSET) % 360
            if angle > 180:
                angle -= 360
            if min_angle <= angle <= max_angle:
                obstacles.append(p)
        return obstacles
    

    def get_speed_from_angle(self, angle):
        angle = abs(angle)
        if (angle > 180):
            angle = 360 - angle

    
        # if angle > SCAN_FRONT_DEG:
        #     return FORWARD_SPEED
        # else:
        print(f"Calculating speed for angle {angle:.2f}°")
        angle = abs(angle)
        angle_factor = angle / SCAN_FRONT_DEG
        speed = SLOW_SPEED + (FORWARD_SPEED - SLOW_SPEED) * angle_factor
        print(f"Adjusting speed: {speed:.3f}")
        return speed