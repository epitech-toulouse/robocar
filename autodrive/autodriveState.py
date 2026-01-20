
from control.State import State
from control.Motor import Motor
from control.Gamepad import Gamepad
from .LidarParser import LidarParser


# DIRECTION DRIVE PARAMETERS

SCAN_FRONT_DEG = 10   # Degrees to scan in front of car
SAFE_DISTANCE = 2.0  # meters
SLOW_DISTANCE = 1.4   # meters
STOP_DISTANCE = 0.5    # meters

FORWARD_SPEED = 0.1   # Conservative speed
BACKWARD_SPEED = -0.03  # Reverse speed
SLOW_SPEED = 0.05


# STREERING AVOIDANCE PARAMETERS
STERRING_SCAN_FRONT_DEG = 30   # Degrees to scan in front of car
STERRING_SCAN_DISTANCE = 0.5  # meters
STEER_ANGLE = 0.8    # Max steering
STEER_SMOOTHING = 0.8  # Reduce steering aggressiveness (0.0 to 1.0)




class AutoDriveState(State):
    def __init__(self):
        print("Initializing AutoDrive...")
        self.lidar = LidarParser()
    
    def stop(self):
        self.lidar.stop()

    def run_single(self, motor : Motor, gamepad : Gamepad):
        self.direction_drive(motor, gamepad)
        self.speed_drive(motor, gamepad)


    def get_obstacles_in_range(self, points, min_angle, max_angle):
        obstacles = []
        for p in points:
            angle = (p['angle']) % 360
            if angle > 180:
                angle -= 360
            if min_angle <= angle <= max_angle:
                obstacles.append(p)
        return obstacles
    




    def direction_drive(self, motor : Motor, gamepad : Gamepad):
           
        points = self.lidar.get_points()
        if not points:
            time.sleep(0.05)
            return
        front_obstacles = self.get_obstacles_in_range(points, -STERRING_SCAN_FRONT_DEG, STERRING_SCAN_FRONT_DEG)

        min_dist = float('inf') 
        closest_obstacle = None
        for ob in front_obstacles:
            if ob['distance'] < min_dist:
                min_dist = ob['distance']
                closest_obstacle = ob
        
        if min_dist < STERRING_SCAN_DISTANCE:
            
            if closest_obstacle:
                angle = closest_obstacle['angle'] % 360
                if (angle > 180):
                    angle = 360 - angle # Normalize to [-30, 30]

                print(f"Closest obstacle at angle {angle:.2f}° and distance {min_dist:.2f}m")
                
                distance_factor = 1.0 - (min_dist / STERRING_SCAN_DISTANCE)
                distance_factor = max(0.0, min(1.0, distance_factor))
                
                angle_factor = angle / STERRING_SCAN_FRONT_DEG
                
                steering = -angle_factor * STEER_ANGLE * STEER_SMOOTHING * distance_factor
                
                print(f"Steering away from obstacle at angle {angle:.2f}° (dist: {min_dist:.2f}m): Steering set to {steering:.2f}")
                motor.set_steering_objective(steering)
            else:
                print("No closest obstacle found despite min_dist < STERRING_SCAN_DISTANCE")
                motor.set_steering_objective(0.0)


    
    def speed_drive(self, motor : Motor, gamepad : Gamepad):   
        points = self.lidar.get_points()
        if not points:
            time.sleep(0.05)
            return
     

        front_obstacles = self.get_obstacles_in_range(points, -SCAN_FRONT_DEG, SCAN_FRONT_DEG)

        closest_obstacle = None
        min_dist = float('inf') 
        for ob in front_obstacles:
            if ob['distance'] < min_dist:
                min_dist = ob['distance']
                closest_obstacle = ob

        if min_dist < STOP_DISTANCE:
            print(f"Too close to obstacle! Min dist: {min_dist:.2f}m. Reversing...")
            motor.set_speed_objective(BACKWARD_SPEED)
            return
        
        if closest_obstacle and min_dist < SAFE_DISTANCE:
            motor.set_speed_objective(self.get_speed_from_angle(closest_obstacle['angle']))
        else:
            motor.set_speed_objective(FORWARD_SPEED)
    
    def get_speed_from_angle(self, angle):
        angle = abs(angle)
        if (angle > 180):
            angle = 360 - angle

        print(f"Calculating speed for angle {angle:.2f}°")
        angle = abs(angle)
        angle_factor = angle / SCAN_FRONT_DEG
        speed = SLOW_SPEED + (FORWARD_SPEED - SLOW_SPEED) * angle_factor
        print(f"Adjusting speed: {speed:.3f}")
        return speed
    

                        
      
