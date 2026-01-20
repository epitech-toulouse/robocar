
from control.State import State
from control.Motor import Motor
from control.Gamepad import Gamepad
from .LidarParser import LidarParser


# DIRECTION DRIVE PARAMETERS

SCAN_FRONT_DEG = 10   # Degrees to scan in front of car
SAFE_DISTANCE = 1.3  # meters
SLOW_DISTANCE = 0.4   # meters
FORWARD_SPEED = 0.06   # Conservative speed
BACKWARD_SPEED = -0.02  # Reverse speed
STOP_DISTANCE = 0.2    # meters
SLOW_SPEED = 0.04


# STREERING AVOIDANCE PARAMETERS
STERRING_SCAN_FRONT_DEG = 20   # Degrees to scan in front of car
STERRING_SCAN_DISTANCE = 50  # cm
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

        min_dist = STERRING_SCAN_DISTANCE 
        closest_obstacle = None
        for ob in front_obstacles:
            if ob['distance'] < min_dist:
                min_dist = ob['distance']
                closest_obstacle = ob
        
        if min_dist < STERRING_SCAN_DISTANCE:
            print(f"Obstacle detected! Min dist: {min_dist:.2f}m. Avoiding...")
            
            if closest_obstacle:
                obstacle_angle = closest_obstacle['angle'] % 360
                if obstacle_angle > 180:
                    obstacle_angle -= 360
                
                steering = -obstacle_angle / STERRING_SCAN_FRONT_DEG * STEER_ANGLE * STEER_SMOOTHING
                print(f"Steering away from obstacle at angle {closest_obstacle['angle']:.2f}°: Steering set to {steering:.2f}")
                motor.set_steering_objective(steering)
            else:
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
    

                        
      
