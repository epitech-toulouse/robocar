import math
import time
import numpy as np
from control.State import State
from control.Motor import Motor
from control.Gamepad import Gamepad
from .LidarParser import LidarParser

# --- FTG CONFIGURATION ---
FOV_DEG = 160            # Field of view (+/- 80 degrees)
BUBBLE_RADIUS = 0.50     # Safety bubble radius in meters (inflates obstacles)
MAX_LIDAR_DIST = 4.0     # Cap distance for normalization
SAFE_THRESHOLD = 0.5     # Threshold to consider a point as "free space"

DRIVE_SPEED_MAX = 0.10
DRIVE_SPEED_MIN = 0.02
STEERING_SENSITIVITY = 1.0 


class AutoDriveState(State):
    def __init__(self):
        print("Initializing AutoDrive FTG...")
        self.lidar = LidarParser()
        
    def stop(self):
        self.lidar.stop()

    def run_single(self, motor : Motor, gamepad : Gamepad):
        points = self.lidar.get_points()
        if not points:
            time.sleep(0.01)
            return

        # Get points
        ranges, angles = self.preprocess_scan(points)
        
        if len(ranges) == 0:
            return

        # Safety Bubble
        proc_ranges = self.apply_safety_bubble(ranges, angles)

        # Find Max Gap
        start_i, end_i = self.find_max_gap(proc_ranges)

        # Find Goal Point
        goal_idx = self.find_best_goal(start_i, end_i, proc_ranges)
        
        goal_angle = angles[goal_idx] if goal_idx is not None else 0.0
        # goal_dist = proc_ranges[goal_idx] if goal_idx is not None else 0.0

        # Actuate
        self.apply_command(motor, goal_angle)


    def preprocess_scan(self, points):
        """ Filters points to FOV and converts to sorted numpy arrays """
        raw_angles = []
        raw_ranges = []
        
        for p in points:
            angle = p['angle']
            dist = p['distance']
            
            if angle > 180:
                angle -= 360
            
            if -FOV_DEG/2 <= angle <= FOV_DEG/2:
                if dist <= 0 or dist > MAX_LIDAR_DIST:
                    dist = MAX_LIDAR_DIST
                
                raw_angles.append(np.radians(angle))
                raw_ranges.append(dist)
        
        raw_angles = np.array(raw_angles)
        raw_ranges = np.array(raw_ranges)
        
        if len(raw_angles) > 0:
            sort_idx = np.argsort(raw_angles)
            return raw_ranges[sort_idx], raw_angles[sort_idx]
        return np.array([]), np.array([])

    def apply_safety_bubble(self, ranges, angles):
        """ Zeroes out ranges near the closest obstacle """
        if len(ranges) == 0:
            return ranges
            
        proc_ranges = np.copy(ranges)
        
        # Find closest point index
        min_idx = np.argmin(proc_ranges)
        min_dist = proc_ranges[min_idx]
        
        # Only create bubble if obstacle is close enough to matter
        if min_dist < MAX_LIDAR_DIST:
            r = max(min_dist, 0.001)
            proj_angle = np.arctan(BUBBLE_RADIUS / r) # in radians
            
            min_angle = angles[min_idx]
            
            # Mask angles within the bubble radius
            angle_diff = np.abs(angles - min_angle)
            mask = angle_diff < proj_angle
            
            # Set danger zone to 0.0 (treat as obstacle)
            proc_ranges[mask] = 0.0
            
        return proc_ranges

    def find_max_gap(self, ranges):
        """ Finds the start and end index of the longest consecutive run of non-zero ranges """
        # Mask where range is acceptable (non-zero or above threshold)
        mask = ranges > SAFE_THRESHOLD
        
        max_len = 0
        current_len = 0
        start_i = 0
        
        best_start = 0
        best_end = 0
        
        for i, val in enumerate(mask):
            if val: # is gap
                if current_len == 0:
                    start_i = i
                current_len += 1
            else: # gap ended
                if current_len > max_len:
                    max_len = current_len
                    best_start = start_i
                    best_end = i
                current_len = 0
                
        if current_len > max_len:
             best_start = start_i
             best_end = len(mask)
             
        if max_len == 0:
            return None, None
            
        return best_start, best_end

    def find_best_goal(self, start_i, end_i, ranges):
        """ Finds the furthest (deepest) point within the selected gap """
        if start_i is None or end_i is None:
            return None
            
        gap_slice = ranges[start_i:end_i]
        if len(gap_slice) == 0:
            return None
            
        argmax = np.argmax(gap_slice)
        return start_i + argmax

    def apply_command(self, motor, goal_angle_rad):
        steer_angle_deg = np.degrees(goal_angle_rad)
        
        steering = np.clip(steer_angle_deg / 30.0, -1.0, 1.0)
        
        speed = DRIVE_SPEED_MAX - (abs(steering) * (DRIVE_SPEED_MAX - DRIVE_SPEED_MIN))
        speed = max(speed, DRIVE_SPEED_MIN)
        print("Steering: {:.2f}, Speed: {:.3f}".format(steering, speed))
        motor.set_steering_objective(steering)
        motor.set_speed_objective(speed)
