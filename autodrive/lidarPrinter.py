#store the nearest point detected on each degree to known the car boundary

from LidarParser import LidarParser
import time

lidar = LidarParser()
running = True

lidar_dir = {}

while running:
    points = lidar.get_points()
    if not points:
        time.sleep(0.05)
        continue
    print(points)
    for p in points:
        angle = p['angle']
        distance = p['distance']
        if (lidar_dir.get(angle) is None):
            lidar_dir[angle] = distance
        if (lidar_dir[angle] > distance):
            lidar_dir[angle] = distance
    print(f"Car border: {lidar_dir}")
    time.sleep(0.05)
