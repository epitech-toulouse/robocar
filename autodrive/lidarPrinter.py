#print the nearest detected object distance

from LidarParser import LidarParser
import time

lidar = LidarParser()
while True:
    points = lidar.get_points()
    if not points:
        time.sleep(0.05)
        continue
    min_dist = min([p['distance'] for p in points])
    print(f"Nearest object: {min_dist:.2f}m")
    time.sleep(0.05)