#store the nearest point detected on each degree to known the car boundary

from LidarParser import LidarParser
import time

lidar = LidarParser()
car_border_dist = [0.0] * 360
running = True

while running:
    points = lidar.get_points()
    if not points:
        time.sleep(0.05)
        continue
    for p in points:
        angle = p['angle']
        angle = (angle + ANGLE_OFFSET) % 360
        distance = p['distance']
        if distance < car_border_dist[angle]:
            car_border_dist[angle] = distance
    print(f"Car border: {car_border_dist}")
    time.sleep(0.05)