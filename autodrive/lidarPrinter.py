#store the nearest point detected on each degree to known the car boundary

from LidarParser import LidarParser
import time

lidar = LidarParser()
car_border_dist = [0.0] * 360
running = True

while running:
    #check if user input command exit to stop program
    if input("Press enter to stop or 'q' to quit: ") == 'q':
        running = False
    #check if user input command to get car border
    if input("Press 'a' to get car border: ") == 'a':
        print(f"Car border: {car_border_dist}")
    points = lidar.get_points()
    if not points:
        time.sleep(0.05)
        continue
    for p in points:
        angle = p['angle']
        distance = p['distance']
        if distance < car_border_dist[angle]:
            car_border_dist[angle] = distance
    print(f"Car border: {car_border_dist}")
    time.sleep(0.05)