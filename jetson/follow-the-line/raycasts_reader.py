import math
import numpy as np
from PIL import Image

DEG2RAD = lambda angle: angle * math.pi / 180.0
OBSTACLE_THRESHOLD = 250

def calc_raycast_angles(number_of_rays, max_angle):
    step = max_angle / (number_of_rays - 1)
    start_angle = -max_angle / 2
    return [start_angle + i * step for i in range(number_of_rays)]

def is_obstacle(img, x, y):
    h, w, _ = img.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return True
    r, g, b = img[y, x]
    intensity = (int(r) + int(g) + int(b)) // 3
    if intensity <= 200:
        img[y, x] = [0, 0, 255]
    return intensity > 200

def cast_ray(img, ox, oy, angle, max_distance):
    dx = math.cos(DEG2RAD(angle - 90))
    dy = math.sin(DEG2RAD(angle - 90))
    for step in range(max_distance):
        x = int(ox + dx * step)
        y = int(oy + dy * step)
        if is_obstacle(img, x, y):
            return (math.sqrt((x - ox) ** 2 + (y - oy) ** 2)) * 0.8
    return -1.0

def raycast_image(image_path, number_of_rays=10, max_angle=180.0):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    h, w, _ = img_np.shape
    origin_x = w // 2
    origin_y = h - 10

    angles = calc_raycast_angles(number_of_rays, max_angle)
    return [
        cast_ray(img_np, origin_x, origin_y, angle, OBSTACLE_THRESHOLD)
        for angle in angles
    ]

def raycast_image_from_array(mask_array, number_of_rays=10, max_angle=180.0):
    if mask_array.ndim == 2:
        img_np = np.stack([mask_array] * 3, axis=-1)
    elif mask_array.shape[-1] == 1:
        img_np = np.concatenate([mask_array] * 3, axis=-1)
    else:
        img_np = mask_array.copy()

    h, w, _ = img_np.shape
    origin_x = w // 2
    origin_y = h - 10

    angles = calc_raycast_angles(number_of_rays, max_angle)
    return [
        cast_ray(img_np, origin_x, origin_y, angle, OBSTACLE_THRESHOLD)
        for angle in angles
    ]
