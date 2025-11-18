import evdev.ecodes
import evdev
from pyvesc import VESC
import Jetson.GPIO as GPIO
import time
import argparse
from enum import Enum
from multiprocessing import Process, Value
from ctypes import c_double
import tensorflow as tf
import numpy as np
from PIL import Image
import subprocess
import re
import cv2
import gc

runMode = None
gamepad = None
light_toggled = False

GPIO.setmode(GPIO.BOARD)

channels = [36, 38, 40] #40 = bip 38 = avant 36 = arrière
lights_channels = [36, 38]

GPIO.setup(channels, GPIO.OUT)

# Configure TensorFlow memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def complete_mask_lines(mask: np.ndarray) -> np.ndarray:
    # Si le masque est de forme (H, W, 1), on le réduit à (H, W)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[:, :, 0]

    assert mask.ndim == 2, f"Expected 2D mask, got shape {mask.shape}"

    # Binarisation stricte
    binary = (mask > 127).astype(np.uint8) * 255

    # Étape 1 : Morphologie pour combler des trous
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # Plus large pour gros trous
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Étape 2 : Connexion manuelle des composants blancs proches
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    connected = closed.copy()

    # Comparer les extrémités de chaque contour et relier ceux qui sont proches
    for i in range(len(contours)):
        for j in range(i + 1, len(contours)):
            cnt1 = contours[i]
            cnt2 = contours[j]

            # On compare les points extrêmes des deux contours
            for pt1 in cnt1:
                for pt2 in cnt2:
                    p1 = tuple(pt1[0])
                    p2 = tuple(pt2[0])
                    dist = np.linalg.norm(np.array(p1) - np.array(p2))
                    if dist < 30:  # seuil ajustable selon l'espacement max à relier
                        cv2.line(connected, p1, p2, 255, thickness=2)

    return connected

class GameMode(Enum):
    CONTROLLER = 1
    IA = 2

serial_port = '/dev/ttyACM0'
interval = 0.5

# VESC process (runs independently)
def manage_vesc(speed_obj, steering_obj):
    with VESC(serial_port=serial_port) as motor:
        speed = 0.0
        while True:
            if speed < speed_obj.value:
                speed += 0.02
                if speed > speed_obj.value:
                    speed = speed_obj.value
            elif speed > speed_obj.value:
                speed -= 0.05
                if speed < speed_obj.value:
                    speed = speed_obj.value

            motor.set_duty_cycle(speed)
            motor.set_servo((steering_obj.value + 1) / 2)
            #print(f"[SPEED] {speed:.2f} [STEERING] {steering_obj.value:.2f}")

# Image utilities
def load_and_preprocess_image(path, target_size=(128, 128)):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)
    return image

def mask_predict(image_array, model):
    result = model.predict(image_array, verbose=0)
    print(f"Mask shape: {result[0].shape}")
    return result[0]

def save_mask_to_file(mask, path="predicted_mask.png"):
    if mask.shape[-1] == 1:
        mask = mask[..., 0]
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    img = Image.fromarray(binary_mask, mode='L')
    img.save(path)
    img.close()  # Fix: Explicit cleanup
    return path

def run_ia_loop(speed_obj, steer_obj):
    global runMode
    from camera_script import init_camera, get_image_array
    from raycasts_reader import raycast_image

    # Fix: Load models ONCE outside the loop
    mask_model = tf.keras.models.load_model('128_mask_gen.h5', compile=False)
    drive_model = tf.keras.models.load_model("trained_model.h5", compile=False)
    init_camera()
    target_frame_duration = 1.0 / 30.0  # 33.33 ms per frame

    loop_count = 0  # Fix: Add loop counter for garbage collection

    while runMode == GameMode.IA:
        frame_start = time.time()
        for event in gamepad.read_loop():
            if not check_change_gamemode(event):
                break

        createImage()
        image_array = load_and_preprocess_image("/home/robocartls/robocar/images/frame.jpg")
        predicted_mask = mask_predict(image_array, mask_model)
        binary_mask = (predicted_mask > 0.5).astype(np.uint8) * 255
        completed_mask = complete_mask_lines(binary_mask)

        # Fix: Proper PIL cleanup
        img = Image.fromarray(completed_mask)
        img.save("completed_mask.png")
        img.close()

        save_mask_to_file(predicted_mask)

        raycasts = raycast_image("completed_mask.png", number_of_rays=10, max_angle=180.0)

        batch_raycasts = np.array(raycasts).reshape(1, -1)
        predictions = drive_model.predict({'raycasts': batch_raycasts}, verbose=0)

        pred_speed = 0.25
        pred_steering = predictions[1][0][0]

        speed_obj.value = pred_speed / 10
        steer_obj.value = pred_steering

        print(f"[IA] Predicted speed: {pred_speed:.2f}, steering: {pred_steering:.2f}")

        # Fix: Periodic garbage collection
        loop_count += 1
        if loop_count % 10 == 0:
            gc.collect()

        # Maintain 30 FPS
        frame_duration = time.time() - frame_start
        sleep_time = target_frame_duration - frame_duration
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(f"[⚠️] Frame over time budget: {frame_duration:.3f}s")

def light_toggle():
    global light_toggle
    light_toggle = not light_toggle
    if (light_toggle):
        GPIO.output(channels, GPIO.HIGH)
    else:
        GPIO.output(channels, GPIO.LOW)

# New: image-only loop (no driving)
def run_image_loop():
    from camera_script import init_camera, createImage

    # Fix: Load model ONCE outside the loop
    mask_model = tf.keras.models.load_model('128_mask_gen.h5', compile=False)
    init_camera()

    loop_count = 0  # Fix: Add loop counter for garbage collection

    while True:
        createImage()
        image_array = load_and_preprocess_image("/home/robocartls/robocar/images/frame.jpg")
        predicted_mask = mask_predict(image_array, mask_model)
        binary_mask = (predicted_mask > 0.5).astype(np.uint8) * 255
        completed_mask = complete_mask_lines(binary_mask)

        # Fix: Proper PIL cleanup
        img = Image.fromarray(completed_mask)
        img.save("completed_mask.png")
        img.close()

        mask_file_path = save_mask_to_file(predicted_mask)
        print(f"[IMAGE] Saved mask to {mask_file_path}")

        # Fix: Periodic garbage collection
        loop_count += 1
        if loop_count % 10 == 0:
            gc.collect()

        time.sleep(0.5)

def check_change_gamemode(event):
    global runMode
    if event.code == evdev.ecodes.BTN_NORTH:
        light_toggle()
    if event.code == evdev.ecodes.BTN_SOUTH:
        if runMode == GameMode.IA:
            runMode = GameMode.CONTROLLER
        else:
            runMode = GameMode.IA
        return True
    return False


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robocar Controller')
    parser.add_argument('--mode', '-m', choices=['controller', 'ia', 'image'], default='controller')
    args = parser.parse_args()

    if args.mode == 'ia':
        gamemode = GameMode.IA
        print("Starting in IA mode...")
    elif args.mode == 'image':
        gamemode = None
        print("Starting in IMAGE mode...")
    else:
        gamemode = GameMode.CONTROLLER
        print("Starting in CONTROLLER mode...")

    devices = []
    while not devices:
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        if not devices:
            print("Waiting for devices...")
            time.sleep(1)
    for device in devices:
        if "F710" in device.name:
            gamepad = evdev.InputDevice(device.path)
            break
    if gamepad is None:
        print("F710 controller not found.")
        exit(1)

    # Shared values between processes
    speed_objective = Value(c_double, 0.0)
    steering_objective = Value(c_double, 0.0)

    # Start VESC motor process unless in image-only mode
    if args.mode != 'image':
        p = Process(target=manage_vesc, args=(speed_objective, steering_objective))
        p.start()

    # Image-only mode: no motor, just masks
    runMode = gamemode

    if args.mode == 'image':
        run_image_loop()

    running = True
    
    while running:
        # IA driving mode
        if runMode == GameMode.IA:
            run_ia_loop(speed_objective, steering_objective)
        # Manual controller mode
        elif runMode == GameMode.CONTROLLER:
            R2_value = 0.0
            L2_value = 0.0

            for event in gamepad.read_loop():
                check_change_gamemode(event)
                if (runMode == GameMode.IA):
                    break
                if event.type == evdev.ecodes.EV_ABS:
                    abs_event = evdev.categorize(event)

                    if event.code == 0:  # Left stick horizontal
                        steering_objective.value = abs_event.event.value / 32767
                    elif event.code == 5:  # R2
                        R2_value = abs_event.event.value / 255 / 3
                    elif event.code == 2:  # L2
                        L2_value = abs_event.event.value / 255 / 5

                    speed_objective.value = R2_value - L2_value
        else:
            print("Error gamemode unknown!")
            exit(1)

GPIO.cleanup()
