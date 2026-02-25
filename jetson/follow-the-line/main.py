import evdev.ecodes
import evdev
from itertools import combinations
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
from camera_script import init_camera

runMode = None
gamepad = None
light_toggled = False
backlight_toggled = False
bip_toggled = False

GPIO.setmode(GPIO.BOARD)

init_camera()

channels = [36, 38, 40] 
lights_channels = [36, 38]

GPIO.setup(channels, GPIO.OUT)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def complete_mask_lines_ultra_fast(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[:, :, 0]
    
    binary = (mask > 127).astype(np.uint8) * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return closed
    
class GameMode(Enum):
    CONTROLLER = 1
    IA = 2

serial_port = '/dev/ttyACM0'
interval = 0.5

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
            motor.set_servo((-steering_obj.value + 1) / 2)

def load_and_preprocess_image(path, target_size=(128, 128)):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)
    return image

def mask_predict_tflite(image_array, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(image_array.shape) == 3:
        image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

def drive_predict_tflite(raycast_batch, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], raycast_batch)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def drive_predict_tflite(raycast_batch, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    raycast_batch = raycast_batch.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], raycast_batch)
    interpreter.invoke()
    outputs = []
    for output_detail in output_details:
        output_data = interpreter.get_tensor(output_detail['index'])
        outputs.append(output_data)
    return outputs

def save_mask_to_file(mask, path="predicted_mask.png"):
    if mask.shape[-1] == 1:
        mask = mask[..., 0]
    binary_mask = (mask > 0.1).astype(np.uint8) * 255
    img = Image.fromarray(binary_mask, mode='L')
    img.save(path)
    img.close()  
    return path

def run_ia_loop(speed_obj, steer_obj):
    global runMode
    global bip_toggled
    global light_toggled
    global backlight_toggled
    from camera_script import get_image_array
    from raycasts_reader import raycast_image_from_array

    mask_interpreter = tf.lite.Interpreter(model_path="128_mask_gen.tflite")
    mask_interpreter.allocate_tensors()
    drive_interpreter = tf.lite.Interpreter(model_path="trained_model.tflite")
    drive_interpreter.allocate_tensors()

    while runMode == GameMode.IA:
        start_total = time.time()

        bip_toggled = False
        light_toggled = False
        backlight_toggled = False
        try:
            for event in gamepad.read():
                check_change_gamemode(event)
        except BlockingIOError:
            pass
        bip_update()
        light_update()
        backlight_update()

        start_frame = time.time()
        frame = get_image_array()
        if frame is None:
            print("[DEBUG] No frame captured")
            continue
        print(f"[DEBUG] Frame capture time: {(time.time() - start_frame) * 1000:.1f} ms")

        start_preprocess = time.time()
        image_array = np.expand_dims(np.array(frame).astype(np.float32) / 255.0, axis=0)
        print(f"[DEBUG] Preprocessing time: {(time.time() - start_preprocess) * 1000:.1f} ms")

        start_mask = time.time()
        predicted_mask = mask_predict_tflite(image_array, mask_interpreter)
        print(f"[DEBUG] Mask prediction time: {(time.time() - start_mask) * 1000:.1f} ms")

        start_binary = time.time()
        binary_mask = (predicted_mask > 0.3).astype(np.uint8) * 255
        completed_mask = complete_mask_lines_ultra_fast(binary_mask)
        print(f"[DEBUG] Mask post-processing time: {(time.time() - start_binary) * 1000:.1f} ms")

        start_raycast = time.time()
        raycasts = raycast_image_from_array(completed_mask, number_of_rays=10, max_angle=180.0)
        print(f"[DEBUG] Raycast computation time: {(time.time() - start_raycast) * 1000:.1f} ms")

        start_drive = time.time()
        batch_raycasts = np.array(raycasts).reshape(1, -1)
        predictions = drive_predict_tflite(batch_raycasts, drive_interpreter)
        print(f"[DEBUG] Drive model prediction time: {(time.time() - start_drive) * 1000:.1f} ms")

        pred_speed = 0.3
        pred_steering = predictions[1][0][0] * 1.3

        speed_obj.value = pred_speed / 10
        steer_obj.value = pred_steering

        total_time = (time.time() - start_total) * 1000
        print(f"[IA] Predicted speed: {pred_speed:.2f}, steering: {pred_steering:.2f}")
        print(f"[DEBUG] TOTAL loop time: {total_time:.1f} ms\n")

def light_update():
    if (light_toggled):
        GPIO.output(38, GPIO.HIGH)
    else:
        GPIO.output(38, GPIO.LOW)

def backlight_update():
    global backlight_toggled
    if (backlight_toggled):
        GPIO.output(36, GPIO.HIGH)
    else:
        GPIO.output(36, GPIO.LOW)

def bip_update():
    global bip_toggled
    if (bip_toggled):
        GPIO.output(40, GPIO.HIGH)
    else:
        GPIO.output(40, GPIO.LOW)

def run_image_loop():
    from camera_script import get_image_array

    interpreter = tf.lite.Interpreter(model_path="128_mask_gen.tflite")
    interpreter.allocate_tensors()
    init_camera()

    while True:
        frame = get_image_array()
        if frame is None:
            continue
        image_array = np.expand_dims(np.array(frame).astype(np.float32) / 255.0, axis=0)
        cv2.imwrite("frame.jpg", image_array)
        predicted_mask = mask_predict_tflite(image_array, interpreter)
        binary_mask = (predicted_mask > 0.4).astype(np.uint8) * 255
        completed_mask = complete_mask_lines(binary_mask)

        img = Image.fromarray(completed_mask)
        img.save("completed_mask.png")
        img.close()

        mask_file_path = save_mask_to_file(predicted_mask)
        print(f"[IMAGE] Saved mask to {mask_file_path}")

        time.sleep(0.5)

def check_change_gamemode(event):
    global runMode
    global light_toggled
    global backlight_toggled
    global bip_toggled
    if event.code == evdev.ecodes.BTN_NORTH:
        light_toggled = True
        return True
    if event.code == evdev.ecodes.BTN_WEST:
        bip_toggled = True
        return True
    if event.code == evdev.ecodes.BTN_EAST:
        backlight_toggled = True
        return True
    if event.code == evdev.ecodes.BTN_SOUTH and event.value == 1:
        if runMode == GameMode.IA:
            runMode = GameMode.CONTROLLER
        else:
            runMode = GameMode.IA
        return True
    return False

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

    speed_objective = Value(c_double, 0.0)
    steering_objective = Value(c_double, 0.0)

    if args.mode != 'image':
        p = Process(target=manage_vesc, args=(speed_objective, steering_objective))
        p.start()

    if args.mode == 'image':
        run_image_loop()

    runMode = gamemode
    running = True

    GPIO.output(40, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(40, GPIO.LOW)

    while running:
        if runMode == GameMode.IA:
            run_ia_loop(speed_objective, steering_objective)
        elif runMode == GameMode.CONTROLLER:
            R2_value = 0.0
            L2_value = 0.0

            bip_toggled = False
            light_toggled = False
            backlight_toggled = False
            for event in gamepad.read_loop():
                check_change_gamemode(event)
                if (runMode == GameMode.IA):
                    break
                if event.type == evdev.ecodes.EV_ABS:
                    abs_event = evdev.categorize(event)

                    if event.code == 0: 
                        steering_objective.value = (-abs_event.event.value) / 32767
                    elif event.code == 5: 
                        R2_value = abs_event.event.value / 255 / 3
                    elif event.code == 2:
                        L2_value = abs_event.event.value / 255 / 5

                    speed_objective.value = R2_value - L2_value
            bip_update()
            light_update()
            backlight_update()
        else:
            print("Error gamemode unknown!")
            exit(1)

GPIO.cleanup()
