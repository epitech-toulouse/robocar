# Robocar Project

Welcome to the **Robocar** project! This repository contains the source code for an autonomous vehicle capable of line following and obstacle detection.

## Project Structure

The project is organized into three main components:

### 1.`follow-the-line`
This directory contains the core intelligence of the autonomous car. It uses Computer Vision and Deep Learning to navigate.

*   **AI & Computer Vision**: Uses a TensorFlow Lite model (`128_mask_gen.tflite`) to segment the lane lines from the camera feed.
*   **Path Planning**: Processes the segmented mask using raycasting to determine the optimal steering angle and speed.
*   **VESC Control**: Interfaces with the VESC motor controller to drive the car.
*   **Dual Modes**: Supports both autonomous driving (`IA` mode) and manual control via gamepad (`CONTROLLER` mode).

**Usage:**
```bash
# Run in autonomous mode
python3 main.py --mode ia

# Run in manual controller mode
python3 main.py --mode controller
```

### 2. `lidar`
A high-performance C++ utility for interfacing with the **LD19 LiDAR** sensor.

*   **Fast Parsing**: Efficiently parses raw serial data from the LiDAR.
*   **Visualization**: Includes a real-time visualization tool built with **SFML** to see what the robot sees.
*   **C++ Implementation**: Optimized for low-latency obstacle detection.

**Building & Running:**
```bash
cd lidar
mkdir build && cd build
cmake ..
make
./ld19_lidar
```

### 3.`RobocarController`
A dedicated Python module for manual teleoperation.

*   **🕹️ Gamepad Support**: Designed for the **Logitech F710** controller.
*   **🔌 Plug & Play**: Automatically detects the controller and maps inputs to steering and throttle commands.
*   **🐍 Simple Interface**: Easy-to-understand Python script using `evdev`.

---