# ROS 2 + Gazebo Setup (Scaffold)

This folder contains a first ROS 2 simulation scaffold for Ubuntu.

## Target versions

- ROS 2 Jazzy
- Gazebo Harmonic
- Ubuntu 24.04

## Important

Do these steps inside Ubuntu 24.04 (native or VM).

## 1) Install ROS 2 Jazzy (Ubuntu 24.04)

### 1.1 System prep

```bash
sudo apt update
sudo apt install -y software-properties-common curl gnupg2 lsb-release
sudo add-apt-repository universe -y
```

### 1.2 Add ROS 2 apt repository

```bash
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### 1.3 Install ROS 2 Jazzy desktop + tools

```bash
sudo apt update
sudo apt install -y ros-jazzy-desktop ros-dev-tools
```

### 1.4 Source ROS automatically

```bash
# Bash users
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Zsh users
echo "source /opt/ros/jazzy/setup.zsh" >> ~/.zshrc
source ~/.zshrc
```

Important:

- Do not run `source ~/.bashrc` from a `zsh` shell.
- In `zsh`, always source `setup.zsh` (not `setup.bash`).

### 1.5 Quick check

```bash
ros2 -h
```

## 2) Install Gazebo Harmonic

### 2.1 Add Gazebo apt repository

```bash
sudo curl -fsSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
```

### 2.2 Install Gazebo Harmonic

```bash
sudo apt update
sudo apt install -y gz-harmonic
```

### 2.3 Install ROS <-> Gazebo bridge packages

```bash
sudo apt install -y ros-jazzy-ros-gz ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge
```

### 2.4 Quick checks

```bash
gz sim --version
ros2 pkg list | grep ros_gz
```

## 3) Build this simulation package

From a ROS 2 workspace:

```bash
mkdir -p ~/robocar_ws/src
cd ~/robocar_ws/src

# IMPORTANT: replace this with your real absolute path
# Example for this project layout:
ln -s ~/Desktop/robocar/esp/src/vescController/simulation robocar_sim

cd ~/robocar_ws
source /opt/ros/jazzy/setup.bash
colcon list
colcon build --packages-select robocar_sim --symlink-install
source install/setup.bash
```

## 4) Run

```bash
source /opt/ros/jazzy/setup.bash
source ~/robocar_ws/install/setup.bash
ros2 launch robocar_sim robocar_sim.launch.py
```

## 4.1) Run with interactive menu (recommended)

Terminal 1 (Gazebo + bridge only):

```bash
source /opt/ros/jazzy/setup.bash
source ~/robocar_ws/install/setup.bash
ros2 launch robocar_sim robocar_sim.launch.py start_controller:=false
```

Terminal 2 (controller with menu):

```bash
source /opt/ros/jazzy/setup.bash
source ~/robocar_ws/install/setup.bash
ros2 run robocar_sim robocar_sim_controller
```

Or with helper script (menu + RViz LiDAR window):

```bash
./alexis.sh --menu --rviz
```

## 4.2) If you get "package robocar_sim not found"

This usually means the symlink target path is wrong or broken.

Check the package link:

```bash
ls -la ~/robocar_ws/src
readlink -f ~/robocar_ws/src/robocar_sim
test -f ~/robocar_ws/src/robocar_sim/package.xml && echo OK || echo BROKEN
```

If broken, recreate it with the correct path:

```bash
cd ~/robocar_ws/src
rm -f robocar_sim
ln -s ~/Desktop/robocar/esp/src/vescController/simulation robocar_sim
```

Then rebuild and source again:

```bash
cd ~/robocar_ws
source /opt/ros/jazzy/setup.bash
colcon list
colcon build --packages-select robocar_sim --symlink-install
source install/setup.bash
ros2 pkg list | grep robocar_sim
```

## Added files

- `package.xml`
- `CMakeLists.txt`
- `main.cpp` (ROS controller with LiDAR conversion + menu)
- `launch/robocar_sim.launch.py`
- `config/controller.yaml`
- `worlds/robocar_empty.sdf`

## Current scope

- Starts Gazebo world with floor, obstacles, rear wheel propulsion, and front steering joints limited to +/-35 deg
- Car has a top-mounted LiDAR sensor publishing `/scan` at 20 Hz with 720 horizontal samples
- Starts ROS-Gazebo bridge for `/clock`, `/scan`, `/cmd_vel`
- Starts `robocar_sim_controller` ROS node
- Converts LaserScan into point list similar to project LiDAR parsing style:
	- angle normalized to `[0, 360)`
	- range filtered to `(0.05m, 12.0m)`
	- intensity filtering when available
	- sorted by angle
- Publishes converted points as flat triples on `/robocar/lidar_points_flat`

Note:

- A simple terminal menu is available in the controller node.
- `cmd_vel` publishing is available from the menu (`cmd v w`); front steering is published separately to the two front steering joints.

## Menu commands

When the node starts, use:

- `h` : show menu
- `s` : print status (point count, LiDAR Hz, current cmd_vel)
- `a` : toggle auto LiDAR summary print
- `cmd <linear> <angular>` : set command velocity values
- `stop` : set command velocity to `0 0`
- `q` : stop menu input thread

## Quick topic checks

```bash
ros2 topic hz /scan
ros2 topic echo /robocar/lidar_points_flat --once
```

Expected:

- `/scan` should be around `10 Hz`.
- `/robocar/lidar_points_flat` packs points by triples:
	- index `0`: angle in degrees
	- index `1`: distance in meters
	- index `2`: intensity (0..255)

The conversion logic follows your LiDAR parsing behavior:

- angle normalized to `[0, 360)`
- min/max range filter `(0.05m, 12.0m)`
- intensity filtering when intensity data is available
- sorted by angle

## Next steps

- Add robot model (SDF/URDF)
- Plug LiDAR/GPS topics from model sensors
- Connect controller to real algorithm wrapper (`IDrivingAlgorithm` path)
- Add scenario worlds
