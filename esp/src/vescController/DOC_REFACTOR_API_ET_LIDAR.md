# Documentation des changements effectues

Date: 10/04/2026

## 1. Refactor des API en interfaces

Les classes API ont ete alignees sur une convention interface avec prefixe `I`.

Fichiers modifies dans `main/api/`:

- `main/api/driving_algorithm_interface.hpp`
  - `IDrivingAlgorithm`
- `main/api/lidar_sensor_api.hpp`
  - `ILidarSensor`
  - `getData(...)` est pure virtuelle
- `main/api/gps_sensor_api.hpp`
  - `IGpsSensor`
- `main/api/user_controller_api.hpp`
  - `IUserController`
- `main/api/vesc_controller_api.hpp`
  - `IVescController`

Chaque interface expose un constructeur par defaut et un destructeur virtuel.

## 2. Alignement implementation VESC

Le composant physique VESC herite de l interface VESC.

Fichier:

- `api/vescController.hpp`

Changements:

- heritage sur `IVescController`
- include vers `../main/api/vesc_controller_api.hpp`

## 3. Copie de l algo obstacle avoidance dans le dossier algo

Le code de l evitement d obstacles existant dans `main/drive.*` a ete recopie dans:

- `main/algo/LidarObstacleAvoidance/drive.hpp`
- `main/algo/LidarObstacleAvoidance/drive.cpp`

Objectif: isoler ce module d algo dans son dossier dedie sans changer la logique.

## 4. Ajout d un wrapper interface pour l algo lidar

Un adaptateur a ete ajoute pour utiliser l algo via les interfaces API:

- `main/algo/LidarObstacleAvoidance/lidar_obstacle_avoidance_algorithm.hpp`
- `main/algo/LidarObstacleAvoidance/lidar_obstacle_avoidance_algorithm.cpp`

Ce wrapper:

- implemente `IDrivingAlgorithm`
- depend de `ILidarSensor`
- convertit `lidar_array_t` vers `std::vector<LidarPoint>`
- reutilise `AutonomousDriver::compute_commands(...)`
- remplit `DrivingAlgorithmOutput`

## 5. Build CMake

Le fichier source du wrapper a ete ajoute au composant main:

- `main/CMakeLists.txt`

Source ajoutee:

- `algo/LidarObstacleAvoidance/lidar_obstacle_avoidance_algorithm.cpp`

## 6. Etat de la simulation

Arborescence minimale en place dans `simulation/`, fichiers vides (squelette):

- `simulation/CMakeLists.txt`
- `simulation/main.cpp`
- `simulation/include/simulation_types.hpp`
- `simulation/sensors/sim_gps_sensor.hpp`
- `simulation/sensors/sim_gps_sensor.cpp`
- `simulation/sensors/sim_lidar_sensor.hpp`
- `simulation/sensors/sim_lidar_sensor.cpp`
- `simulation/output/sim_vesc_controller.hpp`
- `simulation/output/sim_vesc_controller.cpp`
- `simulation/world/simulation_world.hpp`
- `simulation/world/simulation_world.cpp`

## 7. Points non encore branches

- `main/main.cpp` utilise encore directement `AutonomousDriver` et `GpsAutonomousDriver`.
- Le nouveau wrapper `LidarObstacleAvoidanceAlgorithm` est compile mais pas encore instancie dans la boucle principale.
- La simulation est uniquement scaffoldee, sans implementation fonctionnelle.

## 8. Prochaine etape recommandee

Brancher `LidarObstacleAvoidanceAlgorithm` dans `main/main.cpp` via `IDrivingAlgorithm`, puis ajouter une implementation concrete `ILidarSensor` compatible avec ce flux.
