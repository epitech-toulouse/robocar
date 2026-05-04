# Architecture de la simulation

## Vue d'ensemble

La simulation est construite autour de trois blocs:

- Gazebo: monde physique, voiture, circuit, LiDAR.
- Bridge ROS/Gazebo: conversion des messages entre Gazebo et ROS 2.
- Controleur ROS 2: lecture du LiDAR, execution de l'algorithme, publication des commandes.

## Schema des donnees

```text
Gazebo world
  worlds/robocar_empty.sdf
        |
        | publie /scan, /odom, /clock
        v
ros_gz_bridge
        |
        | /scan en sensor_msgs/msg/LaserScan
        v
robocar_sim_controller
        |
        | convertit le scan
        v
SimLidarSensor -> LidarObstacleAvoidanceAlgorithm
        |
        | target_speed + target_steering
        v
SimVescController
        |
        | publie /cmd_vel
        v
ros_gz_bridge
        |
        | /cmd_vel vers Gazebo
        v
Gazebo DiffDrive plugin
```

## Demarrage

Le demarrage standard passe par:

```bash
ros2 launch robocar_sim robocar_sim.launch.py
```

Le launch file fait les actions suivantes:

1. Charge le monde `worlds/robocar_empty.sdf`.
2. Lance Gazebo avec `ros_gz_sim`.
3. Lance `ros_gz_bridge`.
4. Lance `robocar_sim_controller`.
5. Publie une transformation statique pour que RViz puisse afficher le LiDAR.
6. Lance RViz si `start_rviz:=true`.

## Controle autonome

En mode autonome, le noeud `robocar_sim_controller` fait tourner un timer toutes les 100 ms.

A chaque tick:

1. Il demande a `LidarObstacleAvoidanceAlgorithm` de calculer une sortie.
2. Si le calcul reussit, il envoie `target_speed` et `target_steering` a `SimVescController`.
3. `SimVescController` limite les valeurs et les convertit en `Twist`.
4. Le `Twist` est publie sur `/cmd_vel`.

## Controle manuel

En mode manuel, l'algorithme n'est pas utilise pour commander la voiture.

La commande peut venir de:

- la boucle menu texte integree au noeud C++;
- l'application Qt `tools/robocar_menu_qt.py`.

Les commandes manuelles utilisent:

- `/robocar/menu/autonomous_enabled` pour passer AUTO/MANUAL;
- `/robocar/menu/manual_cmd_vel` pour envoyer la vitesse;
- parfois `/cmd_vel` directement depuis le menu Qt.

## Conversion LiDAR

Gazebo publie un `LaserScan`. Ce message contient surtout:

- `ranges`: distances mesurees;
- `angle_min`: angle de depart;
- `angle_increment`: ecart entre deux rayons;
- `intensities`: intensites optionnelles.

Le controleur fait deux conversions:

1. Publication debug sur `/robocar/lidar_points_flat`:
   - angle en degres;
   - distance en metres;
   - intensite entre 0 et 255.
2. Conversion pour l'algorithme via `SimLidarSensor`:
   - tableau indexe par angle;
   - distances en centimetres;
   - valeurs invalides remplacees par `UNDEFINED_LIDAR_VALUE`.

## Points de configuration

Les valeurs principales sont dans `config/controller.yaml`.

Les parametres les plus utiles a modifier sont:

- `autonomous_enabled`: demarrer en autonome ou manuel.
- `sim_auto_speed_scale`: augmenter ou reduire la vitesse issue de l'algorithme.
- `sim_max_linear_speed_mps`: vitesse lineaire maximale.
- `sim_max_angular_speed_radps`: vitesse angulaire maximale.
- `sim_reverse_steering`: inverser le sens de direction si la voiture tourne du mauvais cote.
- `auto_print`: afficher automatiquement un resume LiDAR.

## Ce qui est deja actif

- Monde Gazebo avec circuit et obstacles.
- Modele de voiture simple avec deux roues motrices.
- Plugin `DiffDrive` pour appliquer `/cmd_vel`.
- LiDAR 360 degres a 10 Hz.
- Bridge `/scan`, `/odom`, `/clock`, `/cmd_vel`.
- Controleur ROS 2 C++.
- Interface menu texte.
- Interface menu Qt.
- RViz configure pour afficher le LiDAR.

## Ce qui est prepare mais pas encore actif

- `sensors/sim_gps_sensor.*`: emplacement prevu pour simuler ou adapter un GPS.
- `world/simulation_world.*`: emplacement prevu pour isoler une logique de monde simulation cote C++.

