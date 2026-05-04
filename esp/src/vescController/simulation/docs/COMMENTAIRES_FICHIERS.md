# Commentaires fichier par fichier

Cette page decrit le role de chaque fichier dans `esp/src/vescController/simulation`.

## Racine du package

### `README_ROS_GAZEBO.md`

Guide d'installation et d'execution. Il explique comment installer ROS 2 Jazzy, Gazebo Harmonic, construire le package avec `colcon`, lancer la simulation et verifier les topics.

### `package.xml`

Fichier manifeste du package ROS 2 `robocar_sim`.

Il declare:

- le nom du package;
- sa version;
- ses dependances de compilation et d'execution;
- le type de build `ament_cmake`.

ROS 2 et `colcon` lisent ce fichier pour savoir de quoi le package a besoin.

### `CMakeLists.txt`

Fichier de compilation CMake.

Il:

- active C++17;
- trouve les dependances ROS 2 (`rclcpp`, `sensor_msgs`, `geometry_msgs`, `std_msgs`);
- recupere les sources de l'algorithme LiDAR dans `../main/algo/LidarObstacleAvoidance`;
- cree l'executable `robocar_sim_controller`;
- installe le script Qt;
- installe les dossiers `launch`, `worlds` et `config` dans le share ROS du package.

### `main.cpp`

Point d'entree de l'executable C++.

Il:

1. initialise ROS 2 avec `rclcpp::init`;
2. cree le noeud `RobocarSimControllerNode`;
3. lance la boucle ROS avec `rclcpp::spin`;
4. arrete ROS proprement avec `rclcpp::shutdown`.

Il ne contient pas la logique metier; il sert seulement a demarrer le noeud.

### `robocar_sim_controller_node.hpp`

Declaration de la classe principale `RobocarSimControllerNode`.

Il liste:

- les publishers;
- les subscribers;
- les timers;
- les parametres;
- l'etat courant du LiDAR;
- l'etat des commandes;
- le capteur simule `SimLidarSensor`;
- le controleur simule `SimVescController`;
- l'algorithme `IDrivingAlgorithm`.

Ce fichier montre la structure interne du noeud.

### `robocar_sim_controller_node.cpp`

Implementation du noeud principal.

Il fait le travail central:

- declare les parametres ROS;
- cree les publishers et subscribers;
- lit `/scan`;
- filtre les mesures invalides;
- convertit les angles en degres;
- publie `/robocar/lidar_points_flat`;
- alimente `SimLidarSensor`;
- execute l'algorithme d'evitement d'obstacles;
- convertit la sortie en `/cmd_vel`;
- gere le mode autonome ou manuel;
- affiche un menu texte optionnel.

Le timer `commandTimer` tourne toutes les 100 ms et publie la commande de conduite. Le timer `statusTimer` affiche l'etat du LiDAR et de la commande chaque seconde.

### `alexis.sh`

Script helper pour construire et lancer la simulation plus facilement.

Il:

- cree le workspace ROS si besoin;
- cree le lien symbolique du package dans `~/robocar_ws/src`;
- source ROS 2 Jazzy;
- lance `colcon build`;
- source l'installation du workspace;
- lance la simulation;
- peut lancer le menu Qt;
- peut lancer RViz;
- peut faire un build propre avec `--clean`.

C'est le script pratique pour eviter de taper toutes les commandes ROS a la main.

## `launch/`

### `launch/robocar_sim.launch.py`

Launch file principal.

Il declare plusieurs options:

- `world`: chemin du monde Gazebo;
- `start_controller`: lancer ou non le controleur;
- `controller_menu_enabled`: activer ou non le menu texte;
- `start_rviz`: lancer ou non RViz;
- `rviz_config`: chemin de la configuration RViz.

Il demarre:

- Gazebo via `ros_gz_sim`;
- `ros_gz_bridge`;
- `robocar_sim_controller`;
- un `static_transform_publisher`;
- RViz si demande.

Le bridge convertit les topics `/clock`, `/scan`, `/odom` et `/cmd_vel`.

## `config/`

### `config/controller.yaml`

Configuration du noeud `robocar_sim_controller`.

Il fixe:

- les noms des topics;
- le mode autonome initial;
- les limites de vitesse;
- le facteur d'echelle de l'algorithme;
- l'inversion eventuelle du steering;
- l'activation du menu;
- l'affichage automatique du resume LiDAR.

### `config/robocar_lidar.rviz`

Configuration RViz pour visualiser le LiDAR.

Elle affiche:

- une grille;
- le topic `/scan` sous forme de points;
- une camera orbit autour du LiDAR.

Le fixed frame est `map`, et le launch file ajoute une transformation statique entre `map` et `robocar/lidar_link/top_lidar`.

## `worlds/`

### `worlds/robocar_empty.sdf`

Monde Gazebo principal.

Il contient:

- la gravite;
- les plugins systeme Gazebo;
- le sol;
- un circuit type Formule 1 avec bordures et obstacles;
- le modele `robocar`;
- les roues, joints et castors;
- le plugin `gz::sim::systems::DiffDrive`;
- un LiDAR `gpu_lidar` qui publie `/scan`.

Le plugin `DiffDrive` ecoute `/cmd_vel`, publie `/odom` et permet a Gazebo de deplacer la voiture.

## `sensors/`

### `sensors/sim_lidar_sensor.hpp`

Declaration de `SimLidarSensor`.

Cette classe implemente l'interface `ILidarSensor` attendue par l'algorithme existant. Elle stocke le dernier scan LiDAR converti dans le format du projet.

### `sensors/sim_lidar_sensor.cpp`

Implementation de `SimLidarSensor`.

La methode importante est `updateFromScan`.

Elle:

- recoit un `sensor_msgs/msg/LaserScan`;
- ignore les distances non finies ou hors limites;
- convertit les angles radians en degres;
- arrondit l'angle pour indexer le tableau LiDAR;
- convertit les distances metres en centimetres;
- garde la distance la plus proche pour chaque angle;
- marque les donnees comme disponibles.

`getData` permet ensuite a l'algorithme de recuperer le tableau LiDAR.

### `sensors/sim_gps_sensor.hpp`

Fichier vide actuellement.

Il sert probablement d'emplacement futur pour une classe GPS simulee, par exemple pour adapter une position Gazebo ou `/odom` vers une interface GPS du projet.

### `sensors/sim_gps_sensor.cpp`

Fichier vide actuellement.

Il contiendra l'implementation du futur capteur GPS simule si cette partie est ajoutee.

## `output/`

### `output/sim_vesc_controller.hpp`

Declaration de `SimVescController`.

Cette classe implemente l'interface `IVescController` attendue par le code existant, mais au lieu de parler a un vrai VESC, elle produit une commande ROS `Twist`.

### `output/sim_vesc_controller.cpp`

Implementation de `SimVescController`.

Elle:

- active/desactive le controleur;
- stocke une vitesse normalisee entre `-1` et `1`;
- stocke une direction normalisee entre `0` et `1`;
- limite les valeurs avec `std::clamp`;
- convertit la commande normalisee en `geometry_msgs/msg/Twist`.

`toTwistCommand` applique les limites configurees dans `SimControlConfig`:

- `maxLinearSpeedMps`;
- `maxAngularSpeedRadps`;
- `reverseSteering`.

## `include/`

### `include/simulation_types.hpp`

Petit fichier de types partages.

Il definit `SimControlConfig`, la configuration utilisee par `SimVescController` pour convertir une commande normalisee en vitesse ROS.

## `world/`

### `world/simulation_world.hpp`

Fichier vide actuellement.

Il semble reserve pour une future abstraction C++ du monde de simulation.

### `world/simulation_world.cpp`

Fichier vide actuellement.

Il contiendra l'implementation de cette future abstraction si elle devient necessaire.

## `tools/`

### `tools/robocar_menu_qt.py`

Interface graphique PyQt5 pour piloter la simulation.

Elle cree un noeud ROS Python `robocar_menu_qt` qui publie:

- `/robocar/menu/autonomous_enabled`;
- `/robocar/menu/manual_cmd_vel`;
- `/cmd_vel`.

La fenetre propose:

- bouton AUTO;
- bouton MANUAL;
- bouton STOP;
- champs vitesse lineaire et vitesse angulaire;
- envoi ponctuel;
- streaming a 10 Hz.

Elle sert a controler la voiture sans utiliser le menu texte du noeud C++.

