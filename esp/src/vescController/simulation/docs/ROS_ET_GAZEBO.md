# Comment marchent ROS 2 et Gazebo dans cette simulation

## ROS 2 en quelques mots

ROS 2 est l'architecture logicielle qui fait communiquer les differentes parties du robot.

Dans cette simulation, ROS 2 sert a:

- lancer les programmes dans le bon ordre;
- transporter les donnees capteurs;
- envoyer les commandes moteur/direction;
- parametrer le comportement du controleur;
- visualiser les donnees avec RViz.

## Les notions ROS 2 importantes

### Package

Un package ROS 2 est un module installable. Ici, le package s'appelle `robocar_sim`.

Les fichiers importants pour declarer le package sont:

- `package.xml`: declare le nom du package et ses dependances.
- `CMakeLists.txt`: explique comment compiler et installer le package.

### Node

Un node est un programme ROS qui tourne dans le systeme.

Ici, le node principal est:

- `robocar_sim_controller`

Il est cree dans `main.cpp`, puis sa logique est dans `robocar_sim_controller_node.cpp`.

### Topic

Un topic est un canal de communication. Un node publie des messages, un autre node les lit.

Les topics principaux sont:

- `/scan`: donnees LiDAR venant de Gazebo.
- `/cmd_vel`: commande de vitesse envoyee a la voiture dans Gazebo.
- `/odom`: odometrie venant de Gazebo.
- `/clock`: temps de simulation.
- `/robocar/lidar_points_flat`: version simplifiee du LiDAR publiee par le controleur.
- `/robocar/menu/autonomous_enabled`: active/desactive le mode autonome.
- `/robocar/menu/manual_cmd_vel`: commande manuelle envoyee par le menu Qt.

### Message

Un message est le format des donnees envoyees sur un topic.

Exemples utilises ici:

- `sensor_msgs/msg/LaserScan`: tableau de distances LiDAR.
- `geometry_msgs/msg/Twist`: vitesse lineaire et vitesse angulaire.
- `std_msgs/msg/Float32MultiArray`: tableau de nombres flottants.
- `std_msgs/msg/Bool`: booleen pour activer ou desactiver le mode autonome.

### Publisher et subscriber

Un publisher envoie des messages sur un topic. Un subscriber ecoute un topic.

Dans `RobocarSimControllerNode`:

- subscriber `/scan`: recoit le LiDAR;
- publisher `/cmd_vel`: commande la voiture;
- publisher `/robocar/lidar_points_flat`: expose les points LiDAR convertis;
- subscriber `/robocar/menu/autonomous_enabled`: recoit le mode AUTO/MANUAL;
- subscriber `/robocar/menu/manual_cmd_vel`: recoit les commandes manuelles.

### Parametres

Les parametres permettent de changer le comportement sans modifier le code.

Ils sont dans `config/controller.yaml`, par exemple:

- `scan_topic`: topic LiDAR a lire.
- `cmd_topic`: topic de commande a publier.
- `autonomous_enabled`: mode autonome active au demarrage.
- `sim_max_linear_speed_mps`: vitesse lineaire maximale.
- `sim_max_angular_speed_radps`: vitesse angulaire maximale.
- `sim_auto_speed_scale`: facteur d'echelle de la vitesse calculee par l'algorithme.

### Launch file

Un launch file demarre plusieurs elements ensemble. Ici, `launch/robocar_sim.launch.py` demarre:

- Gazebo avec le monde SDF;
- le bridge ROS/Gazebo;
- le noeud `robocar_sim_controller`;
- un transform statique pour RViz;
- RViz si demande avec `start_rviz:=true`.

## Gazebo en quelques mots

Gazebo est le simulateur physique et 3D.

Dans cette simulation, Gazebo sert a:

- charger le circuit;
- simuler la voiture;
- simuler les collisions;
- simuler le LiDAR;
- appliquer les commandes `/cmd_vel` au modele de voiture.

## Le monde Gazebo

Le monde est defini dans `worlds/robocar_empty.sdf`.

Ce fichier contient:

- la gravite;
- les plugins systeme de Gazebo;
- le sol;
- les bordures et obstacles du circuit;
- le modele `robocar`;
- les roues et joints;
- le plugin `DiffDrive`;
- le LiDAR `gpu_lidar`.

## SDF, liens, joints et capteurs

Dans Gazebo, un robot est decrit par un fichier SDF.

Les elements importants sont:

- `model`: un objet complet, par exemple `robocar`.
- `link`: une piece physique du robot, par exemple `base_link`, `left_wheel_link`, `lidar_link`.
- `joint`: une liaison entre deux pieces, par exemple les roues avec le chassis.
- `sensor`: un capteur simule, ici le LiDAR.
- `plugin`: un comportement ajoute par Gazebo, ici `DiffDrive` pour piloter les roues.

## Le bridge ROS/Gazebo

Gazebo et ROS 2 n'utilisent pas exactement le meme systeme de messages. Le bridge `ros_gz_bridge` convertit les messages entre les deux mondes.

Dans `robocar_sim.launch.py`, le bridge convertit:

- `/clock`: Gazebo vers ROS 2;
- `/scan`: Gazebo vers ROS 2;
- `/odom`: Gazebo vers ROS 2;
- `/cmd_vel`: ROS 2 vers Gazebo.

Le sens de la fleche est important:

- Gazebo produit `/scan`, ROS le lit.
- ROS produit `/cmd_vel`, Gazebo l'applique.

## Cycle de fonctionnement

1. Gazebo simule le monde et le LiDAR.
2. Le LiDAR Gazebo publie `/scan`.
3. `ros_gz_bridge` convertit `/scan` en message ROS 2 `LaserScan`.
4. `robocar_sim_controller` lit `/scan`.
5. Le controleur convertit le scan pour l'algorithme.
6. L'algorithme calcule une vitesse et une direction.
7. `SimVescController` transforme cette commande en `geometry_msgs/msg/Twist`.
8. Le controleur publie `/cmd_vel`.
9. `ros_gz_bridge` convertit `/cmd_vel` vers Gazebo.
10. Le plugin Gazebo `DiffDrive` deplace la robocar.

