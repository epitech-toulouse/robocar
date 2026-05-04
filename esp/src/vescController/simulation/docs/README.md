# Documentation de la simulation Robocar

Ce dossier explique le fonctionnement de la simulation ROS 2 + Gazebo et le role de chaque fichier du package `robocar_sim`.

## Par ou commencer

1. Lire [ROS_ET_GAZEBO.md](ROS_ET_GAZEBO.md) pour comprendre les bases: noeuds, topics, messages, launch files, monde Gazebo et bridge ROS/Gazebo.
2. Lire [ARCHITECTURE_SIMULATION.md](ARCHITECTURE_SIMULATION.md) pour comprendre le chemin complet des donnees dans cette simulation.
3. Lire [COMMENTAIRES_FICHIERS.md](COMMENTAIRES_FICHIERS.md) pour avoir un commentaire fichier par fichier.

## Resume rapide

La simulation demarre un monde Gazebo contenant une robocar, un circuit et un LiDAR. Gazebo publie les donnees du LiDAR sur son systeme de transport interne. Le bridge `ros_gz_bridge` convertit ces donnees en topics ROS 2, notamment `/scan`.

Le noeud ROS 2 `robocar_sim_controller` lit `/scan`, transforme le scan en donnees utilisables par l'algorithme d'evitement d'obstacles, calcule une commande de conduite, puis publie `/cmd_vel`. Le bridge renvoie `/cmd_vel` vers Gazebo, et le plugin `DiffDrive` applique cette commande au modele de voiture.

