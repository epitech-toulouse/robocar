# Robocar UML Explication

https://drive.google.com/file/d/1jeNW8suWbgOqkJSxqye2-hnbD7rjB7U2/view?usp=drive\_link


## Notes:
Les fonctions internes aux drivers ne sont pas encore complete, manque de visibilité sur la base de code, de meme pour les interfaces
## Explication:

### Simulation

Chaque capteur hérite d'une interface spécifique, permettant de mettre le lidar, le gps, la caméra et le vesc dans une interface de simulation.  
L'implémentation de ces simulations permettra de tester les algorithmes de conduite, sans utiliser la vraie voiture.  
Différentes simulations pourront exister, comme par exemple :  
 - GPS ONLY (les commandes du vesc simulent un déplacement dans les coordonnées GPS, alors récupérées)
 - LIDAR ONLY (le GPS n'est pas set, et le lidar simule différentes configurations et enchaînements de murs/obstacles)
 - COMBINED (simulation complète)

### Interface Utilisateur

La "UserControllerInterface" représente le contrôle manuel et l'interface utilisateur.  
Elle peut :  
 - Désactiver la voiture (arrêt urgence)
 - Mettre à jour les paramètres des algorithmes
 - Activer/Désactiver chaque algorithme
 - Envoyer des données de contrôle manuel
 - Envoyer différents paramètres aux algorithmes (exemple les points GPS)

### Algorithmes

Il existe 5 algorithmes :  
 - LidarObstacleAvoidance (évitement d'obstacles lidar)
 - LidarFar (déplacement du lidar pour aller le plus loin possible)
 - Gps (déplacement vers un/des points définis)
 - FollowLine (suivi de ligne caméra + le reste des trucs caméra [on aura jamais cette cam])
 - ManualDriving (conduite manuelle, logique)

### Décision de conduite

La conduite est réalisée par un mix des algorithmes.

Chaque algo donne deux valeurs :  
 - La priorité, définie dans l'interface utilisateur
 - Le poids, défini en fonction des entrées des algorithmes

La priorité représente un peu l'importance de l'algorithme.  
Par exemple, la conduite manuelle a la priorité la plus importante, et disons qu'elle sera dix fois supérieure à la priorité GPS.

Le poids représente si l'algorithme pense que ses outputs sont importants ou pas.  
Par exemple l'évitement d'obstacle est très **très** important de près, mais peu important de loin.

### Coupe circuit

Le vesc est désactivé si :  
 - Le coupe circuit est débranché (coupé)
 - La manette (interface utilisateur) se déconnecte
 - La manette report un arrêt d'urgence

Pour ractiver le vesc, il faut que la manette soit connectée, et que le coupe circuit soit rebranché.  
Il est important que le coupe circuit soit rebranché en dernier. C'est lui qui ré-active.
