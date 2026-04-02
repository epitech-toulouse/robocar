# Robocar UML Explication

[

drive.google.com

https://drive.google.com/file/d/1jeNW8suWbgOqkJSxqye2-hnbD7rjB7U2/view?usp=drive\_link

](https://drive.google.com/file/d/1jeNW8suWbgOqkJSxqye2-hnbD7rjB7U2/view?usp=drive_link)

## Notes:
Les fonctions internes aux drivers ne sont pas encore complete, manque de visibilité sur la base de code, de meme pour les interfaces
## Explication:
Regardez bien le code couleur:
*   **Set** pointe vers quelquechose qui est modifié
*   **Get** pointe vers quelquechose dont l'information est récupéré
*   **Trigger** pointe vers quelquechose qui se fait activé / reveillé

Exemple: La _DeductorEngine_ pointe vers le _VescDriver_ car il le reveille après une mise à jour du _VescState_ pour qu'il puisse récupérer les nouvelles infos et les appliquées à la vitesse et rotation du véhicule.

Les Drivers sont les parsers qui intéragissent directement avec le matériel, les interfaces stock ces données parfaites et purifiées prêt à la l'application dans la _DeductionEngine_

Chaque interfaces stocks, des informations lié à l'entourage de la voiture, l'alignement avec les lignes (_lineAlignmentOffset_) (camera), les obstacles (_obstacleView_) (camera) avec les infos à propos d'eux (piétons, feu tricolore), et la vision du lidar en 360 degrès stocké en array de float 1d ou chaque cellule représente la distance du point, c'est la _lidarView._
(Pour le gps reste encore à determiner la données stockées, du moins je suis pas assez renseigné (svp dcp mettez à jour))

La _DetuctionEngine_ c'est le cerveau principal, là ou les Decisions engine liés à chaque composant sont des députés, la _DeductionEngine_ est le président qui prend la décision, chaques _DecisionEngine_ stockera une vitesse et rotation du véhicule voulu qui aura été calculé celon chaque spécificité, la _DeductionEngine_ elle déduira la vitesse idéal selon toutes les attentes.

La _DecisionEngine_ est une classe abstraite qui est utilisable pour chaques interfaces, c'est la classe algorithmique qui contiendra la logique d'évitement d'obstacle pour le lidar, de suivi de trajectoire pour le gps et de suivi de ligne pour la camera.

La _ControllerInterface_ est la classe qui sera toujours à l'écoute de notre manette, si la prise en main manuelle est enclenché, le _ControllerInterface_ viendra modifié le _VescState_ permettant de prendre un controlle du véhicule total ou partiel (avec assistance lidar et camera par exemple)
##