# Documentation du module Autodrive

Ce dossier contient les scripts de pilotage pour le Robocar, incluant la navigation autonome basée sur un Lidar et le contrôle manuel via une manette de jeu.

## Description des fichiers

### 1. `main.py`
C'est le script principal exécutable pour lancer le véhicule.

*   **Rôle** : Gère la boucle principale de contrôle, les entrées manette, et la machine à états simple (Auto/Manuel).
*   **Fonctionnalités** :
    *   **Mode Manuel** : Pilotage direct avec la manette (accélération, direction).
    *   **Mode Auto** : Avance autonome avec évitement d'obstacles basique.
    *   **Gestion des lumières** : Supporte l'activation des lumières via GPIO (si sur Jetson).
*   **Utilisation** : `python3 main.py`

### 2. `LidarParser.py`
Interface de bas niveau pour le Lidar LD19.

*   **Rôle** : Établit la communication série avec le capteur Lidar, décode les paquets de données binaires et extrait les mesures de distance.
*   **Sortie** : Fournit une liste de points sous forme de dictionnaire `{'angle': float, 'distance': float}` via la méthode `get_points()`.
*   **Détails techniques** : Filtre les points invalides et maintient un buffer glissant des mesures récentes.

### 3. `autodriveState.py`
Module d'état pour l'intégration dans une architecture plus large (Machine à États).

*   **Rôle** : Définit la classe `AutoDriveState` qui hérite de `State`.
*   **Logique** : Implémente une logique de navigation autonome avec évitement d'obstacles.
*   **Détails de la classe `AutoDriveState`** :
    *   **Constantes** :
        *   `SAFE_DISTANCE` (1.3m) : Seuil de déclenchement de l'évitement.
        *   `SLOW_DISTANCE` (0.4m) : Seuil critique (inutilisé si < SAFE_DISTANCE).
        *   `STEER_ANGLE` (0.8) : Angle de braquage max.
        *   `STEER_SMOOTHING` (0.8) : Facteur d'atténuation du braquage.
        *   `FORWARD_SPEED` (0.06) : Vitesse de croisière.
        *   `SLOW_SPEED` (0.04) : Vitesse minimale.
        *   `SCAN_FRONT_DEG` (20°) : Demi-angle du cône de détection avant.

    *   **Fonctions** :
        *   `__init__(self)` : 
            *   Initialise l'instance et instancie `LidarParser` pour démarrer la lecture du capteur.
        *   `stop(self)` : 
            *   Arrête proprement le driver Lidar en appelant `self.lidar.stop()`.
        *   `run_single(self, motor: Motor, gamepad: Gamepad)` : 
            *   Méthode de contrôle principale appelée en boucle.
            *   Récupère les points Lidar via `lidar.get_points()`.
            *   Filtre les obstacles situés devant le véhicule (entre -20° et +20°).
            *   Détermine l'obstacle le plus proche (`min_dist`).
            *   **Si `min_dist < SAFE_DISTANCE`** : Active l'évitement.
                *   Calcule un angle de braquage opposé à l'angle de l'obstacle pour l'esquiver.
                *   Définit la vitesse via `get_speed_from_angle` (ralentit si l'obstacle est proche de l'axe central).
            *   **Sinon** : Avance tout droit à `FORWARD_SPEED` avec un angle de 0.
        *   `get_obstacles_in_range(self, points, min_angle, max_angle)` : 
            *   Parcourt tous les points Lidar.
            *   Normalise leur angle (-180° à +180°).
            *   Retourne la liste des points situés dans l'intervalle `[min_angle, max_angle]`.
        *   `get_speed_from_angle(self, angle)` : 
            *   Calcule la vitesse cible en fonction de la position de l'obstacle.
            *   Interpole linéairement entre `SLOW_SPEED` (si obstacle à 0°) et `FORWARD_SPEED` (si obstacle à 20°).

### 4. `lidarPrinter.py`
Outil de diagnostic simple.

*   **Rôle** : Affiche en continu dans la console les distances mesurées par le Lidar pour détecter les frontières ou calibrer le capteur.
*   **Utilisation** : Utile pour vérifier que le Lidar fonctionne correctement sans lancer tout le système de pilotage.

## Configuration & Constantes

Les paramètres principaux sont définis en tête des fichiers `main.py` et `autodriveState.py` :

*   `SAFE_DISTANCE` (1.3m) : Distance à laquelle l'évitement d'obstacle s'active.
*   `SLOW_DISTANCE` (0.4m) : Distance critique ou ralentissement.
*   `FORWARD_SPEED` : Vitesse de croisière en mode automatique.
*   `SCAN_FRONT_DEG` (20°) : Angle du cône de détection avant.

## Contrôles (Mode Manuel - Logitech F710)

*   **Stick Gauche (Horizontal)** : Direction (Gauche/Droite).
*   **Gâchette R2** : Accélérateur (Marche Avant).
*   **Gâchette L2** : Frein / Marche Arrière.
*   **Bouton A (Sud)** : Basculer entre mode **AUTO** et **MANUEL**.
*   **Bouton Y (Nord)** : Allumer/Éteindre les phares.

## Dépendances requises

*   Python 3
*   `pyserial` (pour le Lidar)
*   `evdev` (pour la manette)
*   (Optionnel) `Jetson.GPIO` (pour le contrôle des lumières sur NVIDIA Jetson)
