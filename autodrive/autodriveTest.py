import time
from control.State import State
from control.Motor import Motor
from control.Gamepad import Gamepad
from .LidarParser import LidarParser


# DIRECTION DRIVE PARAMETERS
SCAN_FRONT_DEG = 30   # Élargi pour mieux détecter les ouvertures
SAFE_DISTANCE = 4.0   # Distance de sécurité pour ralentir
SLOW_DISTANCE = 1.5   # Distance de ralentissement
STOP_DISTANCE = 0.6   # Distance d'arrêt

FORWARD_SPEED = 0.10   # Vitesse maximale augmentée
BACKWARD_SPEED = -0.04 # Vitesse de recul
SLOW_SPEED = 0.03      # Vitesse minimale

# STEERING AVOIDANCE PARAMETERS
STEERING_SCAN_ANGLE = 45  # Angle de scan pour trouver les ouvertures
STEER_ANGLE = 1         # Angle de braquage max
STEER_SMOOTHING = 0.7     # Lissage du braquage


class AutoDriveState(State):
    def __init__(self):
        print("Initializing AutoDrive...")
        self.lidar = LidarParser()
        self.last_steering = 0.0  # Pour le lissage
        self.reverse_timer = 0    # Compteur pour la marche arrière
    
    def stop(self):
        self.lidar.stop()

    def run_single(self, motor: Motor, gamepad: Gamepad):
        """Exécute un cycle de contrôle"""
        points = self.lidar.get_points()
        if not points:
            time.sleep(0.05)
            return
        
        # Analyse de l'environnement
        front_scan = self.scan_sector(points, -SCAN_FRONT_DEG, SCAN_FRONT_DEG)
        left_scan = self.scan_sector(points, SCAN_FRONT_DEG, STEERING_SCAN_ANGLE)
        right_scan = self.scan_sector(points, -STEERING_SCAN_ANGLE, -SCAN_FRONT_DEG)
        
        # Décision de vitesse et direction
        if self.reverse_timer > 0:
            self.handle_reverse(motor, front_scan)
            self.reverse_timer -= 1
        else:
            self.navigate(motor, front_scan, left_scan, right_scan)

    def scan_sector(self, points, min_angle, max_angle):
        """Analyse un secteur angulaire et retourne les statistiques"""
        obstacles = []
        for p in points:
            angle = self.normalize_angle(p['angle'])
            if min_angle <= angle <= max_angle:
                obstacles.append(p)
        
        if not obstacles:
            return {'min_dist': float('inf'), 'avg_dist': float('inf'), 'count': 0}
        
        distances = [o['distance'] for o in obstacles]
        return {
            'min_dist': min(distances),
            'avg_dist': sum(distances) / len(distances),
            'count': len(distances),
            'obstacles': obstacles
        }

    def normalize_angle(self, angle):
        """Normalise un angle entre -180 et 180"""
        angle = angle % 360
        if angle > 180:
            angle -= 360
        return angle

    def navigate(self, motor: Motor, front, left, right):
        """Logique principale de navigation"""
        
        # Obstacle très proche : marche arrière
        if front['min_dist'] < STOP_DISTANCE:
            print(f"⚠️ OBSTACLE CRITIQUE à {front['min_dist']:.2f}m - MARCHE ARRIÈRE")
            self.initiate_reverse(motor, front)
            return
        
        # Détection du meilleur chemin
        best_direction = self.find_best_path(front, left, right)
        
        # Calcul de la vitesse adaptative
        speed = self.calculate_speed(front['min_dist'])
        
        # Calcul du braquage avec lissage
        target_steering = best_direction['steering']
        smooth_steering = self.smooth_steering(target_steering)
        
        # Application des commandes
        motor.set_steering_objective(smooth_steering)
        motor.set_speed_objective(speed)
        
        print(f"📍 Direction: {best_direction['name']} | "
              f"Distance: {front['min_dist']:.2f}m | "
              f"Vitesse: {speed:.3f} | "
              f"Braquage: {smooth_steering:.2f}")

    def find_best_path(self, front, left, right):
        """Trouve la meilleure direction à prendre"""
        
        # Calcul des scores pour chaque direction
        paths = [
            {
                'name': 'AVANT',
                'steering': 0.0,
                'score': front['avg_dist'],
                'free': front['min_dist'] > SAFE_DISTANCE
            },
            {
                'name': 'GAUCHE',
                'steering': STEER_ANGLE,
                'score': left['avg_dist'] * 0.8,  # Léger malus pour les virages
                'free': left['min_dist'] > SLOW_DISTANCE
            },
            {
                'name': 'DROITE',
                'steering': -STEER_ANGLE,
                'score': right['avg_dist'] * 0.8,
                'free': right['min_dist'] > SLOW_DISTANCE
            }
        ]
        
        # Filtrer les chemins bloqués et trouver le meilleur
        free_paths = [p for p in paths if p['free']]
        
        if not free_paths:
            # Aucun chemin libre : choisir le moins pire
            return max(paths, key=lambda p: p['score'])
        
        # Choisir le chemin avec le meilleur score
        best = max(free_paths, key=lambda p: p['score'])
        
        # Ajustement fin du braquage si on va tout droit
        if best['name'] == 'AVANT':
            best['steering'] = self.fine_tune_steering(front['obstacles'])
        
        return best

    def fine_tune_steering(self, obstacles):
        """Ajustement fin du braquage pour rester au centre"""
        if not obstacles:
            return 0.0
        
        # Trouver l'obstacle le plus proche
        closest = min(obstacles, key=lambda o: o['distance'])
        angle = self.normalize_angle(closest['angle'])
        
        # Correction proportionnelle
        correction = -angle / SCAN_FRONT_DEG * 0.3  # Correction douce
        return max(-STEER_ANGLE, min(STEER_ANGLE, correction))

    def calculate_speed(self, min_distance):
        """Calcul adaptatif de la vitesse selon la distance"""
        if min_distance >= SAFE_DISTANCE:
            return FORWARD_SPEED
        elif min_distance <= STOP_DISTANCE:
            return 0.0
        else:
            # Interpolation linéaire
            factor = (min_distance - STOP_DISTANCE) / (SAFE_DISTANCE - STOP_DISTANCE)
            return SLOW_SPEED + (FORWARD_SPEED - SLOW_SPEED) * factor

    def smooth_steering(self, target_steering):
        """Lisse le braquage pour éviter les changements brusques"""
        smooth = (STEER_SMOOTHING * self.last_steering + 
                  (1 - STEER_SMOOTHING) * target_steering)
        self.last_steering = smooth
        return smooth

    def initiate_reverse(self, motor: Motor, front):
        """Démarre une séquence de marche arrière"""
        self.reverse_timer = 15  # Nombre de cycles en marche arrière
        
        # Tourner du côté opposé à l'obstacle
        if front['obstacles']:
            closest = min(front['obstacles'], key=lambda o: o['distance'])
            angle = self.normalize_angle(closest['angle'])
            # Braquer à l'opposé de l'obstacle
            steering = STEER_ANGLE if angle < 0 else -STEER_ANGLE
        else:
            steering = STEER_ANGLE  # Par défaut, tourner à gauche
        
        motor.set_steering_objective(steering)
        motor.set_speed_objective(BACKWARD_SPEED)

    def handle_reverse(self, motor: Motor, front):
        """Gère la marche arrière"""
        print(f"🔄 MARCHE ARRIÈRE ({self.reverse_timer} cycles restants)")
        # Maintenir les commandes de marche arrière
        # (déjà configurées dans initiate_reverse)