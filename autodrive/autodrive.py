import time
import sys
import os
from control.State import State
from control.Motor import Motor
from control.Gamepad import Gamepad
from .LidarParserCpp import LidarParser



vitesse_factor = 1.5;

# DIRECTION DRIVE PARAMETERS
SCAN_FRONT_DEG = 25   # Élargi pour mieux détecter les ouvertures
SAFE_DISTANCE = 4.0   # Distance de sécurité pour ralentir
SLOW_DISTANCE = 1.5   # Distance de ralentissement
STOP_DISTANCE = 0.6   # Distance d'arrêt

FORWARD_SPEED = 0.06   # Vitesse maximale augmentée
BACKWARD_SPEED = -0.04 # Vitesse de recul
SLOW_SPEED = 0.04   # Vitesse minimale

# STEERING AVOIDANCE PARAMETERS
STEERING_SCAN_ANGLE = 65  # Angle de scan pour trouver les ouvertures
STEER_ANGLE = 1         # Angle de braquage max (1 = 45° réel)


class AutoDriveState(State):
    def __init__(self, use_gps=False, gps_host='localhost'):
        print("Initializing AutoDrive...")
        # self.lidar = LidarParser()
        self.lidar = LidarParser();
        self.last_steering = 0.0  # Pour le lissage
        self.reverse_timer = 0    # Compteur pour la marche arrière
        self.reverse_steering = 0.0  # Direction pendant la marche arrière
        
        # GPS Reader
        self.gps = None
        self.use_gps = use_gps
        self._last_gps_print = 0  # Initialiser le compteur
        self._last_gps_update = 0  # Dernier update GPS
        self._last_goal_distances = []  # Les 5 dernières distances au goal
        self._distance_improving = True  # Est-ce qu'on se rapproche?
        self._cached_gps_data = None  # Données GPS en cache
        
    
    def stop(self):
        self.lidar.stop()
        if self.gps:
            self.gps.stop()

    def run_single(self, motor: Motor, gamepad: Gamepad):
        """Exécute un cycle de contrôle"""
        points = self.lidar.get_points()
        if not points:
            time.sleep(0.05)
            return
                
        front_scan = self.scan_sector(points, -SCAN_FRONT_DEG, SCAN_FRONT_DEG, "AVANT")
        left_scan = self.scan_sector(points, -STEERING_SCAN_ANGLE, -SCAN_FRONT_DEG, "GAUCHE")
        right_scan = self.scan_sector(points, SCAN_FRONT_DEG, STEERING_SCAN_ANGLE, "DROITE")
        largescans = self.scan_sector(points, -180, 180, "ALL")

        
        if self.reverse_timer > 0:
            self.handle_reverse(motor, largescans)
            self.reverse_timer -= 1
        else:
            self.navigate(motor, front_scan, left_scan, right_scan, largescans)

    def scan_sector(self, points, min_angle, max_angle, sector_name=""):
        """Analyse un secteur angulaire et retourne les statistiques"""
        obstacles = []
        for p in points:
            angle = self.normalize_angle(p['angle'])
            if min_angle <= angle <= max_angle:
                obstacles.append(p)
        
        if not obstacles:
            # if sector_name:
            #     print(f"  [{sector_name}] ({min_angle:+4.0f}° to {max_angle:+4.0f}°): ⚫ NO DATA")
            return {'min_dist': float('inf'), 'avg_dist': float('inf'), 'count': 0, 'obstacles': []}
        
        distances = [o['distance'] for o in obstacles]
        result = {
            'min_dist': min(distances),
            'avg_dist': sum(distances) / len(distances),
            'count': len(distances),
            'obstacles': obstacles
        }
        # if sector_name:
        #     status = "🟢" if result['min_dist'] > SAFE_DISTANCE else "🟡" if result['min_dist'] > SLOW_DISTANCE else "🔴"
        #     print(f"  [{sector_name}] ({min_angle:+4.0f}° to {max_angle:+4.0f}°): {status} {result['min_dist']:.2f}m (avg:{result['avg_dist']:.2f}m, pts:{result['count']})")
        
        
        return result

    def normalize_angle(self, angle):
        """Normalise un angle entre -180 et 180"""
        angle = angle % 360
        if angle > 180:
            angle -= 360
        return angle

    def navigate(self, motor: Motor, front, left, right, largescans):
        """Logique principale de navigation avec intégration GPS"""
        
        
        # Obstacle très proche : marche arrière
        if front['min_dist'] < STOP_DISTANCE:
            # print(f"⚠️ OBSTACLE CRITIQUE à {front['min_dist']:.2f}m - MARCHE ARRIÈRE")
            self.initiate_reverse(motor, largescans)
            return
        
        speed = self.calculate_speed(front['min_dist'])
        
        # Si l'avant est bien dégagé, aller tout droit sans chercher d'alternative
        if front['min_dist'] > SAFE_DISTANCE * 1.2:
            motor.set_steering_objective(0.0)
            motor.set_speed_objective(speed)
            return
        
        best_direction = self.find_best_path(front, left, right)
        target_steering = best_direction['steering']
        
        motor.set_steering_objective(target_steering)
        motor.set_speed_objective(speed)
        


    def find_best_path(self, front, left, right):
        """Trouve la meilleure direction à prendre avec braquage proportionnel et GPS"""
        
        # Calculer le braquage proportionnel basé sur l'espace disponible
        right_steering = self.calculate_proportional_steering(right, 1.0)   # Positif = droite
        left_steering = self.calculate_proportional_steering(left, -1.0)    # Négatif = gauche
        # Front steering : correction fine pour rester centré
        front_steering = self.fine_tune_steering(front['obstacles'])

        # Calcul des scores pour chaque direction
        paths = [
            {
                'name': 'AVANT',
                'steering': front_steering,
                'score': front['avg_dist'],
                'free': front['min_dist'] > SAFE_DISTANCE
            },
            {
                'name': 'DROITE',
                'steering': right_steering,
                'score': right['avg_dist'] * 0.9,
                'free': right['min_dist'] > SLOW_DISTANCE
            },
            {
                'name': 'GAUCHE',
                'steering': left_steering,
                'score': left['avg_dist'] * 0.9,
                'free': left['min_dist'] > SLOW_DISTANCE
            }
        ]
        
        free_paths = [p for p in paths if p['free']]
        
        if not free_paths:
            best = max(paths, key=lambda p: p['score'])
            return best
        
        best = max(free_paths, key=lambda p: p['score'])
        return best
    
    def calculate_proportional_steering(self, sector_scan, direction):
        """Calcule un braquage proportionnel basé sur la distance et densité d'obstacles"""

        # MAX STEER_ANGLE == 45° => 1 = 45°     

        min_dist = sector_scan['min_dist']
        avg_dist = sector_scan['avg_dist']
        
        # Plus l'obstacle est proche, plus on tourne fort
        if avg_dist > 3.5 * vitesse_factor:
            # Beaucoup d'espace : virage modéré
            factor = 0.6
        elif avg_dist > 2.5 * vitesse_factor:
            # Espace confortable : virage franc
            factor = 0.75
        elif avg_dist > 1.8 * vitesse_factor:
            # Espace moyen : virage fort
            factor = 0.9
        else:
            # Peu d'espace : braquage max
            factor = 1.0
        
        # Ajuster selon la distance minimale (sécurité)
        if min_dist < SLOW_DISTANCE:
            factor = 1.0  # Braquage max si danger proche
        
        return direction * STEER_ANGLE * factor

    def fine_tune_steering(self, obstacles):
        """Ajustement fin du braquage pour rester au centre"""
        if not obstacles:
            return 0.0
        
        closest = min(obstacles, key=lambda o: o['distance'])
        
        # Si l'obstacle le plus proche est loin, pas besoin de corriger → tout droit
        if closest['distance'] > SAFE_DISTANCE:
            return 0.0
        
        angle = self.normalize_angle(closest['angle'])
        
        # Zone morte : si l'obstacle est quasi centré (±5°), ne pas corriger
        if abs(angle) < 5:
            return 0.0
        
        correction = -angle / SCAN_FRONT_DEG * 0.7
        return max(-STEER_ANGLE, min(STEER_ANGLE, correction))

    def calculate_speed(self, min_distance):
        """Calcul adaptatif de la vitesse selon la distance"""
        if min_distance >= SAFE_DISTANCE:
            return FORWARD_SPEED
        elif min_distance <= STOP_DISTANCE:
            return 0.0
        else:
            factor = (min_distance - STOP_DISTANCE) / (SAFE_DISTANCE - STOP_DISTANCE)
            return SLOW_SPEED + (FORWARD_SPEED - SLOW_SPEED) * factor


    def initiate_reverse(self, motor: Motor, largescans):
        """Démarre une séquence de marche arrière"""
        self.reverse_timer = 30  # Nombre de cycles en marche arrière
        
        if largescans['obstacles']:
            left_sector = [o for o in largescans['obstacles'] if 30 <= self.normalize_angle(o['angle']) <= 150]
            right_sector = [o for o in largescans['obstacles'] if -150 <= self.normalize_angle(o['angle']) <= -30]
            
            left_min = min([o['distance'] for o in left_sector]) if left_sector else float('inf')
            right_min = min([o['distance'] for o in right_sector]) if right_sector else float('inf')
            
            if left_min > right_min:
                self.reverse_steering = -STEER_ANGLE
            else:
                self.reverse_steering = STEER_ANGLE
        else:
            self.reverse_steering = 0.0
        
        motor.set_steering_objective(self.reverse_steering)
        motor.set_speed_objective(BACKWARD_SPEED)

    def handle_reverse(self, motor: Motor, largescans):
        """Gère la marche arrière"""
        print(f"🔄 MARCHE ARRIÈRE ({self.reverse_timer} cycles restants)")
        
        # Re-appliquer la vitesse à chaque cycle (sinon le motor loop la remet à 0)
        motor.set_speed_objective(BACKWARD_SPEED)
        
        if self.reverse_timer < 10:
            # Derniers cycles : roues droites pour sortir droit
            motor.set_steering_objective(0.0)
        else:
            # Maintenir le braquage choisi au début
            motor.set_steering_objective(self.reverse_steering)
