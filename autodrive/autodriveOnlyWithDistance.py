import time
import sys
import os
from control.State import State
from control.Motor import Motor
from control.Gamepad import Gamepad
from .LidarParserCpp import LidarParser
from .LidarSender import LidarUdpServer

# Ajouter le chemin pour importer gps_simple_reader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gps.gps_simple_reader import SimpleGPSReader


vitesse_factor = 1.5

# DIRECTION DRIVE PARAMETERS
SCAN_FRONT_DEG = 15   # Élargi pour mieux détecter les ouvertures
SAFE_DISTANCE = 3.5   # Distance de sécurité pour ralentir
SLOW_DISTANCE = 1.2   # Distance de ralentissement
STOP_DISTANCE = 0.45   # Distance d'arrêt

FORWARD_SPEED = 0.04   # Vitesse maximale augmentée
BACKWARD_SPEED = -0.04 # Vitesse de recul
SLOW_SPEED = 0.02   # Vitesse minimale

# STEERING AVOIDANCE PARAMETERS
STEERING_SCAN_ANGLE = 60  # Angle de scan pour trouver les ouvertures
STEER_ANGLE = 1        # Angle de braquage max

REVERSE_DURATION = 1    # Durée de la marche arrière en secondes

# WALL CENTERING PARAMETERS
WALL_SCAN_MIN_ANGLE = 70   # Angle min pour détecter les murs latéraux
WALL_SCAN_MAX_ANGLE = 110  # Angle max pour détecter les murs latéraux
WALL_MIN_POINTS = 6        # Minimum de points pour considérer un mur détecté
WALL_DETECT_MAX = 6.0      # Distance max pour considérer un mur
WALL_DETECT_MIN = 0.2      # Distance min pour ignorer le bruit
CENTERING_GAIN = 0.6       # Gain de correction (0..1)
CENTERING_MAX = 0.6        # Limite de correction (en fraction de STEER_ANGLE)


class AutoDriveState(State):
    def __init__(self, use_gps=False, gps_host='localhost', gps_port=25001):
        print("Initializing AutoDrive...")
        self.lidar = LidarParser()
        # self.lidar = LidarParserUDP(host='127.0.0.1', port=8888)
       # self.sender = LidarUdpServer(port=8888)
        self.reverse_end_time = 0  # Fin de la marche arrière (timestamp)
        
        # GPS Reader
        self.gps = None
        self.use_gps = use_gps
        self._last_gps_print = 0  # Initialiser le compteur
        self._last_gps_update = 0  # Dernier update GPS
        self._last_goal_distances = []  # Les 5 dernières distances au goal
        self._distance_improving = True  # Est-ce qu'on se rapproche?
        self._cached_gps_data = None  # Données GPS en cache
        
        if self.use_gps:
            print(f"🛰️  [DEBUG] Activation du GPS ({gps_host}:{gps_port})...")
            self.gps = SimpleGPSReader(gps_host, gps_port)
            if not self.gps.start():
                print("⚠️  [DEBUG] Mode LIDAR seul (GPS non disponible)")
                self.use_gps = False
            else:
                print("✅ [DEBUG] GPS connecté avec succès")
        else:
            print("ℹ️  [DEBUG] Mode LIDAR seul (GPS désactivé)")
    
    def stop(self):
        self.lidar.stop()
        if self.gps:
            self.gps.stop()

    def process_lidar_passive(self):
        """Lit et envoie les données LIDAR sans agir sur les moteurs (Monitoring)"""
        points = self.lidar.get_points()
  #        if points and hasattr(self, 'sender'):
   #         self.sender.send_points(points)

    def run_single(self, motor: Motor, gamepad: Gamepad):
        """Exécute un cycle de contrôle"""
        points = self.lidar.get_points()
        if not points:
            #time.sleep(0.05)
            return
            
        # Send points via UDP
       # if hasattr(self, 'sender'):
        #    self.sender.send_points(points)
        
        # Récupérer les données GPS si disponibles (toutes les 0.5s)
        gps_data = None
        if self.use_gps and self.gps:
            current_time = time.time()
            if current_time - self._last_gps_update >= 0.5:
                gps_data = self.gps.get_position()
                self._cached_gps_data = gps_data
                self._last_gps_update = current_time
            else:
                gps_data = self._cached_gps_data
            
            # Debug GPS
            if gps_data:
                # Afficher les données GPS toutes les 2 secondes
                if time.time() - self._last_gps_print >= 2.0:
                    if gps_data.get('goal_distance') is not None and gps_data.get('turn_angle') is not None:
                        print(f"📍 [GPS] Pos: {gps_data['lat']:.6f}°, {gps_data['lon']:.6f}° | Goal: {gps_data['goal_distance']:.1f}m @ {gps_data['goal_bearing']:.0f}° | Turn: {gps_data['turn_angle']:.0f}°")
                    else:
                        heading_str = f"{gps_data['heading']:.0f}°" if gps_data.get('heading') else "N/A"
                        print(f"📍 [GPS] Pos: {gps_data['lat']:.6f}°, {gps_data['lon']:.6f}° | Heading: {heading_str} | Goal data: distance={gps_data.get('goal_distance')}, bearing={gps_data.get('goal_bearing')}, turn={gps_data.get('turn_angle')}")
                    self._last_gps_print = time.time()
            else:
                if time.time() - self._last_gps_print >= 2.0:
                    print("⚠️  [GPS] Aucune donnée GPS reçue")
                    self._last_gps_print = time.time()
        
        front_scan = self.scan_sector(points, -SCAN_FRONT_DEG, SCAN_FRONT_DEG, "AVANT")
        left_scan = self.scan_sector(points, -STEERING_SCAN_ANGLE, -(SCAN_FRONT_DEG - 10), "GAUCHE")
        right_scan = self.scan_sector(points, (SCAN_FRONT_DEG - 10), STEERING_SCAN_ANGLE, "DROITE")
        wall_left_scan = self.scan_sector(points, WALL_SCAN_MIN_ANGLE, WALL_SCAN_MAX_ANGLE, "MUR_GAUCHE")
        wall_right_scan = self.scan_sector(points, -WALL_SCAN_MAX_ANGLE, -WALL_SCAN_MIN_ANGLE, "MUR_DROIT")
        largescans = self.scan_sector(points, -180, 180, "ALL")

        
        if time.time() < self.reverse_end_time:
            self.handle_reverse(motor, largescans)
        else:
            self.navigate(motor, front_scan, left_scan, right_scan, wall_left_scan, wall_right_scan, largescans, gps_data)

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

    def navigate(self, motor: Motor, front, left, right, wall_left, wall_right, largescans, gps_data=None):
        """Logique principale de navigation avec intégration GPS"""
        
        # Vérifier si on est arrivé à destination (GPS uniquement sur distance)
        if gps_data and gps_data.get('goal_distance') is not None:
            goal_distance = gps_data['goal_distance']
            if goal_distance <= 2.0:
                print(f"🎯 OBJECTIF ATTEINT! Distance: {goal_distance:.2f}m - ARRÊT")
                motor.set_steering_objective(0.0)
                motor.set_speed_objective(0.0)
                return
        
        # Obstacle très proche : marche arrière
        if front['min_dist'] < STOP_DISTANCE:
            # print(f"⚠️ OBSTACLE CRITIQUE à {front['min_dist']:.2f}m - MARCHE ARRIÈRE")
            self.initiate_reverse(motor, largescans)
            return
        
        best_direction = self.find_best_path(front, left, right, gps_data)
        speed = self.calculate_speed(front['min_dist'])
        target_steering = best_direction['steering']

        # Centrage par détection des murs gauche/droite (priorité faible)
        if best_direction['name'] == 'AVANT' and front['min_dist'] > SLOW_DISTANCE:
            centering = self.calculate_wall_centering(wall_left, wall_right, front['min_dist'])
            if centering != 0.0:
                target_steering += centering
                target_steering = max(-STEER_ANGLE, min(STEER_ANGLE, target_steering))
        
        # Correction proactive : si un mur est très proche sur un côté,
        # forcer le braquage pour s'en éloigner avant de devoir reculer
        SIDE_DANGER_DIST = STOP_DISTANCE
        if left['min_dist'] < SIDE_DANGER_DIST or right['min_dist'] < SIDE_DANGER_DIST:
            if left['min_dist'] < right['min_dist']:
                # Mur proche à gauche → braquer à droite
                urgency = 1.0 - (left['min_dist'] / SIDE_DANGER_DIST)
                side_correction = STEER_ANGLE * urgency * 0.7
                target_steering = max(target_steering, side_correction)
            else:
                # Mur proche à droite → braquer à gauche
                urgency = 1.0 - (right['min_dist'] / SIDE_DANGER_DIST)
                side_correction = -STEER_ANGLE * urgency * 0.7
                target_steering = min(target_steering, side_correction)
            speed = min(speed, SLOW_SPEED)  # Ralentir aussi
        
        motor.set_steering_objective(target_steering)
        motor.set_speed_objective(speed)
        


    def find_best_path(self, front, left, right, gps_data=None):
        """Trouve la meilleure direction à prendre avec braquage proportionnel et GPS"""
        
        # Calculer le braquage proportionnel basé sur l'espace disponible
        right_steering = self.calculate_proportional_steering(right, 1.0)   # Positif = droite
        left_steering = self.calculate_proportional_steering(left, -1.0)    # Négatif = gauche
        front_steering = self.calculate_proportional_steering(front, 0.0) 

        
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
                'score': right['avg_dist'] * 0.8,  # Léger malus pour les virages
                'free': right['min_dist'] > SLOW_DISTANCE
            },
            {
                'name': 'GAUCHE',
                'steering': left_steering,
                'score': left['avg_dist'] * 0.8,
                'free': left['min_dist'] > SLOW_DISTANCE
            }
        ]
        
        # GPS : utiliser la distance pour guider la navigation
        if gps_data and gps_data.get('goal_distance') is not None:
            distance = gps_data['goal_distance']
            
            # Ajouter la distance à l'historique (garder les 5 dernières)
            self._last_goal_distances.append(distance)
            if len(self._last_goal_distances) > 5:
                self._last_goal_distances.pop(0)
            
            # Déterminer la tendance sur les 5 derniers points
            if len(self._last_goal_distances) >= 3:
                # Calculer la moyenne des 2 premières et 2 dernières distances
                old_avg = sum(self._last_goal_distances[:2]) / 2
                new_avg = sum(self._last_goal_distances[-2:]) / 2
                distance_change = new_avg - old_avg
                
                self._distance_improving = distance_change < -0.3  # Seuil réduit pour plus de sensibilité
                
                if self._distance_improving:
                    print(f"📉 [GPS] Distance: {distance:.1f}m ✅ (tendance: {-distance_change:.1f}m | historique: {len(self._last_goal_distances)} pts)")
                else:
                    print(f"📈 [GPS] Distance: {distance:.1f}m ⚠️ (tendance: +{distance_change:.1f}m | historique: {len(self._last_goal_distances)} pts)")
            else:
                print(f"📏 [GPS] Distance: {distance:.1f}m (collecte: {len(self._last_goal_distances)}/5 pts)")
            
            # Influence GPS selon la distance et la tendance
            if len(self._last_goal_distances) >= 3:
                if distance > 50.0:
                    gps_weight = 2.0
                elif distance > 20.0:
                    gps_weight = 1.5
                elif distance > 10.0:
                    gps_weight = 1.0
                elif distance > 5.0:
                    gps_weight = 0.6
                else:
                    gps_weight = 0.3
                
                # Si on se rapproche: favoriser AVANT (continuer)
                # Si on s'éloigne: favoriser les côtés (chercher une autre direction)
                if self._distance_improving:
                    # On se rapproche: continuer tout droit!
                    for path in paths:
                        if path['name'] == 'AVANT':
                            path['score'] += gps_weight * 2.5
                            print(f"   🎯 [AVANT] Bonus GPS: +{gps_weight * 2.5:.2f} (tendance positive!)")
                else:
                    # On s'éloigne: essayer de tourner pour trouver le goal
                    for path in paths:
                        if path['name'] in ['GAUCHE', 'DROITE']:
                            path['score'] += gps_weight * 1.5
                            print(f"   🔄 [{path['name']}] Bonus GPS: +{gps_weight * 1.5:.2f} (chercher nouvelle direction)")

        free_paths = [p for p in paths if p['free']]
        
        if not free_paths:
            best = max(paths, key=lambda p: p['score'])
            # print(f"⚠️ Aucun chemin libre! Meilleur choix: {best['name']} (steering={best['steering']:+.2f})")
            return best
        
        best = max(free_paths, key=lambda p: p['score'])
        
        # Ajustement fin SEULEMENT si on va droit ET qu'il y a un obstacle très proche et décentré
        if best['name'] == 'AVANT' and front['min_dist'] < SAFE_DISTANCE * 0.7:
            best['steering'] = self.fine_tune_steering(front['obstacles'])
        
        # print(f"✅ Chemin choisi: {best['name']} (steering={best['steering']:+.2f})")
        return best
    
    def calculate_proportional_steering(self, sector_scan, direction):
        """Calcule un braquage proportionnel basé sur la distance et densité d'obstacles"""
        min_dist = sector_scan['min_dist']
        avg_dist = sector_scan['avg_dist']
        
        # Plus l'espace est grand, moins on tourne fort
        # Utiliser la distance moyenne pour un meilleur jugement
        if avg_dist > 4.0 * vitesse_factor:
            # Beaucoup d'espace : virage très doux
            factor = 0.35
        elif avg_dist > 3.0 * vitesse_factor:
            # Espace confortable : virage doux
            factor = 0.55
        elif avg_dist > 2.2 * vitesse_factor:
            # Espace moyen : virage modéré
            factor = 0.75
        else:
            # Peu d'espace : virage prononcé
            factor = 0.95 
        
        # Ajuster selon la distance minimale (sécurité)
        if min_dist < STOP_DISTANCE * 2:
            factor = 1.0  # Très proche : braquage max
        elif min_dist < SLOW_DISTANCE:
            factor = min(1.0, factor + 0.4)  # Proche : tourner nettement plus fort
        elif min_dist < SAFE_DISTANCE * 0.5:
            factor = min(1.0, factor + 0.2)  # Moyen : correction modérée
        
        return direction * STEER_ANGLE * factor

    def calculate_wall_centering(self, wall_left, wall_right, front_min_dist):
        """Calcule un braquage pour rester centré entre deux murs"""
        left_dist = self._wall_distance(wall_left)
        right_dist = self._wall_distance(wall_right)
        
        if left_dist is None or right_dist is None:
            return 0.0
        
        # Normaliser l'écart pour rester stable même si les distances changent
        denom = max(left_dist + right_dist, 0.001)
        error = (right_dist - left_dist) / denom  # + => tourner à droite
        
        # Réduire la correction si un obstacle est plus proche devant
        if front_min_dist < SAFE_DISTANCE:
            front_factor = max(0.3, (front_min_dist - SLOW_DISTANCE) / (SAFE_DISTANCE - SLOW_DISTANCE))
        else:
            front_factor = 1.0
        
        correction = error * CENTERING_GAIN * front_factor
        correction = max(-CENTERING_MAX, min(CENTERING_MAX, correction))
        return correction * STEER_ANGLE

    def _wall_distance(self, wall_scan):
        """Retourne une distance fiable du mur ou None si non détecté"""
        if wall_scan['count'] < WALL_MIN_POINTS:
            return None
        dist = wall_scan['avg_dist']
        if dist == float('inf'):
            return None
        if dist < WALL_DETECT_MIN or dist > WALL_DETECT_MAX:
            return None
        return dist

    def fine_tune_steering(self, obstacles):
        """Ajustement fin du braquage pour rester au centre"""
        if not obstacles:
            return 0.0
        
        closest = min(obstacles, key=lambda o: o['distance'])
        angle = self.normalize_angle(closest['angle'])
        
        # Plus l'obstacle est proche, plus la correction est forte
        proximity_boost = max(1.0, SAFE_DISTANCE / max(closest['distance'], 0.1))
        correction = -angle / SCAN_FRONT_DEG * 0.5 * proximity_boost
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
        self.reverse_end_time = time.time() + REVERSE_DURATION
        
        if largescans['obstacles']:
            left_sector = [o for o in largescans['obstacles'] if 30 <= self.normalize_angle(o['angle']) <= 150]
            right_sector = [o for o in largescans['obstacles'] if -150 <= self.normalize_angle(o['angle']) <= -30]
            
            left_min = min([o['distance'] for o in left_sector]) if left_sector else float('inf')
            right_min = min([o['distance'] for o in right_sector]) if right_sector else float('inf')
            
            
            if left_min > right_min:
                steering = -STEER_ANGLE
            else:
                steering = STEER_ANGLE 
        else:
            steering = 0.0 
        
        motor.set_steering_objective(steering)
        motor.set_speed_objective(BACKWARD_SPEED)

    def handle_reverse(self, motor: Motor, largescans):
        """Gère la marche arrière"""
        remaining = self.reverse_end_time - time.time()
        print(f"🔄 MARCHE ARRIÈRE ({remaining:.1f}s restantes)")
        if remaining < 0.3:
            motor.set_steering_objective(0.0)
