import time
import sys
import os
from control.State import State
from control.Motor import Motor
from control.Gamepad import Gamepad
from .LidarParserCpp import LidarParser

# Ajouter le chemin pour importer gps_simple_reader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gps.gps_simple_reader import SimpleGPSReader


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
STEER_ANGLE = 1         # Angle de braquage max
STEER_SMOOTHING = 0.7     # Lissage du braquage


class AutoDriveState(State):
    def __init__(self, use_gps=False, gps_host='localhost', gps_port=25001):
        print("Initializing AutoDrive...")
        self.lidar = LidarParser()
        self.last_steering = 0.0  # Pour le lissage
        self.reverse_timer = 0    # Compteur pour la marche arrière
        
        # GPS Reader
        self.gps = None
        self.use_gps = use_gps
        self._last_gps_print = 0  # Initialiser le compteur
        
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

    def run_single(self, motor: Motor, gamepad: Gamepad):
        """Exécute un cycle de contrôle"""
        points = self.lidar.get_points()
        if not points:
            time.sleep(0.05)
            return
        
        # Récupérer les données GPS si disponibles
        gps_data = None
        if self.use_gps and self.gps:
            gps_data = self.gps.get_position()
            
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
        left_scan = self.scan_sector(points,-STEERING_SCAN_ANGLE, -(SCAN_FRONT_DEG - 10), "GAUCHE")
        right_scan = self.scan_sector(points, (SCAN_FRONT_DEG - 10), STEERING_SCAN_ANGLE, "DROITE")
        largescans = self.scan_sector(points, -180, 180, "ALL")

        
        if self.reverse_timer > 0:
            self.handle_reverse(motor, largescans)
            self.reverse_timer -= 1
        else:
            self.navigate(motor, front_scan, left_scan, right_scan, largescans, gps_data)

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

    def navigate(self, motor: Motor, front, left, right, largescans, gps_data=None):
        """Logique principale de navigation avec intégration GPS"""
        
        # Obstacle très proche : marche arrière
        if front['min_dist'] < STOP_DISTANCE:
            # print(f"⚠️ OBSTACLE CRITIQUE à {front['min_dist']:.2f}m - MARCHE ARRIÈRE")
            self.initiate_reverse(motor, largescans)
            return
        
        best_direction = self.find_best_path(front, left, right, gps_data)
        speed = self.calculate_speed(front['min_dist'])
        target_steering = best_direction['steering']
        
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
        
        # Intégration GPS : ajouter un bonus au score selon la direction du goal
        if gps_data and gps_data.get('turn_angle') is not None and gps_data.get('goal_distance') is not None:
            turn_angle = gps_data['turn_angle']
            distance = gps_data['goal_distance']
            
            print(f"🧭 [GPS-NAV] Turn angle: {turn_angle:.1f}°, Distance: {distance:.1f}m")
            
            # Bonus GPS décroissant selon la distance (plus d'influence quand on est loin)
            if distance > 50.0:
                gps_weight = 0.5  # Influence forte quand on est loin
            elif distance > 20.0:
                gps_weight = 0.3  # Influence moyenne
            elif distance > 10.0:
                gps_weight = 0.25  # Influence faible
            else:
                gps_weight = 0.15  # Très faible quand proche (priorité aux obstacles)
            
            print(f"🎯 [GPS-NAV] GPS weight: {gps_weight}")
            
            # Calculer le bonus pour chaque direction selon l'angle de virage GPS
            for path in paths:
                old_score = path['score']
                if turn_angle < -10:  # Tourner à gauche
                    if path['name'] == 'GAUCHE':
                        path['score'] += gps_weight * 2.0
                    elif path['name'] == 'AVANT':
                        path['score'] += gps_weight * 0.5
                elif turn_angle > 10:  # Tourner à droite
                    if path['name'] == 'DROITE':
                        path['score'] += gps_weight * 2.0
                    elif path['name'] == 'AVANT':
                        path['score'] += gps_weight * 0.5
                else:  # Tout droit (on course)
                    if path['name'] == 'AVANT':
                        path['score'] += gps_weight * 2.0
                
                if path['score'] != old_score:
                    print(f"   [{path['name']}] Score: {old_score:.2f} → {path['score']:.2f} (bonus: +{path['score']-old_score:.2f})")
        else:
            if gps_data:
                print(f"⚠️  [GPS-NAV] GPS data incomplete: turn_angle={gps_data.get('turn_angle')}, distance={gps_data.get('goal_distance')}")

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
        if avg_dist > 3.5 * vitesse_factor:
            # Beaucoup d'espace : virage très doux
            factor = 0.3
        elif avg_dist > 2.5 * vitesse_factor:
            # Espace confortable : virage doux
            factor = 0.5
        elif avg_dist > 1.8 * vitesse_factor:
            # Espace moyen : virage modéré
            factor = 0.7
        else:
            # Peu d'espace : virage prononcé
            factor = 0.9 
        
        # Ajuster selon la distance minimale (sécurité)
        if min_dist < SLOW_DISTANCE:
            factor = min(1.0, factor + 0.2)  # Tourner plus fort si danger proche
        
        return direction * STEER_ANGLE * factor

    def fine_tune_steering(self, obstacles):
        """Ajustement fin du braquage pour rester au centre"""
        if not obstacles:
            return 0.0
        
        closest = min(obstacles, key=lambda o: o['distance'])
        angle = self.normalize_angle(closest['angle'])
        
        correction = -angle / SCAN_FRONT_DEG * 0.3 
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

    def smooth_steering(self, target_steering):
        """Lisse le braquage pour éviter les changements brusques"""
        smooth = (STEER_SMOOTHING * self.last_steering + 
                  (1 - STEER_SMOOTHING) * target_steering)
        self.last_steering = smooth
        return smooth

    def initiate_reverse(self, motor: Motor, largescans):
        """Démarre une séquence de marche arrière"""
        self.reverse_timer = 30  # Nombre de cycles en marche arrière
        
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
        print(f"🔄 MARCHE ARRIÈRE ({self.reverse_timer} cycles restants)")
        if (self.reverse_timer < 10):
            # print("🔙 FIN DE MARCHE ARRIÈRE dans 10 cycle restet steering")
            motor.set_steering_objective(0.0);
