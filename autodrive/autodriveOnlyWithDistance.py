import time
import sys
import os
from control.State import State
from control.Motor import Motor
from control.Gamepad import Gamepad
from .LidarParserCpp import LidarParser
from .LidarSender import LidarSender

# Ajouter le chemin pour importer gps_simple_reader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gps.gps_simple_reader import SimpleGPSReader


vitesse_factor = 1.5;

# DIRECTION DRIVE PARAMETERS
SCAN_FRONT_DEG = 25   # Élargi pour mieux détecter les ouvertures
SAFE_DISTANCE = 4.0   # Distance de sécurité pour ralentir
SLOW_DISTANCE = 1.5   # Distance de ralentissement
STOP_DISTANCE = 0.6   # Distance d'arrêt

MAX_SPEED_LIMIT = 0.4  # Ne jamais dépasser cette vitesse

FORWARD_SPEED = 0.06   # Vitesse maximale souhaitée
BACKWARD_SPEED = -0.04 # Vitesse de recul
SLOW_SPEED = 0.04   # Vitesse minimale

# STEERING AVOIDANCE PARAMETERS
STEERING_SCAN_ANGLE = 65  # Angle de scan pour trouver les ouvertures
STEER_ANGLE = 1         # Angle de braquage max

REVERSE_DURATION = 1.5    # Durée de la marche arrière en secondes

# STEERING SMOOTHING
STEERING_DEADBAND = 0.03
STEERING_SMOOTH_ALPHA = 0.35
STEERING_SMOOTH_ALPHA_FAST = 0.6
STEERING_RATE_LIMIT = 0.12
STEERING_RATE_LIMIT_FAST = 0.25

# STUCK / ESCAPE
STUCK_TIME_THRESHOLD = 1.2
ESCAPE_REVERSE_DURATION = 1.0
ESCAPE_FORWARD_DURATION = 1.0
ESCAPE_COOLDOWN = 3.0
PATH_SWITCH_HYSTERESIS = 0.35


class AutoDriveState(State):
    def __init__(self, use_gps=False, gps_host='localhost', gps_port=25001):
        print("Initializing AutoDrive...")
        self.lidar = LidarParser()
        # self.lidar = LidarParserUDP(host='127.0.0.1', port=8888)
        self.sender = LidarSender(host='127.0.0.1', port=8888)
        self.reverse_end_time = 0  # Fin de la marche arrière (timestamp)
        
        # GPS Reader
        self.gps = None
        self.use_gps = use_gps
        self._last_gps_print = 0  # Initialiser le compteur
        self._last_gps_update = 0  # Dernier update GPS
        self._last_goal_distances = []  # Les 5 dernières distances au goal
        self._distance_improving = True  # Est-ce qu'on se rapproche?
        self._cached_gps_data = None  # Données GPS en cache
        self._steering_filtered = 0.0
        self._last_choice = 'AVANT'
        self._blocked_since = None
        self._escape_end_time = 0.0
        self._escape_switch_time = 0.0
        self._escape_dir = 1
        self._last_escape_time = 0.0
        
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
            
        # Send points via UDP
        if hasattr(self, 'sender'):
            self.sender.send_points(points)
        
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
        left_scan = self.scan_sector(points,-STEERING_SCAN_ANGLE, -(SCAN_FRONT_DEG - 10), "GAUCHE")
        right_scan = self.scan_sector(points, (SCAN_FRONT_DEG - 10), STEERING_SCAN_ANGLE, "DROITE")
        largescans = self.scan_sector(points, -180, 180, "ALL")

        
        now = time.time()
        should_escape = self.update_blocked_state(front_scan, left_scan, right_scan, now)

        if self.is_escape_active(now):
            self.handle_escape(motor, front_scan, left_scan, right_scan)
            return

        if should_escape:
            self.initiate_escape(left_scan, right_scan)
            self.handle_escape(motor, front_scan, left_scan, right_scan)
            return

        if now < self.reverse_end_time:
            self.handle_reverse(motor, largescans)
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
        
        # Vérifier si on est arrivé à destination (GPS uniquement sur distance)
        if gps_data and gps_data.get('goal_distance') is not None:
            goal_distance = gps_data['goal_distance']
            if goal_distance <= 2.0:
                print(f"🎯 OBJECTIF ATTEINT! Distance: {goal_distance:.2f}m - ARRÊT")
                self.apply_steering(motor, 0.0, aggressive=True)
                self.apply_speed(motor, 0.0)
                return
        
        # Obstacle très proche : marche arrière
        if front['min_dist'] < STOP_DISTANCE:
            # print(f"⚠️ OBSTACLE CRITIQUE à {front['min_dist']:.2f}m - MARCHE ARRIÈRE")
            self.initiate_reverse(motor, largescans)
            return
        
        best_direction = self.find_best_path(front, left, right, gps_data)
        speed = self.calculate_speed(front['min_dist'])
        target_steering = best_direction['steering']
        
        self.apply_steering(motor, target_steering)
        self.apply_speed(motor, speed)
        


    def find_best_path(self, front, left, right, gps_data=None):
        """Trouve la meilleure direction à prendre avec braquage proportionnel et GPS"""
        
        # Calculer le braquage proportionnel basé sur l'espace disponible
        right_steering = self.calculate_proportional_steering(right, 1.0)   # Positif = droite
        left_steering = self.calculate_proportional_steering(left, -1.0)    # Négatif = gauche
        front_steering = self.compute_corridor_steering(left, right)

        
        # Calcul des scores pour chaque direction
        paths = [
            {
                'name': 'AVANT',
                'steering': front_steering,
                'score': self.compute_path_score(front, front_steering),
                'free': front['min_dist'] > SLOW_DISTANCE
            },
            {
                'name': 'DROITE',
                'steering': right_steering,
                'score': self.compute_path_score(right, right_steering) - 0.2,  # Léger malus pour les virages
                'free': right['min_dist'] > SLOW_DISTANCE
            },
            {
                'name': 'GAUCHE',
                'steering': left_steering,
                'score': self.compute_path_score(left, left_steering) - 0.2,
                'free': left['min_dist'] > SLOW_DISTANCE
            }
        ]

        # Biais vers l'avant si c'est vraiment dégagé (réduit l'oscillation)
        if front['min_dist'] > SAFE_DISTANCE:
            paths[0]['score'] += 0.6
        
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

        # Hystérésis: éviter de changer de direction pour un gain trop faible
        last = next((p for p in free_paths if p['name'] == self._last_choice), None)
        if last and (best['score'] - last['score']) < PATH_SWITCH_HYSTERESIS:
            best = last
        
        # Ajustement fin SEULEMENT si on va droit ET qu'il y a un obstacle très proche et décentré
        if best['name'] == 'AVANT' and front['min_dist'] < SAFE_DISTANCE * 0.7:
            fine = self.fine_tune_steering(front['obstacles'])
            best['steering'] = self.clamp(best['steering'] + fine, -STEER_ANGLE, STEER_ANGLE)

        # print(f"✅ Chemin choisi: {best['name']} (steering={best['steering']:+.2f})")
        self._last_choice = best['name']
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
        max_forward = min(FORWARD_SPEED, MAX_SPEED_LIMIT)
        if min_distance >= SAFE_DISTANCE:
            return max_forward
        elif min_distance <= STOP_DISTANCE:
            return 0.0
        else:
            factor = (min_distance - STOP_DISTANCE) / (SAFE_DISTANCE - STOP_DISTANCE)
            return min(
                max_forward,
                SLOW_SPEED + (max_forward - SLOW_SPEED) * factor
            )

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
        
        self.apply_steering(motor, steering, aggressive=True)
        self.apply_speed(motor, BACKWARD_SPEED)

    def handle_reverse(self, motor: Motor, largescans):
        """Gère la marche arrière"""
        remaining = self.reverse_end_time - time.time()
        print(f"🔄 MARCHE ARRIÈRE ({remaining:.1f}s restantes)")
        if remaining < 0.3:
            self.apply_steering(motor, 0.0, aggressive=True)

    def clamp(self, value, min_value, max_value):
        return max(min_value, min(max_value, value))

    def apply_steering(self, motor: Motor, target_steering: float, aggressive: bool = False):
        if abs(target_steering) < STEERING_DEADBAND:
            target_steering = 0.0

        alpha = STEERING_SMOOTH_ALPHA_FAST if aggressive else STEERING_SMOOTH_ALPHA
        rate_limit = STEERING_RATE_LIMIT_FAST if aggressive else STEERING_RATE_LIMIT

        previous = self._steering_filtered
        smoothed = previous + alpha * (target_steering - previous)
        delta = smoothed - previous
        if delta > rate_limit:
            smoothed = previous + rate_limit
        elif delta < -rate_limit:
            smoothed = previous - rate_limit

        smoothed = self.clamp(smoothed, -STEER_ANGLE, STEER_ANGLE)
        self._steering_filtered = smoothed
        motor.set_steering_objective(smoothed)

    def apply_speed(self, motor: Motor, speed: float):
        speed = self.clamp(speed, -MAX_SPEED_LIMIT, MAX_SPEED_LIMIT)
        motor.set_speed_objective(speed)

    def compute_corridor_steering(self, left, right):
        denom = max(0.1, left['avg_dist'] + right['avg_dist'])
        balance = (right['avg_dist'] - left['avg_dist']) / denom
        balance = self.clamp(balance, -0.6, 0.6)
        return balance * (STEER_ANGLE * 0.6)

    def compute_path_score(self, sector_scan, steering):
        avg_dist = min(sector_scan['avg_dist'], 6.0)
        min_dist = min(sector_scan['min_dist'], 6.0)
        density = sector_scan['count'] / max(1.0, avg_dist * 8.0)

        score = avg_dist * 0.65 + min_dist * 0.35
        score -= density * 0.5
        score -= abs(steering) * 0.15
        if sector_scan['min_dist'] < SLOW_DISTANCE:
            score -= 0.8
        return score

    def update_blocked_state(self, front, left, right, now):
        if self.is_escape_active(now):
            return False

        blocked = (
            front['min_dist'] < STOP_DISTANCE and
            left['min_dist'] < STOP_DISTANCE and
            right['min_dist'] < STOP_DISTANCE
        )
        if blocked:
            if self._blocked_since is None:
                self._blocked_since = now
        else:
            self._blocked_since = None

        if self._blocked_since is None:
            return False

        if (now - self._blocked_since) < STUCK_TIME_THRESHOLD:
            return False

        if (now - self._last_escape_time) < ESCAPE_COOLDOWN:
            return False

        return True

    def is_escape_active(self, now=None):
        if now is None:
            now = time.time()
        return now < self._escape_end_time

    def initiate_escape(self, left, right):
        now = time.time()
        self._last_escape_time = now
        self._escape_switch_time = now + ESCAPE_REVERSE_DURATION
        self._escape_end_time = now + ESCAPE_REVERSE_DURATION + ESCAPE_FORWARD_DURATION
        self._blocked_since = None
        self.reverse_end_time = 0.0

        if left['avg_dist'] > right['avg_dist'] + 0.1:
            self._escape_dir = -1
        elif right['avg_dist'] > left['avg_dist'] + 0.1:
            self._escape_dir = 1
        else:
            self._escape_dir *= -1

        print("🧭 [ESCAPE] Blocage détecté, tentative de sortie...")

    def handle_escape(self, motor: Motor, front, left, right):
        now = time.time()
        steering = self._escape_dir * STEER_ANGLE

        if now < self._escape_switch_time:
            # Phase 1: reculer en braquant fort
            self.apply_steering(motor, steering, aggressive=True)
            self.apply_speed(motor, BACKWARD_SPEED)
            return

        # Phase 2: avancer doucement en braquant vers la sortie
        forward_steer = -self._escape_dir * STEER_ANGLE * 0.8
        if front['min_dist'] < SLOW_DISTANCE:
            self.apply_steering(motor, steering, aggressive=True)
            self.apply_speed(motor, BACKWARD_SPEED)
            return

        self.apply_steering(motor, forward_steer, aggressive=True)
        self.apply_speed(motor, min(SLOW_SPEED, MAX_SPEED_LIMIT))
