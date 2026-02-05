import time
import socket
import math
import threading
import os
import sys
from control.State import State
from control.Motor import Motor
from control.Gamepad import Gamepad
from .LidarParserCpp import LidarParser

# Import GPS FusionEngine
root_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, root_dir)

try:
    from fusion_engine_client.messages.core import PoseMessage
    from fusion_engine_client.messages.defs import yaw_to_heading
    from fusion_engine_client.parsers import FusionEngineDecoder
    GPS_AVAILABLE = True
except ImportError:
    print("⚠️ GPS FusionEngine non disponible")
    GPS_AVAILABLE = False


vitesse_factor = 1.5;

# DIRECTION DRIVE PARAMETERS
SCAN_FRONT_DEG = 25   # Élargi pour mieux détecter les ouvertures
SAFE_DISTANCE = 3.0   # Distance de sécurité pour ralentir
SLOW_DISTANCE = 1.5   # Distance de ralentissement
STOP_DISTANCE = 0.55   # Distance d'arrêt

FORWARD_SPEED = 0.11   # Vitesse maximale augmentée
BACKWARD_SPEED = -0.04 # Vitesse de recul
SLOW_SPEED = 0.07      # Vitesse minimale

# STEERING AVOIDANCE PARAMETERS
STEERING_SCAN_ANGLE = 70  # Angle de scan pour trouver les ouvertures
STEER_ANGLE = 1         # Angle de braquage max
STEER_SMOOTHING = 0.7     # Lissage du braquage

# GPS PARAMETERS
GPS_HOST = "localhost"
GPS_PORT = 25000
GOAL_REACHED_DISTANCE = 2.0  # Distance en mètres pour considérer l'objectif atteint


class GPSNavigator:
    """Gère la connexion GPS et le calcul de navigation vers un objectif"""
    def __init__(self, hostname=GPS_HOST, port=GPS_PORT, goal_lat=None, goal_lon=None):
        self.hostname = hostname
        self.port = port
        self.goal_lat = goal_lat
        self.goal_lon = goal_lon
        
        # État GPS
        self.current_lat = None
        self.current_lon = None
        self.current_alt = None
        self.heading_deg = None
        self.bearing_to_goal = None
        self.distance_to_goal = None
        self.solution_type = None
        
        self.transport = None
        self.decoder = None
        self.running = False
        self.thread = None
        self.last_update = 0
        
    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        """Calcule le cap de point 1 vers point 2 en degrés (0-360)"""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        lon_diff_rad = math.radians(lon2 - lon1)
        
        x = math.sin(lon_diff_rad) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(lon_diff_rad)
        
        bearing_rad = math.atan2(x, y)
        bearing_deg = math.degrees(bearing_rad)
        bearing_deg = (bearing_deg + 360) % 360
        
        return bearing_deg
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calcule la distance entre deux points en mètres (Haversine)"""
        R = 6371000  # Rayon de la Terre en mètres
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        distance = R * c
        return distance
    
    def smallest_angle_diff(self, from_deg, to_deg):
        """Retourne la plus petite différence d'angle signée (-180..180]"""
        return (to_deg - from_deg + 180.0) % 360.0 - 180.0
    
    def start(self):
        """Démarre la connexion GPS en thread séparé"""
        if not GPS_AVAILABLE:
            print("⚠️ GPS non disponible")
            return False
            
        try:
            self.transport = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.transport.settimeout(5.0)
            self.transport.connect((socket.gethostbyname(self.hostname), self.port))
            self.decoder = FusionEngineDecoder()
            self.running = True
            
            self.thread = threading.Thread(target=self._gps_loop, daemon=True)
            self.thread.start()
            
            print(f"✅ GPS connecté à {self.hostname}:{self.port}")
            if self.goal_lat and self.goal_lon:
                print(f"🎯 Objectif: {self.goal_lat:.6f}°, {self.goal_lon:.6f}°")
            return True
        except Exception as e:
            print(f"❌ Erreur connexion GPS: {e}")
            return False
    
    def _gps_loop(self):
        """Boucle de réception des données GPS"""
        while self.running:
            try:
                received_data = self.transport.recv(1024)
                if not received_data:
                    break
                    
                messages = self.decoder.on_data(received_data)
                current_time = time.time()
                
                for header, message in messages:
                    if isinstance(message, PoseMessage):
                        # Mise à jour de la position
                        self.current_lat = message.lla_deg[0]
                        self.current_lon = message.lla_deg[1]
                        self.current_alt = message.lla_deg[2]
                        
                        # Calcul du cap du véhicule
                        yaw_deg = message.ypr_deg[0]
                        self.heading_deg = yaw_to_heading(yaw_deg)
                        
                        # Type de solution GPS
                        self.solution_type = str(message.solution_type)
                        
                        # Calcul vers l'objectif si défini
                        if self.goal_lat and self.goal_lon:
                            self.bearing_to_goal = self.calculate_bearing(
                                self.current_lat, self.current_lon,
                                self.goal_lat, self.goal_lon
                            )
                            self.distance_to_goal = self.calculate_distance(
                                self.current_lat, self.current_lon,
                                self.goal_lat, self.goal_lon
                            )
                        
                        self.last_update = current_time
                        break
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"⚠️ Erreur GPS: {e}")
                time.sleep(1)
    
    def stop(self):
        """Arrête la connexion GPS"""
        self.running = False
        if self.transport:
            try:
                self.transport.close()
            except:
                pass
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def get_navigation_data(self):
        """Retourne les données de navigation actuelles"""
        if not self.current_lat or not self.heading_deg:
            return None
            
        data = {
            'lat': self.current_lat,
            'lon': self.current_lon,
            'alt': self.current_alt,
            'heading': self.heading_deg,
            'solution': self.solution_type
        }
        
        if self.goal_lat and self.goal_lon and self.bearing_to_goal is not None:
            angle_diff = self.smallest_angle_diff(self.heading_deg, self.bearing_to_goal)
            data.update({
                'bearing_to_goal': self.bearing_to_goal,
                'distance_to_goal': self.distance_to_goal,
                'angle_to_goal': angle_diff,
                'goal_reached': self.distance_to_goal < GOAL_REACHED_DISTANCE
            })
        
        return data


class AutoDriveState(State):
    def __init__(self, use_gps=False, gps_host=GPS_HOST, gps_port=GPS_PORT, goal_lat=None, goal_lon=None):
        print("Initializing AutoDrive...")
        self.lidar = LidarParser()
        self.last_steering = 0.0  # Pour le lissage
        self.reverse_timer = 0    # Compteur pour la marche arrière
        
        # GPS Navigation
        self.gps = None
        self.use_gps = use_gps and GPS_AVAILABLE
        if self.use_gps:
            self.gps = GPSNavigator(gps_host, gps_port, goal_lat, goal_lon)
            if not self.gps.start():
                print("⚠️ Mode LIDAR seul (GPS non disponible)")
                self.use_gps = False
        else:
            print("ℹ️ Mode LIDAR seul")
    
    def stop(self):
        self.lidar.stop()
        if self.gps:
            self.gps.stop()

    def run_single(self, motor: Motor, gamepad: Gamepad):
        """Exécute un cycle de contrôle"""
        # Récupérer les données GPS si disponibles
        gps_data = None
        if self.use_gps and self.gps:
            gps_data = self.gps.get_navigation_data()
            if gps_data:
                if 'distance_to_goal' in gps_data:
                    print(f"📍 GPS: {gps_data['distance_to_goal']:.1f}m vers objectif, angle: {gps_data['angle_to_goal']:+.0f}° (cap:{gps_data['heading']:.0f}°)")
                    if gps_data['goal_reached']:
                        print("🎯 OBJECTIF ATTEINT!")
                        motor.set_speed_objective(0.0)
                        motor.set_steering_objective(0.0)
                        return
        
        points = self.lidar.get_points()
        if not points:
            time.sleep(0.05)
            return
        
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
        if sector_name:
            status = "🟢" if result['min_dist'] > SAFE_DISTANCE else "🟡" if result['min_dist'] > SLOW_DISTANCE else "🔴"
            print(f"  [{sector_name}] ({min_angle:+4.0f}° to {max_angle:+4.0f}°): {status} {result['min_dist']:.2f}m (avg:{result['avg_dist']:.2f}m, pts:{result['count']})")
        
        
        return result

    def normalize_angle(self, angle):
        """Normalise un angle entre -180 et 180"""
        angle = angle % 360
        if angle > 180:
            angle -= 360
        return angle

    def navigate(self, motor: Motor, front, left, right, largescans, gps_data=None):
        """Logique principale de navigation"""
        
        # Obstacle très proche : marche arrière
        if front['min_dist'] < STOP_DISTANCE:
            print(f"⚠️ OBSTACLE CRITIQUE à {front['min_dist']:.2f}m - MARCHE ARRIÈRE")
            self.initiate_reverse(motor, largescans)
            return
        
        best_direction = self.find_best_path(front, left, right, gps_data)
        speed = self.calculate_speed(front['min_dist'])
        target_steering = best_direction['steering']
        
        motor.set_steering_objective(target_steering)
        motor.set_speed_objective(speed)
        


    def find_best_path(self, front, left, right, gps_data=None):
        """Trouve la meilleure direction à prendre avec braquage proportionnel"""
        
        # Calculer le braquage proportionnel basé sur l'espace disponible
        right_steering = self.calculate_proportional_steering(right, 1.0)   # Positif = droite
        left_steering = self.calculate_proportional_steering(left, -1.0)    # Négatif = gauche
        front_steering = self.calculate_proportional_steering(front, 0.0) 

        # Calculer le bonus de continuité basé sur la direction actuelle
        def direction_weight(target_steering):
            """Calcule un bonus pour favoriser la continuité de direction"""
            angle_diff = abs(target_steering - self.last_steering)
            # Plus la différence est petite, plus le bonus est élevé (max +20%)
            return 1.0 + (1.0 - min(angle_diff / STEER_ANGLE, 1.0)) * 0.2
        
        # Calculer le bonus GPS si disponible
        def gps_weight(target_steering):
            """Calcule un bonus basé sur l'alignement avec l'objectif GPS"""
            if not gps_data or 'angle_to_goal' not in gps_data:
                return 1.0
            
            # Convertir l'angle vers l'objectif en steering (-1 à 1)
            angle_to_goal = gps_data['angle_to_goal']
            goal_steering = max(-STEER_ANGLE, min(STEER_ANGLE, angle_to_goal / 90.0))
            
            # Calculer la différence entre le steering proposé et le steering idéal GPS
            steering_diff = abs(target_steering - goal_steering)
            
            # Bonus jusqu'à +30% si aligné avec l'objectif GPS
            return 1.0 + (1.0 - min(steering_diff / STEER_ANGLE, 1.0)) * 0.3
        
        # Calcul des scores pour chaque direction avec pondération
        paths = [
            {
                'name': 'AVANT',
                'steering': front_steering,
                'score': front['avg_dist'] * direction_weight(front_steering) * gps_weight(front_steering),
                'free': front['min_dist'] > SAFE_DISTANCE
            },
            {
                'name': 'DROITE',
                'steering': right_steering,
                'score': right['avg_dist'] * 0.8 * direction_weight(right_steering) * gps_weight(right_steering),
                'free': right['min_dist'] > SLOW_DISTANCE
            },
            {
                'name': 'GAUCHE',
                'steering': left_steering,
                'score': left['avg_dist'] * 0.8 * direction_weight(left_steering) * gps_weight(left_steering),
                'free': left['min_dist'] > SLOW_DISTANCE
            }
        ]

        free_paths = [p for p in paths if p['free']]
        
        if not free_paths:
            best = max(paths, key=lambda p: p['score'])
            print(f"⚠️ Aucun chemin libre! Meilleur choix: {best['name']} (steering={best['steering']:+.2f})")
            return best
        
        best = max(free_paths, key=lambda p: p['score'])
        
        # Ajustement fin SEULEMENT si on va droit ET qu'il y a un obstacle très proche et décentré
        if best['name'] == 'AVANT' and front['min_dist'] < SAFE_DISTANCE * 0.7:
            best['steering'] = self.fine_tune_steering(front['obstacles'])
        
        print(f"✅ Chemin choisi: {best['name']} (steering={best['steering']:+.2f})")
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
            print("🔙 FIN DE MARCHE ARRIÈRE dans 10 cycle restet steering")
            motor.set_steering_objective(0.0);
