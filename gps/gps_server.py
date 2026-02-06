#!/usr/bin/env python3
"""
Serveur GPS - Lit depuis le port 25000 et expose les données en JSON sur le port 25001
"""

import os
import socket
import sys
import time
import json
import threading
import math

# Add the Python root directory for FusionEngine imports
root_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from fusion_engine_client.messages.core import PoseMessage
from fusion_engine_client.parsers import FusionEngineDecoder


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate the bearing from point 1 to point 2 in degrees (0-360)."""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lon_diff_rad = math.radians(lon2 - lon1)
    
    x = math.sin(lon_diff_rad) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(lon_diff_rad)
    
    bearing_rad = math.atan2(x, y)
    bearing_deg = math.degrees(bearing_rad)
    
    return (bearing_deg + 360) % 360


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points in meters using Haversine formula."""
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def smallest_angle_diff_deg(from_deg, to_deg):
    """Return signed smallest angle difference from from_deg to to_deg in degrees (-180..180]."""
    return (to_deg - from_deg + 180.0) % 360.0 - 180.0


class GPSServer:
    def __init__(self, source_host='localhost', source_port=25000, listen_port=25001, goal_lat=None, goal_lon=None):
        self.source_host = source_host
        self.source_port = source_port
        self.listen_port = listen_port
        self.goal_lat = goal_lat
        self.goal_lon = goal_lon
        
        # Données GPS actuelles
        self.current_data = {
            'lat': None,
            'lon': None,
            'alt': None,
            'heading': None,  # yaw en degrés (0-360)
            'pitch': None,
            'roll': None,
            'solution_type': None,
            'position_std_enu_m': None,  # [east, north, up]
            'ypr_std_deg': None,  # [yaw, pitch, roll]
            'velocity_body_mps': None,  # [x, y, z]
            'velocity_std_body_mps': None,  # [x, y, z]
            'timestamp': 0
        }
        self.data_lock = threading.Lock()
        self.running = False
        
    def read_gps_stream(self):
        """Thread qui lit les données GPS depuis le port source"""
        print(f"📡 Connexion à la source GPS {self.source_host}:{self.source_port}...")
        
        while self.running:
            try:
                transport = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                transport.connect((socket.gethostbyname(self.source_host), self.source_port))
                print(f"✅ Connecté à la source GPS")
                
                decoder = FusionEngineDecoder()
                
                while self.running:
                    received_data = transport.recv(4096)
                    if not received_data:
                        break
                    
                    messages = decoder.on_data(received_data)
                    
                    for header, message in messages:
                        if isinstance(message, PoseMessage):
                            # lla_deg -> latitude, longitude, altitude en degré
                            lat = message.lla_deg[0]
                            lon = message.lla_deg[1]
                            alt = message.lla_deg[2]
                            
                            # ypr_deg -> yaw pitch roll en degrés
                            yaw_deg = message.ypr_deg[0]
                            pitch_deg = message.ypr_deg[1]
                            roll_deg = message.ypr_deg[2]
                            
                            # Le yaw est déjà en degrés (0-360), pas besoin de conversion
                            if not math.isnan(yaw_deg):
                                heading = yaw_deg
                            else:
                                heading = None
                            
                            pitch = pitch_deg if not math.isnan(pitch_deg) else None
                            roll = roll_deg if not math.isnan(roll_deg) else None
                            
                            # Solution type (DGPS, RTK, etc.)
                            solution_type = str(message.solution_type)
                            
                            # Standard deviations
                            position_std = list(message.position_std_enu_m) if hasattr(message, 'position_std_enu_m') else None
                            ypr_std = list(message.ypr_std_deg) if hasattr(message, 'ypr_std_deg') else None
                            
                            # Velocity
                            velocity = list(message.velocity_body_mps) if hasattr(message, 'velocity_body_mps') else None
                            velocity_std = list(message.velocity_std_body_mps) if hasattr(message, 'velocity_std_body_mps') else None
                            
                            # Mettre à jour les données
                            with self.data_lock:
                                self.current_data = {
                                    'lat': lat,
                                    'lon': lon,
                                    'alt': alt,
                                    'heading': heading,
                                    'pitch': pitch,
                                    'roll': roll,
                                    'solution_type': solution_type,
                                    'position_std_enu_m': position_std,
                                    'ypr_std_deg': ypr_std,
                                    'velocity_body_mps': velocity,
                                    'velocity_std_body_mps': velocity_std,
                                    'timestamp': time.time()
                                }
                            break
                
                transport.close()
            except Exception as e:
                print(f"⚠️ Erreur connexion GPS: {e}")
                time.sleep(2)
    
    def handle_client(self, client_socket):
        """Gère un client connecté"""
        try:
            while self.running:
                time.sleep(1)
                with self.data_lock:
                    data = self.current_data.copy()
                
                # Vérifier si les données sont fraîches
                age = time.time() - data['timestamp'] if data['timestamp'] > 0 else None
                if age is None or age >= 2.0:
                    continue
                
                lat = data['lat']
                lon = data['lon']
                alt = data['alt']
                heading = data['heading']
                solution_type = data.get('solution_type', 'N/A')
                
                # Construire le message au format gps_goal.py
                message = f"Current: {lat:.6f}°, {lon:.6f}° [{alt:.2f}m]\n"
                
                # Afficher les infos vers l'objectif
                if self.goal_lat and self.goal_lon:
                    bearing = calculate_bearing(lat, lon, self.goal_lat, self.goal_lon)
                    distance = calculate_distance(lat, lon, self.goal_lat, self.goal_lon)
                    
                    # Direction cardinale
                    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                                'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
                    direction_idx = round(bearing / 22.5) % 16
                    direction = directions[direction_idx]
                    
                    message += f"Distance to goal: {distance:.2f}m\n"
                    message += f"Bearing to goal: {bearing:.1f}° ({direction})\n"
                    
                    if heading is not None:
                        heading_direction_idx = round(heading / 22.5) % 16
                        heading_direction = directions[heading_direction_idx]
                        message += f"Vehicle heading: {heading:.1f}° ({heading_direction})\n"
                        
                        # Instruction de virage
                        angle_diff = smallest_angle_diff_deg(heading, bearing)
                        abs_diff = abs(angle_diff)
                        if abs_diff < 5.0:
                            turn = "On course"
                        else:
                            turn_dir = "Right" if angle_diff > 0 else "Left"
                            turn = f"Turn {turn_dir} {abs_diff:.1f}°"
                        message += f"Turn: {turn}\n"
                    else:
                        message += f"Vehicle heading: N/A (stationary or no IMU data)\n"
                        message += f"Turn: Heading not available\n"
                else:
                    if heading is not None:
                        message += f"Vehicle heading: {heading:.1f}°\n"
                    else:
                        message += f"Vehicle heading: N/A (stationary or no IMU data)\n"
                
                message += f"Solution: {solution_type}\n"
                message += "-" * 60 + "\n"
                
                # Envoyer le message
                client_socket.sendall(message.encode('utf-8'))
                
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            client_socket.close()
    
    def serve_clients(self):
        """Thread qui sert les clients sur le port d'écoute"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', self.listen_port))
        server_socket.listen(5)
        server_socket.settimeout(1.0)
        
        print(f"🌐 Serveur GPS en écoute sur le port {self.listen_port}")
        
        while self.running:
            try:
                client_socket, address = server_socket.accept()
                print(f"📱 Client connecté: {address}")
                client_thread = threading.Thread(target=self.handle_client, args=(client_socket,), daemon=True)
                client_thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"⚠️ Erreur serveur: {e}")
        
        server_socket.close()
    
    def start(self):
        """Démarre le serveur GPS"""
        self.running = True
        
        # Thread pour lire le stream GPS
        gps_thread = threading.Thread(target=self.read_gps_stream, daemon=True)
        gps_thread.start()
        
        # Thread pour servir les clients
        server_thread = threading.Thread(target=self.serve_clients, daemon=True)
        server_thread.start()
        
        print("✅ Serveur GPS démarré")
        if self.goal_lat and self.goal_lon:
            print(f"Goal position: {self.goal_lat:.6f}°, {self.goal_lon:.6f}°")
        print("Waiting for GPS data...\n")
        
        try:
            while True:
                time.sleep(1)
                with self.data_lock:
                    if self.current_data['timestamp'] > 0:
                        age = time.time() - self.current_data['timestamp']
                        if age < 2.0:
                            lat = self.current_data['lat']
                            lon = self.current_data['lon']
                            alt = self.current_data['alt']
                            heading = self.current_data['heading']
                            solution_type = self.current_data.get('solution_type', 'N/A')
                            
                            # Format identique à gps_goal.py
                            print(f"Current: {lat:.6f}°, {lon:.6f}° [{alt:.2f}m]")
                            
                            # Afficher les infos vers l'objectif
                            if self.goal_lat and self.goal_lon:
                                bearing = calculate_bearing(lat, lon, self.goal_lat, self.goal_lon)
                                distance = calculate_distance(lat, lon, self.goal_lat, self.goal_lon)
                                
                                # Direction cardinale
                                directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                                            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
                                direction_idx = round(bearing / 22.5) % 16
                                direction = directions[direction_idx]
                                
                                print(f"Distance to goal: {distance:.2f}m")
                                print(f"Bearing to goal: {bearing:.1f}° ({direction})")
                                
                                if heading is not None:
                                    heading_direction_idx = round(heading / 22.5) % 16
                                    heading_direction = directions[heading_direction_idx]
                                    print(f"Vehicle heading: {heading:.1f}° ({heading_direction})")
                                    
                                    # Instruction de virage
                                    angle_diff = smallest_angle_diff_deg(heading, bearing)
                                    abs_diff = abs(angle_diff)
                                    if abs_diff < 5.0:
                                        turn = "On course"
                                    else:
                                        turn_dir = "Right" if angle_diff > 0 else "Left"
                                        turn = f"Turn {turn_dir} {abs_diff:.1f}°"
                                    print(f"Turn: {turn}")
                                else:
                                    print(f"Vehicle heading: N/A (stationary or no IMU data)")
                                    print(f"Turn: Heading not available")
                            else:
                                if heading is not None:
                                    print(f"Vehicle heading: {heading:.1f}°")
                                else:
                                    print(f"Vehicle heading: N/A (stationary or no IMU data)")
                            
                            print(f"Solution: {solution_type}")
                            print("-" * 60)
        except KeyboardInterrupt:
            print("\n🛑 Arrêt du serveur...")
            self.running = False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Serveur GPS avec objectif optionnel')
    parser.add_argument('--goal-lat', type=float, default=None,
                        help='Latitude de l\'objectif en degrés')
    parser.add_argument('--goal-lon', type=float, default=None,
                        help='Longitude de l\'objectif en degrés')
    
    args = parser.parse_args()
    
    server = GPSServer(goal_lat=args.goal_lat, goal_lon=args.goal_lon)
    server.start()
