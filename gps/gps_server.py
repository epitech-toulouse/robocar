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
from fusion_engine_client.messages.defs import yaw_to_heading
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
            'heading': None,
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
                            lat = message.lla_deg[0]
                            lon = message.lla_deg[1]
                            alt = message.lla_deg[2]
                            
                            # Calculer le heading
                            yaw_deg = message.ypr_deg[0]
                            if not math.isnan(yaw_deg):
                                heading = yaw_to_heading(yaw_deg)
                            else:
                                heading = None
                            
                            # Mettre à jour les données
                            with self.data_lock:
                                self.current_data = {
                                    'lat': lat,
                                    'lon': lon,
                                    'alt': alt,
                                    'heading': heading,
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
                with self.data_lock:
                    data = self.current_data.copy()
                
                # Ajouter un flag pour indiquer si les données sont fraîches
                data['age'] = time.time() - data['timestamp'] if data['timestamp'] > 0 else None
                data['valid'] = data['age'] is not None and data['age'] < 2.0
                
                # Calculer les données vers l'objectif si défini
                if self.goal_lat is not None and self.goal_lon is not None and data['lat'] is not None:
                    data['goal_bearing'] = calculate_bearing(data['lat'], data['lon'], self.goal_lat, self.goal_lon)
                    data['goal_distance'] = calculate_distance(data['lat'], data['lon'], self.goal_lat, self.goal_lon)
                    
                    # Calculer l'angle de virage si on a un heading
                    if data['heading'] is not None:
                        angle_diff = smallest_angle_diff_deg(data['heading'], data['goal_bearing'])
                        data['turn_angle'] = angle_diff
                        
                        if abs(angle_diff) < 5.0:
                            data['turn_direction'] = "straight"
                        elif angle_diff > 0:
                            data['turn_direction'] = "right"
                        else:
                            data['turn_direction'] = "left"
                    else:
                        data['turn_angle'] = None
                        data['turn_direction'] = None
                else:
                    data['goal_bearing'] = None
                    data['goal_distance'] = None
                    data['turn_angle'] = None
                    data['turn_direction'] = None
                
                # Envoyer en JSON avec newline
                json_data = json.dumps(data) + '\n'
                client_socket.sendall(json_data.encode('utf-8'))
                
                time.sleep(0.1)  # 10 Hz
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
            print(f"🎯 Objectif: {self.goal_lat:.6f}°, {self.goal_lon:.6f}°")
        
        try:
            while True:
                time.sleep(1)
                with self.data_lock:
                    if self.current_data['timestamp'] > 0:
                        age = time.time() - self.current_data['timestamp']
                        if age < 2.0:
                            heading_str = f"{self.current_data['heading']:.1f}°" if self.current_data['heading'] is not None else "N/A"
                            print(f"📍 GPS: {self.current_data['lat']:.6f}°, {self.current_data['lon']:.6f}° [Heading: {heading_str}]")
                            
                            # Afficher les infos vers l'objectif
                            if self.goal_lat and self.goal_lon:
                                bearing = calculate_bearing(self.current_data['lat'], self.current_data['lon'], 
                                                          self.goal_lat, self.goal_lon)
                                distance = calculate_distance(self.current_data['lat'], self.current_data['lon'],
                                                            self.goal_lat, self.goal_lon)
                                print(f"🎯 Distance: {distance:.2f}m, Bearing: {bearing:.1f}°")
        except KeyboardInterrupt:
            print("\n🛑 Arrêt du serveur...")
            self.running = False
                time.sleep(1)
                with self.data_lock:
                    if self.current_data['timestamp'] > 0:
    import argparse
    
    parser = argparse.ArgumentParser(description='Serveur GPS avec objectif optionnel')
    parser.add_argument('--goal-lat', type=float, default=None,
                        help='Latitude de l\'objectif en degrés')
    parser.add_argument('--goal-lon', type=float, default=None,
                        help='Longitude de l\'objectif en degrés')
    
    args = parser.parse_args()
    
    server = GPSServer(goal_lat=args.goal_lat, goal_lon=args.goal_lon age = time.time() - self.current_data['timestamp']
                        if age < 2.0:
                            heading_str = f"{self.current_data['heading']:.1f}°" if self.current_data['heading'] is not None else "N/A"
                            print(f"📍 GPS: {self.current_data['lat']:.6f}°, {self.current_data['lon']:.6f}° [Heading: {heading_str}]")
        except KeyboardInterrupt:
            print("\n🛑 Arrêt du serveur...")
            self.running = False


if __name__ == "__main__":
    server = GPSServer()
    server.start()
