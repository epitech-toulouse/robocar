#!/usr/bin/env python3
"""
Serveur GPS - Lit depuis le port 25000 et expose les données brutes sur le port 25001
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


class GPSServer:
    def __init__(self, source_host='localhost', source_port=25000, listen_port=25001):
        self.source_host = source_host
        self.source_port = source_port
        self.listen_port = listen_port
        
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
                
                # Construire le message avec les données brutes GPS
                message = f"=== GPS RAW DATA ===\n"
                message += f"Position (lla_deg):\n"
                message += f"  Latitude:  {data['lat']:.8f}°\n"
                message += f"  Longitude: {data['lon']:.8f}°\n"
                message += f"  Altitude:  {data['alt']:.2f}m\n"
                message += f"\n"
                
                message += f"Attitude (ypr_deg):\n"
                if data['heading'] is not None:
                    message += f"  Yaw (Heading): {data['heading']:.2f}°\n"
                else:
                    message += f"  Yaw (Heading): N/A\n"
                if data['pitch'] is not None:
                    message += f"  Pitch:         {data['pitch']:.2f}°\n"
                else:
                    message += f"  Pitch:         N/A\n"
                if data['roll'] is not None:
                    message += f"  Roll:          {data['roll']:.2f}°\n"
                else:
                    message += f"  Roll:          N/A\n"
                message += f"\n"
                
                if data['position_std_enu_m']:
                    message += f"Position Std Dev (ENU) [m]:\n"
                    message += f"  East:  {data['position_std_enu_m'][0]:.3f}m\n"
                    message += f"  North: {data['position_std_enu_m'][1]:.3f}m\n"
                    message += f"  Up:    {data['position_std_enu_m'][2]:.3f}m\n"
                    message += f"\n"
                
                if data['ypr_std_deg']:
                    message += f"Attitude Std Dev [deg]:\n"
                    message += f"  Yaw:   {data['ypr_std_deg'][0]:.3f}°\n"
                    message += f"  Pitch: {data['ypr_std_deg'][1]:.3f}°\n"
                    message += f"  Roll:  {data['ypr_std_deg'][2]:.3f}°\n"
                    message += f"\n"
                
                if data['velocity_body_mps']:
                    message += f"Velocity (Body Frame) [m/s]:\n"
                    message += f"  X: {data['velocity_body_mps'][0]:.3f}m/s\n"
                    message += f"  Y: {data['velocity_body_mps'][1]:.3f}m/s\n"
                    message += f"  Z: {data['velocity_body_mps'][2]:.3f}m/s\n"
                    message += f"\n"
                
                if data['velocity_std_body_mps']:
                    message += f"Velocity Std Dev [m/s]:\n"
                    message += f"  X: {data['velocity_std_body_mps'][0]:.3f}m/s\n"
                    message += f"  Y: {data['velocity_std_body_mps'][1]:.3f}m/s\n"
                    message += f"  Z: {data['velocity_std_body_mps'][2]:.3f}m/s\n"
                    message += f"\n"
                
                message += f"Solution Type: {data['solution_type']}\n"
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
        print("Waiting for GPS data...\n")
        
        try:
            while True:
                time.sleep(1)
                with self.data_lock:
                    if self.current_data['timestamp'] > 0:
                        age = time.time() - self.current_data['timestamp']
                        if age < 2.0:
                            data = self.current_data
                            
                            print(f"=== GPS RAW DATA ===")
                            print(f"Position: {data['lat']:.8f}°, {data['lon']:.8f}° [{data['alt']:.2f}m]")
                            
                            if data['heading'] is not None:
                                print(f"Yaw (Heading): {data['heading']:.2f}°", end="")
                            else:
                                print(f"Yaw (Heading): N/A", end="")
                            
                            if data['pitch'] is not None:
                                print(f" | Pitch: {data['pitch']:.2f}°", end="")
                            else:
                                print(f" | Pitch: N/A", end="")
                            
                            if data['roll'] is not None:
                                print(f" | Roll: {data['roll']:.2f}°")
                            else:
                                print(f" | Roll: N/A")
                            
                            if data['position_std_enu_m']:
                                print(f"Pos Std: E={data['position_std_enu_m'][0]:.3f}m N={data['position_std_enu_m'][1]:.3f}m U={data['position_std_enu_m'][2]:.3f}m")
                            
                            if data['velocity_body_mps']:
                                print(f"Velocity: X={data['velocity_body_mps'][0]:.2f}m/s Y={data['velocity_body_mps'][1]:.2f}m/s Z={data['velocity_body_mps'][2]:.2f}m/s")
                            
                            print(f"Solution: {data['solution_type']}")
                            print("-" * 60)
        except KeyboardInterrupt:
            print("\n🛑 Arrêt du serveur...")
            self.running = False


if __name__ == "__main__":
    server = GPSServer()
    server.start()
