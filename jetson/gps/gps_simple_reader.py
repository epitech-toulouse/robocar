#!/usr/bin/env python3
"""
Lecteur GPS simple - Lit les données JSON depuis le serveur GPS (port 25001)
"""

import socket
import json
import threading
import time


class SimpleGPSReader:
    def __init__(self, hostname='localhost', port=25000):
        self.hostname = hostname
        self.port = port
        
        # Données GPS actuelles
        self.current_lat = None
        self.current_lon = None
        self.current_alt = None
        self.heading_deg = None
        self.goal_bearing = None
        self.goal_distance = None
        self.turn_angle = None
        self.solution_type = None
        self.last_update = 0
        
        self.transport = None
        self.running = False
        self.thread = None
    
    def start(self):
        """Démarre la connexion GPS en thread séparé"""
        try:
            self.transport = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.transport.settimeout(5.0)
            self.transport.connect((socket.gethostbyname(self.hostname), self.port))
            self.running = True
            
            self.thread = threading.Thread(target=self._gps_loop, daemon=True)
            self.thread.start()
            
            print(f"✅ GPS connecté à {self.hostname}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ Erreur connexion GPS: {e}")
            return False
    
    def _gps_loop(self):
        """Boucle de réception des données GPS - lit le format texte gps_goal.py"""
        buffer = ""
        current_block = {}
        
        while self.running:
            try:
                received_data = self.transport.recv(4096)
                if not received_data:
                    break
                
                buffer += received_data.decode('utf-8')
                
                # Traiter les lignes complètes
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    
                    # Fin d'un bloc de données
                    if line.startswith('---'):
                        if current_block:
                            # Mettre à jour les données si on a au moins la position
                            if 'lat' in current_block:
                                self.current_lat = current_block.get('lat')
                                self.current_lon = current_block.get('lon')
                                self.current_alt = current_block.get('alt')
                                self.heading_deg = current_block.get('heading')
                                self.goal_bearing = current_block.get('goal_bearing')
                                self.goal_distance = current_block.get('goal_distance')
                                self.turn_angle = current_block.get('turn_angle')
                                self.solution_type = current_block.get('solution')
                                self.last_update = time.time()
                            current_block = {}
                        continue
                    
                    # Parser les différentes lignes
                    if line.startswith('Current:'):
                        # Format: "Current: 43.612290°, 1.428899° [190.48m]"
                        try:
                            parts = line.split()
                            lat_str = parts[1].rstrip('°,')
                            lon_str = parts[2].rstrip('°')
                            alt_str = parts[3].strip('[]m')
                            current_block['lat'] = float(lat_str)
                            current_block['lon'] = float(lon_str)
                            current_block['alt'] = float(alt_str)
                        except:
                            pass
                    
                    elif line.startswith('Distance to goal:'):
                        # Format: "Distance to goal: 68.90m"
                        try:
                            distance_str = line.split(':')[1].strip().rstrip('m')
                            current_block['goal_distance'] = float(distance_str)
                        except:
                            pass
                    
                    elif line.startswith('Bearing to goal:'):
                        # Format: "Bearing to goal: 96.3° (E)"
                        try:
                            bearing_str = line.split(':')[1].strip().split('°')[0]
                            current_block['goal_bearing'] = float(bearing_str)
                        except:
                            pass
                    
                    elif line.startswith('Vehicle heading:'):
                        # Format: "Vehicle heading: 262.0° (W)" ou "Vehicle heading: N/A"
                        try:
                            heading_part = line.split(':')[1].strip()
                            if heading_part.startswith('N/A'):
                                current_block['heading'] = None
                            else:
                                heading_str = heading_part.split('°')[0]
                                current_block['heading'] = float(heading_str)
                        except:
                            pass
                    
                    elif line.startswith('Turn:'):
                        # Format: "Turn: Turn Left 165.8°" ou "Turn: On course" ou "Turn: Heading not available"
                        try:
                            turn_part = line.split(':', 1)[1].strip()
                            if 'Heading not available' in turn_part or 'not available' in turn_part:
                                current_block['turn_angle'] = None
                            elif 'Left' in turn_part:
                                angle_str = turn_part.split()[2].rstrip('°')
                                current_block['turn_angle'] = -float(angle_str)  # Négatif pour gauche
                            elif 'Right' in turn_part:
                                angle_str = turn_part.split()[2].rstrip('°')
                                current_block['turn_angle'] = float(angle_str)  # Positif pour droite
                            elif 'On course' in turn_part:
                                current_block['turn_angle'] = 0.0
                        except:
                            pass
                    
                    elif line.startswith('Solution:'):
                        # Format: "Solution: DGPS"
                        try:
                            solution = line.split(':')[1].strip()
                            current_block['solution'] = solution
                        except:
                            pass
                        
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
    
    def get_position(self):
        """Retourne la position actuelle avec données de navigation"""
        if not self.current_lat or time.time() - self.last_update > 5.0:
            return None
        
        return {
            'lat': self.current_lat,
            'lon': self.current_lon,
            'alt': self.current_alt,
            'heading': self.heading_deg,
            'goal_bearing': self.goal_bearing,
            'goal_distance': self.goal_distance,
            'turn_angle': self.turn_angle,
            'solution': self.solution_type,
            'age': time.time() - self.last_update
        }


if __name__ == "__main__":
    import sys
    
    # Test du lecteur
    host = sys.argv[1] if len(sys.argv) > 1 else 'localhost'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 25000
    
    print(f"Connexion à {host}:{port}...")
    reader = SimpleGPSReader(host, port)
    
    if not reader.start():
        sys.exit(1)
    
    print("En attente de données GPS... (Ctrl+C pour arrêter)")
    
    try:
        last_print = 0
        while True:
            pos = reader.get_position()
            current_time = time.time()
            
            if pos and current_time - last_print >= 1.0:
                print(f"Position: {pos['lat']:.6f}°, {pos['lon']:.6f}° [{pos['alt']:.2f}m]")
                if pos['heading'] is not None:
                    print(f"Cap: {pos['heading']:.1f}°")
                else:
                    print(f"Cap: N/A")
                print(f"Age: {pos['age']:.1f}s")
                print("-" * 60)
                last_print = current_time
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nArrêt...")
        reader.stop()
