#!/usr/bin/env python3
"""
Lecteur GPS simple - Lit les données JSON depuis le serveur GPS (port 25001)
"""

import socket
import json
import threading
import time


class SimpleGPSReader:
    def __init__(self, hostname='localhost', port=25001):
        self.hostname = hostname
        self.port = port
        
        # Données GPS actuelles
        self.current_lat = None
        self.current_lon = None
        self.current_alt = None
        self.heading_deg = None
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
        """Boucle de réception des données GPS - lit le JSON"""
        buffer = ""
        
        while self.running:
            try:
                received_data = self.transport.recv(4096)
                if not received_data:
                    break
                
                buffer += received_data.decode('utf-8')
                
                # Traiter les lignes complètes
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    try:
                        data = json.loads(line)
                        
                        if data.get('valid') and data['lat'] is not None:
                            self.current_lat = data['lat']
                            self.current_lon = data['lon']
                            self.current_alt = data['alt']
                            self.heading_deg = data['heading']
                            self.last_update = time.time()
                    except json.JSONDecodeError:
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
        """Retourne la position actuelle"""
        if not self.current_lat or time.time() - self.last_update > 5.0:
            return None
        
        return {
            'lat': self.current_lat,
            'lon': self.current_lon,
            'alt': self.current_alt,
            'heading': self.heading_deg,
            'age': time.time() - self.last_update
        }


if __name__ == "__main__":
    import sys
    
    # Test du lecteur
    host = sys.argv[1] if len(sys.argv) > 1 else 'localhost'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 25001
    
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
