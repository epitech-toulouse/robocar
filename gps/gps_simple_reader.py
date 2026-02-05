#!/usr/bin/env python3
"""
Lecteur GPS simple - Lit les données du port 25000 et extrait position/cap
Sans dépendance FusionEngine - parse les données brutes
"""

import socket
import struct
import threading
import time
import json


class SimpleGPSReader:
    def __init__(self, hostname='localhost', port=25000):
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
        """Boucle de réception des données GPS - parse format binaire FusionEngine"""
        buffer = bytearray()
        
        while self.running:
            try:
                received_data = self.transport.recv(4096)
                if not received_data:
                    break
                
                buffer.extend(received_data)
                
                # Chercher le sync pattern FusionEngine (0x2E31)
                while len(buffer) >= 12:  # Taille minimale d'un header
                    # Chercher le début d'un message
                    if len(buffer) >= 2 and buffer[0] == 0x2E and buffer[1] == 0x31:
                        # Header trouvé
                        if len(buffer) < 12:
                            break
                        
                        # Lire la longueur du payload
                        payload_len = struct.unpack('<I', buffer[8:12])[0]
                        message_len = 12 + payload_len + 4  # header + payload + CRC
                        
                        if len(buffer) < message_len:
                            break
                        
                        # Extraire le message complet
                        message_type = struct.unpack('<H', buffer[2:4])[0]
                        
                        # Type 10000 = PoseMessage
                        if message_type == 10000 and payload_len >= 136:
                            try:
                                # Offset dans le payload pour les données
                                payload = buffer[12:12+payload_len]
                                
                                # Position LLA (3 doubles à offset 8)
                                self.current_lat = struct.unpack('<d', payload[8:16])[0]
                                self.current_lon = struct.unpack('<d', payload[16:24])[0]
                                self.current_alt = struct.unpack('<d', payload[24:32])[0]
                                
                                # YPR angles (3 floats à offset 56)
                                yaw_rad = struct.unpack('<f', payload[56:60])[0]
                                
                                # Convertir yaw en heading (0-360)
                                self.heading_deg = (90.0 - (yaw_rad * 180.0 / 3.14159265359)) % 360.0
                                
                                self.last_update = time.time()
                            except Exception as e:
                                pass
                        
                        # Supprimer le message traité
                        buffer = buffer[message_len:]
                    else:
                        # Pas de sync, chercher le prochain
                        buffer.pop(0)
                        
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
        if self.current_lat is None:
            return None
        
        return {
            'lat': self.current_lat,
            'lon': self.current_lon,
            'alt': self.current_alt,
            'heading': self.heading_deg,
            'timestamp': self.last_update
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
                print(f"Position: {pos['lat']:.6f}°, {pos['lon']:.6f}° | "
                      f"Alt: {pos['alt']:.2f}m | Cap: {pos['heading']:.1f}°")
                last_print = current_time
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Arrêt...")
    finally:
        reader.stop()
