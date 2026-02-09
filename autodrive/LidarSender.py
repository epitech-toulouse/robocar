import socket
import json

class LidarSender:
    def __init__(self, host='127.0.0.1', port=8888):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"📡 LidarSender initialized on {host}:{port}")

    def send_points(self, points):
        """Envoie les points LIDAR au format attendu par le serveur (ligne par ligne)"""
        # Format attendu: angle,distance,intensity
        # On peut envoyer point par point ou grouper un peu
        # UDP a une limite de taille (~1400 bytes safe), donc on envoie par paquets
        
        buffer = ""
        for p in points:
            # p est un dict {'angle': float, 'distance': float, 'intensity': int} (si dispo) ou juste distance
            # LidarParserCpp / LidarParser sort des dicts.
            
            angle = p.get('angle', 0)
            distance = p.get('distance', 0)
            intensity = p.get('intensity', 0) # Assumer 0 si pas là
            
            line = f"{angle:.2f},{distance:.3f},{intensity}\n"
            
            if len(buffer) + len(line) > 1024:
                self._send_raw(buffer)
                buffer = line
            else:
                buffer += line
        
        if buffer:
            self._send_raw(buffer)

    def _send_raw(self, message):
        try:
            self.sock.sendto(message.encode('utf-8'), (self.host, self.port))
        except Exception as e:
            print(f"❌ UDP Send Error: {e}")

    def close(self):
        self.sock.close()
