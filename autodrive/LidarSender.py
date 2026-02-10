import socket
import threading
import time

class LidarUdpServer:
    def __init__(self, port=8888):
        self.port = port
        self.running = True
        self.clients = set() # Set of (ip, port) tuples
        self.lock = threading.Lock()
        
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(('0.0.0.0', port))
            self.sock.settimeout(0.5)
            print(f"📡 UDP Server initialized on 0.0.0.0:{port}")
            
            self.thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.thread.start()
        except Exception as e:
            print(f"❌ Failed to start UDP Server: {e}")
            self.sock = None

    def _listen_loop(self):
        print(f"👂 Listening for UDP connections on port {self.port}...")
        while self.running and self.sock:
            try:
                data, addr = self.sock.recvfrom(1024)
                if data:
                    msg = data.decode('utf-8', errors='ignore').strip()
                    # Si c'est un message de connexion (ou n'importe quoi d'un nouveau client)
                    with self.lock:
                        if addr not in self.clients:
                            self.clients.add(addr)
                            print(f"➕ New client connected: {addr[0]}:{addr[1]}")
                        
                    if msg == "CONNECT":
                        # Optionnel: Répondre OK? Pas nécessaire pour le protocole actuel
                        pass
            except socket.timeout:
                continue
            except Exception as e:
                print(f"⚠️ UDP Listen Error: {e}")

    def send_points(self, points):
        """Broadcast LIDAR points to all connected clients"""
        if not self.sock or not self.clients:
            return

        # Format: angle,distance,intensity
        buffer = ""
        for p in points:
            angle = p.get('angle', 0)
            distance = p.get('distance', 0)
            intensity = p.get('intensity', 0)
            
            line = f"{angle:.2f},{distance:.3f},{intensity}\n"
            
            if len(buffer) + len(line) > 1024:
                self._broadcast(buffer)
                buffer = line
            else:
                buffer += line
        
        if buffer:
            self._broadcast(buffer)

    def _broadcast(self, message):
        encoded = message.encode('utf-8')
        with self.lock:
            # Create a copy to iterate safely if set changes
            targets = list(self.clients)
            
        for addr in targets:
            try:
                self.sock.sendto(encoded, addr)
            except Exception as e:
                print(f"❌ UDP Send Error to {addr}: {e}")
                # Optionnel: supprimer le client s'il est inaccessible? 
                # UDP ne throw pas souvent d'erreur d'envoi sauf si interface down.

    def close(self):
        self.running = False
        if self.sock:
            self.sock.close()

