
import socket
import threading
import time
import re

class LidarParserUDP:
    def __init__(self, host='127.0.0.1', port=8888):
        self.host = host
        self.port = port
        self.running = True
        self.points = []
        self.lock = threading.Lock()
        self.sock = None
        
        print(f"📡 Initializing UDP LIDAR Parser on {host}:{port}")
        
        self.thread = threading.Thread(target=self.__loop__, daemon=True)
        self.thread.start()

    def __loop__(self):
        while self.running:
            try:
                # Create socket
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sock.settimeout(1.0)
                
                # Send registration message
                print(f"Connecting to LIDAR UDP server at {self.host}:{self.port}...")
                self.sock.sendto(b"CONNECT", (self.host, self.port))
                
                fails = 0
                while self.running:
                    try:
                        data, _ = self.sock.recvfrom(4096)
                        if data:
                            self._process_data(data.decode('utf-8', errors='ignore'))
                            fails = 0
                    except socket.timeout:
                        fails += 1
                        if fails > 5:
                            print("⚠️ UDP Timeout - Reconnecting...")
                            # Keep alive / Re-register
                            self.sock.sendto(b"CONNECT", (self.host, self.port))
                            fails = 0
                    except Exception as e:
                        print(f"⚠️ UDP Receive Error: {e}")
                        break
                        
            except Exception as e:
                print(f"❌ UDP Connection Error: {e}")
                time.sleep(2)
            finally:
                if self.sock:
                    self.sock.close()
                    self.sock = None

    def _process_data(self, data_str):
        # Data format comes as accumulated strings: "angle,dist,int\nangle,dist,int\n..."
        lines = data_str.strip().split('\n')
        new_points = []
        
        for line in lines:
            try:
                parts = line.split(',')
                if len(parts) >= 2:
                    angle = float(parts[0])
                    distance = float(parts[1])
                    intensity = int(parts[2]) if len(parts) > 2 else 0
                    
                    # Filter invalid points
                    if distance > 0.05 and distance < 12.0:
                        new_points.append({
                            'angle': angle,
                            'distance': distance,
                            'intensity': intensity
                        })
            except ValueError:
                continue
                
        with self.lock:
            self.points.extend(new_points)
            # Keep last 2000 points (approx 1-2 full scans)
            if len(self.points) > 2000:
                self.points = self.points[-2000:]

    def get_points(self):
        with self.lock:
            # Return copy of points and CLEAR buffer for fresh data
            current_points = list(self.points)
            self.points = []  # Effacer le buffer pour avoir des données fraîches au prochain appel
            return current_points

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()
