"""LD19 LiDAR Parser for Robocar"""

import serial
import threading
import struct
import time

class LidarParser:
    def __init__(self, port='/dev/ttyUSB0', baudrate=230400) -> None:
        self.ser = serial.Serial(port, baudrate, timeout=0.1)
        self.running = True
        self.points = []
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.__loop__, daemon=True)
        self.thread.start()

    def __loop__(self) -> None:
        buffer = b''
        while self.running:
            try:
                data = self.ser.read(1024)
                if not data:
                    continue
                buffer += data
                
                while len(buffer) >= 47:
                    # Find header
                    header_idx = buffer.find(b'\x54')
                    if header_idx == -1:
                        buffer = b''
                        break
                    
                    if header_idx > 0:
                        buffer = buffer[header_idx:]
                    
                    if len(buffer) < 47:
                        break
                    
                    packet = buffer[:47]
                    self._parse_packet(packet)
                    buffer = buffer[47:]
            except Exception as e:
                print(f"Lidar error: {e}")
                time.sleep(0.1)

    def _parse_packet(self, packet):
        # packet[0] is 0x54
        # packet[1] is VerLen
        # packet[2:4] is Speed
        
        # Start Angle (little endian, 0.01 deg units)
        start_angle = struct.unpack('<H', packet[4:6])[0] / 100.0
        
        # End Angle
        end_angle = struct.unpack('<H', packet[42:44])[0] / 100.0
        
        angle_diff = end_angle - start_angle
        if angle_diff < 0:
            angle_diff += 360.0
        
        angle_step = angle_diff / 11.0
        
        new_points = []
        for i in range(12):
            offset = 6 + i * 3
            distance = struct.unpack('<H', packet[offset:offset+2])[0] / 1000.0 # meters
            intensity = packet[offset+2]
            
            angle = start_angle + i * angle_step
            if angle >= 360.0:
                angle -= 360.0
            
            # Filter valid points
            if 0.05 < distance < 12.0 and intensity > 0:
                new_points.append({'angle': angle, 'distance': distance})
        
        with self.lock:
            # We only keep the latest points to keep it responsive
            # In a real scenario, we might want a full 360 scan, but for obstacle avoidance
            # we can just buffer them or use them as they come.
            # For simplicity, we'll maintain a list of points from the last second
            self.points.extend(new_points)
            if len(self.points) > 2000:
                self.points = self.points[-2000:]

    def get_points(self):
        with self.lock:
            return list(self.points)

    def stop(self):
        self.running = False
        self.ser.close()
