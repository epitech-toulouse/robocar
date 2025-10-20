import serial
import struct
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# Serial config
ser = serial.Serial("/dev/ttyUSB0", 230400, timeout=1)

# Matplotlib interactive plot
plt.ion()
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
sc = ax.scatter([], [], s=2, c='red', alpha=0.6)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_ylim(0, 12)  # 12 meter range
ax.set_title("LD19 LiDAR Scan", pad=20)

# Store recent scans for smoother visualization
scan_history = deque(maxlen=3)

def parse_ld19_packet(packet):
    """
    LD19 packet structure (47 bytes):
    - Header: 0x54 (1 byte)
    - VerLen: 0x2C (1 byte) 
    - Speed: (2 bytes)
    - Start Angle: (2 bytes)
    - Data: 12 points × 3 bytes = 36 bytes
    - End Angle: (2 bytes)
    - Timestamp: (2 bytes)
    - CRC: (1 byte)
    """
    if len(packet) < 47:
        return []
    
    # Check header
    if packet[0] != 0x54:
        return []
    
    points = []
    
    # Parse start angle (LSB format, in 0.01 degree units)
    start_angle = struct.unpack('<H', packet[4:6])[0] / 100.0
    
    # Parse end angle
    end_angle = struct.unpack('<H', packet[42:44])[0] / 100.0
    
    # Calculate angle step
    angle_diff = end_angle - start_angle
    if angle_diff < 0:
        angle_diff += 360
    angle_step = angle_diff / 11.0 if angle_diff > 0 else 0
    
    # Parse 12 measurement points
    for i in range(12):
        offset = 6 + i * 3
        distance = struct.unpack('<H', packet[offset:offset+2])[0]
        intensity = packet[offset+2]
        
        # Calculate angle for this point
        angle = start_angle + i * angle_step
        if angle >= 360:
            angle -= 360
        
        # Convert distance to meters and filter invalid readings
        dist_m = distance / 1000.0
        
        # Only add valid points (distance > 0 and < 12m, intensity > 0)
        if 0.05 < dist_m < 12.0 and intensity > 0:
            points.append((angle, dist_m, intensity))
    
    return points

def update_plot(all_points):
    """Update the polar plot with accumulated points"""
    if not all_points:
        return
    
    angles = [math.radians(a) for a, d, i in all_points]
    distances = [d for a, d, i in all_points]
    intensities = [i for a, d, i in all_points]
    
    # Normalize intensities for color mapping
    intensities_norm = np.array(intensities) / 255.0
    
    sc.set_offsets(np.c_[angles, distances])
    sc.set_array(intensities_norm)
    sc.set_cmap('hot')
    
    plt.draw()
    plt.pause(0.001)

buffer = bytearray()
frame_points = []

print("Starting LD19 LiDAR visualization...")
print("Press Ctrl+C to stop")

try:
    while True:
        data = ser.read(512)
        if data:
            buffer.extend(data)
            
            # Look for packet header 0x54
            while len(buffer) >= 47:
                # Find header
                header_idx = buffer.find(b'\x54')
                
                if header_idx == -1:
                    buffer.clear()
                    break
                
                # Remove data before header
                if header_idx > 0:
                    del buffer[:header_idx]
                
                # Check if we have a complete packet
                if len(buffer) < 47:
                    break
                
                # Extract and parse packet
                packet = buffer[:47]
                points = parse_ld19_packet(packet)
                
                if points:
                    frame_points.extend(points)
                
                # Remove processed packet
                del buffer[:47]
                
                # Update plot every 10 packets (~1/36th rotation)
                if len(frame_points) > 100:
                    scan_history.append(frame_points.copy())
                    
                    # Combine recent scans for smoother visualization
                    all_points = []
                    for scan in scan_history:
                        all_points.extend(scan)
                    
                    update_plot(all_points)
                    frame_points.clear()

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    ser.close()
    plt.close()
    print("Stopped")