import time
import sys
import math
import random

# Simulate a LIDAR scanner
# Outputs: angle,distance,intensity
# Angle: 0-359
# Distance: 0-10 meters
# Intensity: 0-255

angle = 0
while True:
    rad = math.radians(angle)
    
    # Simple simulation: A box room 6m x 6m
    # Center is 0,0. Room is -3 to 3.
    
    dist = 10.0 # Max range
    
    # Check intersection with walls at x=3, x=-3, y=3, y=-3
    if math.cos(rad) != 0:
        d = 3.0 / math.cos(rad)
        if d > 0: dist = min(dist, d)
        d = -3.0 / math.cos(rad)
        if d > 0: dist = min(dist, d)
        
    if math.sin(rad) != 0:
        d = 3.0 / math.sin(rad)
        if d > 0: dist = min(dist, d)
        d = -3.0 / math.sin(rad)
        if d > 0: dist = min(dist, d)
    
    # Add some noise
    dist += random.uniform(-0.05, 0.05)
    
    intensity = random.randint(100, 255)
    
    print(f"{angle:.2f},{dist:.3f},{intensity}")
    sys.stdout.flush()
    
    angle = (angle + 1.5) % 360
    time.sleep(0.005) 
