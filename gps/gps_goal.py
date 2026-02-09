#!/usr/bin/env python3

import os
import socket
import sys
import time
import math

# Add the Python root directory (fusion-engine-client/python/) to the import search path to enable FusionEngine imports
# if this application is being run directly out of the repository and is not installed as a pip package.
root_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from fusion_engine_client.utils.argument_parser import ArgumentParser
from fusion_engine_client.messages import MessagePayload
from fusion_engine_client.messages.core import PoseMessage
from fusion_engine_client.messages.defs import yaw_to_heading
from fusion_engine_client.parsers import FusionEngineDecoder


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate the bearing from point 1 to point 2 in degrees (0-360)."""
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lon_diff_rad = math.radians(lon2 - lon1)
    
    # Calculate bearing
    x = math.sin(lon_diff_rad) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(lon_diff_rad)
    
    bearing_rad = math.atan2(x, y)
    bearing_deg = math.degrees(bearing_rad)
    
    # Normalize to 0-360
    bearing_deg = (bearing_deg + 360) % 360
    
    return bearing_deg


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points in meters using Haversine formula."""
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    distance = R * c
    return distance


def bearing_to_direction(bearing):
    """Convert bearing in degrees to cardinal direction."""
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    index = round(bearing / 22.5) % 16
    return directions[index]


def smallest_angle_diff_deg(from_deg, to_deg):
    """Return signed smallest angle difference from from_deg to to_deg in degrees (-180..180]."""
    return (to_deg - from_deg + 180.0) % 360.0 - 180.0


def turn_instruction(current_heading_deg, target_bearing_deg):
    """Return a simple turn instruction based on heading vs target bearing."""
    diff = smallest_angle_diff_deg(current_heading_deg, target_bearing_deg)
    abs_diff = abs(diff)
    if abs_diff < 5.0:
        return "On course"
    direction = "Right" if diff > 0 else "Left"
    return f"Turn {direction} {abs_diff:.1f}°"


if __name__ == "__main__":
    # Parse command-line arguments.
    parser = ArgumentParser(description="""\
Connect to GPS stream and display direction to a goal position.
""")
    parser.add_argument('-p', '--port', type=int, default=30201,
                        help="The FusionEngine TCP port on the data source.")
    parser.add_argument('hostname',
                        help="The IP address or hostname of the data source.")
    parser.add_argument('--goal-lat', type=float, required=True,
                        help="Goal latitude in degrees.")
    parser.add_argument('--goal-lon', type=float, required=True,
                        help="Goal longitude in degrees.")
    options = parser.parse_args()

    goal_lat = options.goal_lat
    goal_lon = options.goal_lon
    print(options)

    print(f"Goal position: {goal_lat:.6f}°, {goal_lon:.6f}°")
    print("Waiting for GPS data...\n")

    # Connect to the device.
    transport = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    transport.connect((socket.gethostbyname(options.hostname), options.port))

    # Listen for incoming data and parse FusionEngine messages.
    try:
        decoder = FusionEngineDecoder()
        last_print_time = 0
        while True:
            try:
                received_data = transport.recv(1024)
                messages = decoder.on_data(received_data)
                current_time = time.time()
                
                for header, message in messages:
                    try:
                        if isinstance(message, PoseMessage):
                            # Only update twice per second
                            if current_time - last_print_time >= 0.5:
                                current_lat = message.lla_deg[0]
                                current_lon = message.lla_deg[1]
                                current_alt = message.lla_deg[2]
                                
                                # Calculate bearing and distance to goal
                                bearing = calculate_bearing(current_lat, current_lon, goal_lat, goal_lon)
                                distance = calculate_distance(current_lat, current_lon, goal_lat, goal_lon)
                                direction = bearing_to_direction(bearing)

                                # Calculate vehicle heading from device yaw (Point One convention)
                                yaw_deg = message.ypr_deg[0]
                                if not math.isnan(yaw_deg):
                                    heading_deg = yaw_to_heading(yaw_deg)
                                    heading_direction = bearing_to_direction(heading_deg)
                                    turn = turn_instruction(heading_deg, bearing)
                                else:
                                    heading_deg = float('nan')
                                    heading_direction = "N/A"
                                    turn = "Heading not available"
                                
                                # Display information
                                print(f"Current: {current_lat:.6f}°, {current_lon:.6f}° [{current_alt:.2f}m]")
                                print(f"Distance to goal: {distance:.2f}m")
                                print(f"Bearing to goal: {bearing:.1f}° ({direction})")
                                if not math.isnan(heading_deg):
                                    print(f"Vehicle heading: {heading_deg:.1f}° ({heading_direction})")
                                else:
                                    print(f"Vehicle heading: N/A (stationary or no IMU data)")
                                print(f"Turn: {turn}")
                                print(f"Solution: {message.solution_type}")
                                print("-" * 60)
                                
                                last_print_time = current_time
                                
                                # Clear remaining messages in queue
                                break
                    except (AttributeError, IndexError, ValueError) as e:
                        # Ignore invalid messages
                        continue
            except (socket.error, OSError) as e:
                print(f"Connection error: {e}")
                time.sleep(1)
                continue
            except Exception as e:
                # Ignore other parsing errors
                continue
    except KeyboardInterrupt:
        print("\nExiting...")

    # Close the transport when finished.
    transport.close()
