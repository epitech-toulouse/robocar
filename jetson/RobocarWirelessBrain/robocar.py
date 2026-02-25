import socket
import json
import threading
import time
import sys
sys.path.append('../follow-the-line/src')
from motor.Motor import Motor

SERIAL_PORT = '/dev/ttyACM0'

# UDP settings
UDP_IP = "0.0.0.0"
UDP_PORT = 5005

def udp_listener(motor):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening for UDP messages on {UDP_IP}:{UDP_PORT}")

    while True:
        try:
            data, addr = sock.recvfrom(1024)
            message = data.decode('utf-8')
            print(f"Received message: {message} from {addr}")

            # Parse JSON message, e.g., {"speed": 0.5, "steering": 0.2}
            try:
                parsed = json.loads(message)
                if 'speed' in parsed:
                    motor.set_speed_objective(float(parsed['speed']))
                if 'steering' in parsed:
                    motor.set_steering_objective(float(parsed['steering']))
            except json.JSONDecodeError:
                print("Invalid JSON received")
        except Exception as e:
            print(f"Error receiving data: {e}")

    sock.close()

def main():
    motor = Motor(SERIAL_PORT)
    listener_thread = threading.Thread(target=udp_listener, args=(motor,))
    listener_thread.start()

    try:
        while True:
            time.sleep(1)  # Keep main thread alive
    except KeyboardInterrupt:
        print("Stopping...")
        motor.stop()
        listener_thread.join()

if __name__ == "__main__":
    main()