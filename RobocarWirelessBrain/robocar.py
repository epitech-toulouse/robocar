import socket
import json
import threading
import time
from pyvesc import VESC

SERIAL_PORT = '/dev/ttyACM0'

# UDP settings
UDP_IP = "0.0.0.0"
UDP_PORT = 5005

class RobocarController:
    def __init__(self):
        self.speed = 0.0
        self.steering = 0.0
        self.running = True
        self.vesc = VESC(serial_port=SERIAL_PORT)
        self.thread = threading.Thread(target=self.control_loop)
        self.thread.start()

    def control_loop(self):
        current_speed = 0.0
        while self.running:
            # Smooth speed change
            if current_speed < self.speed:
                current_speed += 0.02
                if current_speed > self.speed:
                    current_speed = self.speed
            elif current_speed > self.speed:
                current_speed -= 0.05
                if current_speed < self.speed:
                    current_speed = self.speed

            self.vesc.set_duty_cycle(current_speed)
            self.vesc.set_servo((self.steering + 1) / 2)
            time.sleep(0.01)  # Small delay

    def set_speed(self, speed):
        self.speed = max(-1.0, min(1.0, speed))  # Clamp to -1 to 1

    def set_steering(self, steering):
        self.steering = max(-1.0, min(1.0, steering))  # Clamp to -1 to 1

    def stop(self):
        self.running = False
        self.speed = 0.0
        self.steering = 0.0
        self.thread.join()
        self.vesc.set_duty_cycle(0.0)

def udp_listener(controller):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening for UDP messages on {UDP_IP}:{UDP_PORT}")

    while controller.running:
        try:
            data, addr = sock.recvfrom(1024)
            message = data.decode('utf-8')
            print(f"Received message: {message} from {addr}")

            # Parse JSON message, e.g., {"speed": 0.5, "steering": 0.2}
            try:
                parsed = json.loads(message)
                if 'speed' in parsed:
                    controller.set_speed(float(parsed['speed']))
                if 'steering' in parsed:
                    controller.set_steering(float(parsed['steering']))
            except json.JSONDecodeError:
                print("Invalid JSON received")
        except Exception as e:
            print(f"Error receiving data: {e}")

    sock.close()

def main():
    controller = RobocarController()
    listener_thread = threading.Thread(target=udp_listener, args=(controller,))
    listener_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        controller.stop()
        listener_thread.join()

if __name__ == "__main__":
    main()