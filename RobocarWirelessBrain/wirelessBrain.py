import socket
import json
import time

UDP_IP = "192.168.12.1"
UDP_PORT = 5005

def send_command(speed, steering):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    message = json.dumps({"speed": speed, "steering": steering})
    sock.sendto(message.encode('utf-8'), (UDP_IP, UDP_PORT))
    print(f"Sent: {message}")
    sock.close()

def main():
    send_command(0.5, 0.0)  # Forward
    time.sleep(1)
    send_command(0.0, 0.5)  # Turn right
    time.sleep(1)
    send_command(0.0, 0.0)  # Stop

if __name__ == "__main__":
    main()
