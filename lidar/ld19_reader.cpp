#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <termios.h>
#include <unistd.h>
#include <vector>

const float MAX_RANGE = 12.0f;    // meters
const float ANGLE_OFFSET = 15.0f; // degrees

struct LidarPoint {
  float angle;    // degrees
  float distance; // meters
  uint8_t intensity;
};

class SerialPort {
private:
  int fd;

public:
  SerialPort(const char *port, int baudrate) {
    fd = open(port, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
      std::cerr << "Error opening serial port: " << port << std::endl;
      exit(1);
    }

    struct termios options;
    tcgetattr(fd, &options);

    // Set baudrate 230400
    speed_t speed = B230400;
    cfsetispeed(&options, speed);
    cfsetospeed(&options, speed);

    // 8N1
    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    options.c_cflag |= (CLOCAL | CREAD);

    // Raw mode
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_iflag &= ~(IXON | IXOFF | IXANY);
    options.c_oflag &= ~OPOST;

    tcsetattr(fd, TCSANOW, &options);
    tcflush(fd, TCIOFLUSH);
  }

  ~SerialPort() {
    if (fd >= 0)
      close(fd);
  }

  int read(uint8_t *buffer, int size) { return ::read(fd, buffer, size); }
};

class LD19Parser {
private:
  std::vector<uint8_t> buffer;

public:
  std::vector<LidarPoint> parse_packet(const uint8_t *packet, int len) {
    std::vector<LidarPoint> points;

    if (len < 47)
      return points;
    if (packet[0] != 0x54)
      return points;

    // Parse start angle (little endian, in 0.01 degree units)
    uint16_t start_angle_raw = packet[4] | (packet[5] << 8);
    float start_angle = start_angle_raw / 100.0f;

    // Parse end angle
    uint16_t end_angle_raw = packet[42] | (packet[43] << 8);
    float end_angle = end_angle_raw / 100.0f;

    // Calculate angle difference
    float angle_diff = end_angle - start_angle;
    if (angle_diff < 0)
      angle_diff += 360.0f;

    float angle_step = (angle_diff / 11.0f);

    // Parse 12 measurement points
    for (int i = 0; i < 12; i++) {
      int offset = 6 + i * 3;
      uint16_t distance = packet[offset] | (packet[offset + 1] << 8);
      uint8_t intensity = packet[offset + 2];

      float angle = start_angle + i * angle_step + ANGLE_OFFSET;
      if (angle >= 360.0f)
        angle -= 360.0f;
      else if (angle < 0.0f)
        angle += 360.0f;

      float dist_m = distance / 1000.0f;

      // Filter valid points
      if (dist_m > 0.05f && dist_m < MAX_RANGE && intensity > 0) {
        points.push_back({angle, dist_m, intensity});
      }
    }

    return points;
  }

  std::vector<LidarPoint> process_data(const uint8_t *data, int len) {
    std::vector<LidarPoint> all_points;

    // Add new data to buffer
    for (int i = 0; i < len; i++) {
      buffer.push_back(data[i]);
    }

    // Process packets
    while (buffer.size() >= 47) {
      // Find header
      auto it = std::find(buffer.begin(), buffer.end(), 0x54);

      if (it == buffer.end()) {
        buffer.clear();
        break;
      }

      // Remove data before header
      int header_idx = std::distance(buffer.begin(), it);
      if (header_idx > 0) {
        buffer.erase(buffer.begin(), it);
      }

      // Check if we have complete packet
      if (buffer.size() < 47)
        break;

      // Parse packet
      auto points = parse_packet(buffer.data(), 47);
      all_points.insert(all_points.end(), points.begin(), points.end());

      // Remove processed packet
      buffer.erase(buffer.begin(), buffer.begin() + 47);
    }

    return all_points;
  }
};

int main(int argc, char *argv[]) {
  std::string usb_port = "USB0"; // Default port

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
      usb_port = argv[++i];
    } else if (std::strncmp(argv[i], "/dev/", 5) == 0) {
      // Full path provided - use directly
      usb_port = argv[i];
    } else if (std::strncmp(argv[i], "USB", 3) == 0 ||
               std::strncmp(argv[i], "usbmodem", 8) == 0 ||
               std::strncmp(argv[i], "cu.", 3) == 0 ||
               std::strncmp(argv[i], "tty.", 4) == 0) {
      usb_port = argv[i];
    }
  }

  // Build serial path
  std::string serial_path;
  if (usb_port.find("/dev/") == 0) {
    serial_path = usb_port;
  } else if (usb_port.find("cu.") == 0 || usb_port.find("tty.") == 0) {
    serial_path = "/dev/" + usb_port;
  } else {
    serial_path = "/dev/tty" + usb_port;
  }

  std::cerr << "Opening " << serial_path << "..." << std::endl;

  SerialPort serial(serial_path.c_str(), 230400);
  LD19Parser parser;

  uint8_t read_buffer[1024];

  std::cerr << "Reading LIDAR data (pipe to server)..." << std::endl;

  // Output raw data to stdout (pipe to server)
  while (true) {
    int bytes_read = serial.read(read_buffer, sizeof(read_buffer));
    if (bytes_read > 0) {
      auto new_points = parser.process_data(read_buffer, bytes_read);
      for (const auto &point : new_points) {
        std::cout << point.angle << "," << point.distance << ","
                  << (int)point.intensity << std::endl;
      }
      std::cout.flush();
    }
    usleep(1000); // 1ms delay
  }

  return 0;
}
