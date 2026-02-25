/**************************************************************************/ /**
 * @brief Example of reading and decoding FusionEngine messages from a serial port.
 * @file
 ******************************************************************************/

#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>

#include <point_one/fusion_engine/messages/core.h>
#include <point_one/fusion_engine/parsers/fusion_engine_framer.h>

#include "../common/print_message.h"

using namespace point_one::fusion_engine::examples;
using namespace point_one::fusion_engine::messages;
using namespace point_one::fusion_engine::parsers;

speed_t get_baud(int baud_rate) {
  switch (baud_rate) {
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
    case 230400: return B230400;
    case 460800: return B460800;
    case 921600: return B921600;
    default: return B0;
  }
}

int open_serial_port(const char* device, int baud_rate) {
  int fd = open(device, O_RDWR | O_NOCTTY | O_SYNC);
  if (fd < 0) {
    printf("Error opening %s: %s\n", device, strerror(errno));
    return -1;
  }

  struct termios tty;
  if (tcgetattr(fd, &tty) != 0) {
    printf("Error from tcgetattr: %s\n", strerror(errno));
    close(fd);
    return -1;
  }

  speed_t speed = get_baud(baud_rate);
  if (speed == B0) {
    printf("Error: Unsupported baud rate %d\n", baud_rate);
    close(fd);
    return -1;
  }

  cfsetospeed(&tty, speed);
  cfsetispeed(&tty, speed);

  tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;     // 8-bit chars
  tty.c_cflag |= (CLOCAL | CREAD);                // ignore modem controls, enable reading
  tty.c_cflag &= ~(PARENB | PARODD);              // shut off parity
  tty.c_cflag &= ~CSTOPB;                         // 1 stop bit
  tty.c_cflag &= ~CRTSCTS;                        // no hardware flow control

  // Input flags - Turn off input processing
  tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON);

  // Output flags - Turn off output processing
  tty.c_oflag &= ~OPOST;

  // No line processing
  tty.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);

  // Fetch bytes as they become available
  tty.c_cc[VMIN] = 1;
  tty.c_cc[VTIME] = 0; // No timeout

  if (tcsetattr(fd, TCSANOW, &tty) != 0) {
    printf("Error from tcsetattr: %s\n", strerror(errno));
    close(fd);
    return -1;
  }
  return fd;
}

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    printf("Usage: %s DEVICE [BAUD_RATE]\n", argv[0]);
    printf("Example: %s /dev/ttyUSB0\n", argv[0]);
    return 0;
  }

  const char* device = argv[1];
  int baud_rate = 460800; // Default
  if (argc >= 3) {
    baud_rate = std::atoi(argv[2]);
  }

  int fd = open_serial_port(device, baud_rate);
  if (fd < 0) {
    return 1;
  }

  printf("Listening on %s at %d baud...\n", device, baud_rate);

  // Create a decoder and configure it to print when messages arrive.
  FusionEngineFramer framer(MessageHeader::MAX_MESSAGE_SIZE_BYTES);
  framer.SetMessageCallback(PrintMessage);

  // Read from serial port and decode.
  uint8_t buffer[4096];
  while (true) {
    ssize_t bytes_read = read(fd, buffer, sizeof(buffer));
    if (bytes_read > 0) {
        // Uncomment the line below to see raw hex dump of every chunk
        // PrintHex(buffer, bytes_read); printf("\n");
        framer.OnData(buffer, bytes_read);
    } else if (bytes_read < 0) {
        printf("Error reading: %s\n", strerror(errno));
        break;
    }
  }

  close(fd);
  return 0;
}
