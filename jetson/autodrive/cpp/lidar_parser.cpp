/**
 * LidarParser - C++ LD19 LiDAR Parser Implementation
 * 
 * Ported from Python LidarParser.py with structure from lidar/main.cpp
 */

#include "lidar_parser.hpp"
#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <iostream>
#include <algorithm>

class LidarParserImpl {
private:
    int fd;
    std::atomic<bool> running;
    std::atomic<bool> connected;
    std::thread worker_thread;
    std::mutex points_mutex;
    std::deque<LidarPoint> points;
    std::vector<uint8_t> buffer;
    float angle_offset;
    
    static const size_t MAX_POINTS = 2000;
    static const int PACKET_SIZE = 47;
    
    bool open_serial(const char* port, int baudrate) {
        fd = open(port, O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd < 0) {
            return false;
        }
        
        struct termios options;
        tcgetattr(fd, &options);
        
        // Set baudrate (230400 for LD19)
        speed_t speed = B230400;
        if (baudrate == 115200) speed = B115200;
        cfsetispeed(&options, speed);
        cfsetospeed(&options, speed);
        
        // 8N1 configuration
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
        
        return true;
    }
    
    void parse_packet(const uint8_t* packet) {
        // Packet structure for LD19:
        // [0]: 0x54 header
        // [1]: VerLen
        // [2-3]: Speed
        // [4-5]: Start angle (little endian, 0.01 deg units)
        // [6-41]: 12 points, 3 bytes each (distance LE + intensity)
        // [42-43]: End angle
        // [44-45]: Timestamp
        // [46]: CRC
        
        uint16_t start_angle_raw = packet[4] | (packet[5] << 8);
        float start_angle = start_angle_raw / 100.0f;
        
        uint16_t end_angle_raw = packet[42] | (packet[43] << 8);
        float end_angle = end_angle_raw / 100.0f;
        
        float angle_diff = end_angle - start_angle;
        if (angle_diff < 0) angle_diff += 360.0f;
        
        float angle_step = angle_diff / 11.0f;
        
        std::vector<LidarPoint> new_points;
        
        for (int i = 0; i < 12; i++) {
            int offset = 6 + i * 3;
            uint16_t distance_mm = packet[offset] | (packet[offset + 1] << 8);
            uint8_t intensity = packet[offset + 2];
            
            float angle = start_angle + i * angle_step + angle_offset;
            if (angle >= 360.0f) angle -= 360.0f;
            else if (angle < 0) angle += 360.0f;
            
            float distance_m = distance_mm / 1000.0f;
            
            // Filter valid points (same as Python version)
            if (distance_m > 0.05f && distance_m < 12.0f && intensity > 0) {
                new_points.push_back({angle, distance_m});
            }
        }
        
        // Add to shared buffer
        {
            std::lock_guard<std::mutex> lock(points_mutex);
            for (const auto& p : new_points) {
                points.push_back(p);
            }
            while (points.size() > MAX_POINTS) {
                points.pop_front();
            }
        }
    }
    
    void process_buffer() {
        while (buffer.size() >= PACKET_SIZE) {
            // Find header 0x54
            auto it = std::find(buffer.begin(), buffer.end(), 0x54);
            
            if (it == buffer.end()) {
                buffer.clear();
                break;
            }
            
            // Remove bytes before header
            size_t header_idx = std::distance(buffer.begin(), it);
            if (header_idx > 0) {
                buffer.erase(buffer.begin(), it);
            }
            
            // Check if we have a complete packet
            if (buffer.size() < PACKET_SIZE) break;
            
            // Parse the packet
            parse_packet(buffer.data());
            
            // Remove processed packet
            buffer.erase(buffer.begin(), buffer.begin() + PACKET_SIZE);
        }
    }
    
    void worker_loop(const char* port, int baudrate) {
        // Try to connect
        const char* ports[] = {"/dev/ttyUSB0", "/dev/ttyUSB1", nullptr};
        
        while (running && !connected) {
            // Try specified port first
            if (open_serial(port, baudrate)) {
                connected = true;
                std::cerr << "LidarParser connected to " << port << std::endl;
                break;
            }
            
            // Try default ports
            for (int i = 0; ports[i] != nullptr; i++) {
                if (open_serial(ports[i], baudrate)) {
                    connected = true;
                    std::cerr << "LidarParser connected to " << ports[i] << std::endl;
                    break;
                }
            }
            
            if (!connected) {
                usleep(1000000); // 1 second wait before retry
            }
        }
        
        uint8_t read_buffer[1024];
        
        while (running && connected) {
            ssize_t bytes_read = read(fd, read_buffer, sizeof(read_buffer));
            
            if (bytes_read > 0) {
                // Append to internal buffer
                buffer.insert(buffer.end(), read_buffer, read_buffer + bytes_read);
                process_buffer();
            } else if (bytes_read < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
                // Error reading
                usleep(100000); // 100ms
            } else {
                // No data available
                usleep(1000); // 1ms
            }
        }
        
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
    }
    
public:
    LidarParserImpl(const char* port, int baudrate, float offset)
        : fd(-1), running(true), connected(false), angle_offset(offset) {
        worker_thread = std::thread(&LidarParserImpl::worker_loop, this, port, baudrate);
    }
    
    ~LidarParserImpl() {
        stop();
    }
    
    void stop() {
        running = false;
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
    }
    
    int get_points(LidarPoint* out_buffer, int max_points) {
        std::lock_guard<std::mutex> lock(points_mutex);
        int count = std::min(static_cast<int>(points.size()), max_points);
        for (int i = 0; i < count; i++) {
            out_buffer[i] = points[i];
        }
        return count;
    }
    
    bool is_connected() const {
        return connected;
    }
};

// C API Implementation
extern "C" {

LidarParserHandle lidar_parser_create(const char* port, int baudrate, float angle_offset) {
    return new LidarParserImpl(port, baudrate, angle_offset);
}

void lidar_parser_destroy(LidarParserHandle handle) {
    if (handle) {
        delete static_cast<LidarParserImpl*>(handle);
    }
}

int lidar_parser_get_points(LidarParserHandle handle, LidarPoint* buffer, int max_points) {
    if (!handle) return 0;
    return static_cast<LidarParserImpl*>(handle)->get_points(buffer, max_points);
}

void lidar_parser_stop(LidarParserHandle handle) {
    if (handle) {
        static_cast<LidarParserImpl*>(handle)->stop();
    }
}

int lidar_parser_is_connected(LidarParserHandle handle) {
    if (!handle) return 0;
    return static_cast<LidarParserImpl*>(handle)->is_connected() ? 1 : 0;
}

} // extern "C"
