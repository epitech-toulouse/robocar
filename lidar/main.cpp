#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>

const float PI = 3.14159265f;
const int WINDOW_SIZE = 800;
const int CENTER_X = WINDOW_SIZE / 2;
const int CENTER_Y = WINDOW_SIZE / 2;
const float MAX_RANGE = 12.0f; // meters
const float SCALE = (WINDOW_SIZE / 2 - 50) / MAX_RANGE; // pixels per meter

struct LidarPoint {
    float angle;    // degrees
    float distance; // meters
    uint8_t intensity;
};

class SerialPort {
private:
    int fd;
    
public:
    SerialPort(const char* port, int baudrate) {
        fd = open(port, O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd < 0) {
            std::cerr << "Error opening serial port" << std::endl;
            exit(1);
        }
        
        struct termios options;
        tcgetattr(fd, &options);
        
        // Set baudrate
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
        if (fd >= 0) close(fd);
    }
    
    int read(uint8_t* buffer, int size) {
        return ::read(fd, buffer, size);
    }
};

class LD19Parser {
private:
    std::vector<uint8_t> buffer;
    
public:
    std::vector<LidarPoint> parse_packet(const uint8_t* packet, int len) {
        std::vector<LidarPoint> points;
        
        if (len < 47) return points;
        if (packet[0] != 0x54) return points;
        
        // Parse start angle (little endian, in 0.01 degree units)
        uint16_t start_angle_raw = packet[4] | (packet[5] << 8);
        float start_angle = start_angle_raw / 100.0f;
        
        // Parse end angle
        uint16_t end_angle_raw = packet[42] | (packet[43] << 8);
        float end_angle = end_angle_raw / 100.0f;
        
        // Calculate angle difference
        float angle_diff = end_angle - start_angle;
        if (angle_diff < 0) angle_diff += 360.0f;
        
        float angle_step = (angle_diff / 11.0f);
        
        // Parse 12 measurement points
        for (int i = 0; i < 12; i++) {
            int offset = 6 + i * 3;
            uint16_t distance = packet[offset] | (packet[offset + 1] << 8);
            uint8_t intensity = packet[offset + 2];
            
            float angle = start_angle + i * angle_step;
            if (angle >= 360.0f) angle -= 360.0f;
            
            float dist_m = distance / 1000.0f;
            
            // Filter valid points
            if (dist_m > 0.05f && dist_m < MAX_RANGE && intensity > 0) {
                points.push_back({angle, dist_m, intensity});
            }
        }
        
        return points;
    }
    
    std::vector<LidarPoint> process_data(const uint8_t* data, int len) {
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
            if (buffer.size() < 47) break;
            
            // Parse packet
            auto points = parse_packet(buffer.data(), 47);
            all_points.insert(all_points.end(), points.begin(), points.end());
            
            // Remove processed packet
            buffer.erase(buffer.begin(), buffer.begin() + 47);
        }
        
        return all_points;
    }
};

void draw_grid(sf::RenderWindow& window) {
    // Draw range circles
    for (int r = 2; r <= MAX_RANGE; r += 2) {
        sf::CircleShape circle(r * SCALE);
        circle.setPosition(CENTER_X - r * SCALE, CENTER_Y - r * SCALE);
        circle.setFillColor(sf::Color::Transparent);
        circle.setOutlineColor(sf::Color(50, 50, 50));
        circle.setOutlineThickness(1);
        window.draw(circle);
    }
    
    // Draw crosshairs
    sf::Vertex line1[] = {
        sf::Vertex(sf::Vector2f(CENTER_X, 0), sf::Color(50, 50, 50)),
        sf::Vertex(sf::Vector2f(CENTER_X, WINDOW_SIZE), sf::Color(50, 50, 50))
    };
    sf::Vertex line2[] = {
        sf::Vertex(sf::Vector2f(0, CENTER_Y), sf::Color(50, 50, 50)),
        sf::Vertex(sf::Vector2f(WINDOW_SIZE, CENTER_Y), sf::Color(50, 50, 50))
    };
    window.draw(line1, 2, sf::Lines);
    window.draw(line2, 2, sf::Lines);
}

int main() {
    // Initialize serial port
    SerialPort serial("/dev/ttyUSB0", 230400);
    LD19Parser parser;
    
    // Create window
    sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "LD19 LiDAR SFML");
    window.setFramerateLimit(60);
    
    // Store points
    std::deque<LidarPoint> scan_points;
    const size_t MAX_POINTS = 2000;
    
    // Font for text (optional)
    sf::Font font;
    bool font_loaded = font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
    
    sf::Text text;
    if (font_loaded) {
        text.setFont(font);
        text.setCharacterSize(16);
        text.setFillColor(sf::Color::White);
        text.setPosition(10, 10);
    }
    
    uint8_t read_buffer[1024];
    int frame_count = 0;
    
    std::cout << "Starting LD19 LiDAR visualization..." << std::endl;
    std::cout << "Close window to exit" << std::endl;
    
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }
        
        // Read serial data
        int bytes_read = serial.read(read_buffer, sizeof(read_buffer));
        if (bytes_read > 0) {
            auto new_points = parser.process_data(read_buffer, bytes_read);
            
            for (const auto& point : new_points) {
                scan_points.push_back(point);
                if (scan_points.size() > MAX_POINTS) {
                    scan_points.pop_front();
                }
            }
            
            frame_count++;
        }
        
        // Clear window
        window.clear(sf::Color::Black);
        
        // Draw grid
        draw_grid(window);
        
        // Draw LiDAR points
        sf::VertexArray points(sf::Points);
        for (const auto& point : scan_points) {
            // Convert polar to cartesian
            // Angle 0° = North (top), clockwise
            float angle_rad = (90.0f - point.angle) * PI / 180.0f;
            float x = CENTER_X + point.distance * SCALE * std::cos(angle_rad);
            float y = CENTER_Y - point.distance * SCALE * std::sin(angle_rad);
            
            // Color based on intensity
            uint8_t intensity = point.intensity;
            sf::Color color(255, 255 - intensity, 0, 200);
            
            points.append(sf::Vertex(sf::Vector2f(x, y), color));
        }
        window.draw(points);
        
        // Draw center marker
        sf::CircleShape center(3);
        center.setPosition(CENTER_X - 3, CENTER_Y - 3);
        center.setFillColor(sf::Color::Green);
        window.draw(center);
        
        // Draw info text
        if (font_loaded) {
            text.setString("LD19 LiDAR\nPoints: " + std::to_string(scan_points.size()) + 
                          "\nFrames: " + std::to_string(frame_count));
            window.draw(text);
        }
        
        window.display();
    }
    
    std::cout << "Stopped" << std::endl;
    return 0;
}