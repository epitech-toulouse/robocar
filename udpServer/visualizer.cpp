#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <cmath>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>

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

// Shared data between input thread and main thread
std::deque<LidarPoint> scan_points;
std::mutex pointsMutex;
std::atomic<bool> running(true);

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

// Thread to read from stdin (piped from client)
void inputThread() {
    std::string line;
    while (running && std::getline(std::cin, line)) {
        // Replace commas with spaces to handle CSV format: "angle,distance,intensity"
        std::replace(line.begin(), line.end(), ',', ' ');
        
        std::stringstream ss(line);
        float angle, distance;
        int intensity_int = 0;
        
        if (ss >> angle >> distance) {
            // Try to read intensity if present
            if (ss >> intensity_int) {
                // valid
            } else {
                intensity_int = 200; // Default
            }
            
            // Filter valid points similar to the reference code
            // if (distance > 0.05f && distance < MAX_RANGE) {
                std::lock_guard<std::mutex> lock(pointsMutex);
                scan_points.push_back({angle, distance, (uint8_t)intensity_int});
                
                // Keep buffer size reasonable
                if (scan_points.size() > 2000) {
                    scan_points.pop_front();
                }
            // }
        }
    }
}

int main() {
    // Create window
    sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "LD19 LiDAR Visualizer (UDP Client)");
    window.setFramerateLimit(60);

    // Start input thread
    std::thread t(inputThread);

    // Font for text (optional)
    sf::Font font;
    // Try to load a common font, but don't crash if missing
    bool font_loaded = font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
    if (!font_loaded) {
        // Try another common path
        font_loaded = font.loadFromFile("/usr/share/fonts/liberation/LiberationSans-Regular.ttf");
    }

    sf::Text text;
    if (font_loaded) {
        text.setFont(font);
        text.setCharacterSize(16);
        text.setFillColor(sf::Color::White);
        text.setPosition(10, 10);
    }

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape) {
                window.close();
            }
        }

        // Clear window
        window.clear(sf::Color::Black);

        // Draw grid
        draw_grid(window);

        // Draw LiDAR points
        {
            std::lock_guard<std::mutex> lock(pointsMutex);
            sf::VertexArray points(sf::Points);
            
            for (const auto& point : scan_points) {
                // Convert polar to cartesian
                // Angle 0° = North (top), clockwise
                // The reference code uses: (90.0f - point.angle) * PI / 180.0f;
                // This assumes 0 is North and angle increases clockwise.
                
                float angle_rad = (90.0f - point.angle) * PI / 180.0f;
                float x = CENTER_X + point.distance * SCALE * std::cos(angle_rad);
                float y = CENTER_Y - point.distance * SCALE * std::sin(angle_rad);

                // Color based on intensity
                uint8_t intensity = point.intensity;
                // Reference code: sf::Color(255, 255 - intensity, 0, 200);
                // This makes high intensity -> Red/Orange? 
                // Let's stick to the reference code's coloring logic if possible, 
                // or just use Green for simplicity if intensity isn't calibrated same way.
                // Let's use the reference logic:
                sf::Color color(255, 255 - std::min((int)intensity, 255), 0, 200);

                points.append(sf::Vertex(sf::Vector2f(x, y), color));
            }
            window.draw(points);
            
            // Draw info text
            if (font_loaded) {
                text.setString("Points: " + std::to_string(scan_points.size()));
                window.draw(text);
            }
        }

        // Draw center marker
        sf::CircleShape center(3);
        center.setPosition(CENTER_X - 3, CENTER_Y - 3);
        center.setFillColor(sf::Color::Green);
        window.draw(center);

        window.display();
    }

    running = false;
    t.detach(); // Allow thread to finish or die with process

    return 0;
}
