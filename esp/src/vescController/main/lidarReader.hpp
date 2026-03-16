#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

struct LidarPoint {
    float angleDeg;       // degrees
    float distanceMeters; // meters
    uint8_t intensity;
};

class LidarReader {
public:
    static constexpr std::size_t POINTS_PER_PACKET = 12;
    static constexpr std::size_t PACKET_SIZE = 47;
    static constexpr uint8_t HEADER_BYTE = 0x54;
    static constexpr uint8_t EXPECTED_VER_LEN = 0x2C;

    LidarReader(int rxPin, int txPin = -1, int uartNum = 2, float angleOffsetDeg = 15.0f);
    ~LidarReader();

    int start();
    void stop();
    bool isRunning() const;

    // Pull bytes from UART and update internal parser/scan state.
    bool update();

    // Backwards-compatible alias for update().
    bool poll() {
        return update();
    }

    // Latest completed 360-degree scan, sorted by angle.
    std::vector<LidarPoint> getCurrentWorld() const {
        return currentWorld;
    }

    // Backwards-compatible alias used by existing control code.
    std::vector<LidarPoint> getLatestScanPoints() const {
        return getCurrentWorld();
    }

private:
    static constexpr int BAUD_RATE = 230400;
    static constexpr int READ_BUFFER_SIZE = 256;
    static constexpr int DRIVER_BUFFER_SIZE = 1024;
    static constexpr float MIN_RANGE_METERS = 0.05f;
    static constexpr float MAX_RANGE_METERS = 12.0f;

    int uartNum;
    int txPin;
    int rxPin;
    float angleOffsetDeg;
    bool running;

    std::vector<uint8_t> buffer;
    std::vector<LidarPoint> scanInProgress;
    std::vector<LidarPoint> currentWorld;

    bool hasLastPacketStartAngle;
    float lastPacketStartAngle;

    std::array<uint8_t, READ_BUFFER_SIZE> readBuffer;

    int initUart();
    void processBytes(const uint8_t* data, std::size_t len);
    bool parsePacket(const uint8_t* packet, float* packetStartAngle, std::vector<LidarPoint>* outPoints) const;
    void consumePacket(const uint8_t* packet);
    static float normalizeAngle(float angleDeg);
    static uint8_t computeCrc8(const uint8_t* data, std::size_t len);
};
