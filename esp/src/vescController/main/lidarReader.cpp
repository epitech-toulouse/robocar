#include "lidarReader.hpp"

#include <algorithm>
#include <cmath>

#include "config.h"
#include "driver/uart.h"

namespace {
constexpr std::array<uint8_t, 256> LD19_CRC8_TABLE = {
    0x00, 0x4D, 0x9A, 0xD7, 0x79, 0x34, 0xE3, 0xAE, 0xF2, 0xBF, 0x68, 0x25, 0x8B, 0xC6, 0x11, 0x5C,
    0xA9, 0xE4, 0x33, 0x7E, 0xD0, 0x9D, 0x4A, 0x07, 0x5B, 0x16, 0xC1, 0x8C, 0x22, 0x6F, 0xB8, 0xF5,
    0x1F, 0x52, 0x85, 0xC8, 0x66, 0x2B, 0xFC, 0xB1, 0xED, 0xA0, 0x77, 0x3A, 0x94, 0xD9, 0x0E, 0x43,
    0xB6, 0xFB, 0x2C, 0x61, 0xCF, 0x82, 0x55, 0x18, 0x44, 0x09, 0xDE, 0x93, 0x3D, 0x70, 0xA7, 0xEA,
    0x3E, 0x73, 0xA4, 0xE9, 0x47, 0x0A, 0xDD, 0x90, 0xCC, 0x81, 0x56, 0x1B, 0xB5, 0xF8, 0x2F, 0x62,
    0x97, 0xDA, 0x0D, 0x40, 0xEE, 0xA3, 0x74, 0x39, 0x65, 0x28, 0xFF, 0xB2, 0x1C, 0x51, 0x86, 0xCB,
    0x21, 0x6C, 0xBB, 0xF6, 0x58, 0x15, 0xC2, 0x8F, 0xD3, 0x9E, 0x49, 0x04, 0xAA, 0xE7, 0x30, 0x7D,
    0x88, 0xC5, 0x12, 0x5F, 0xF1, 0xBC, 0x6B, 0x26, 0x7A, 0x37, 0xE0, 0xAD, 0x03, 0x4E, 0x99, 0xD4,
    0x7C, 0x31, 0xE6, 0xAB, 0x05, 0x48, 0x9F, 0xD2, 0x8E, 0xC3, 0x14, 0x59, 0xF7, 0xBA, 0x6D, 0x20,
    0xD5, 0x98, 0x4F, 0x02, 0xAC, 0xE1, 0x36, 0x7B, 0x27, 0x6A, 0xBD, 0xF0, 0x5E, 0x13, 0xC4, 0x89,
    0x63, 0x2E, 0xF9, 0xB4, 0x1A, 0x57, 0x80, 0xCD, 0x91, 0xDC, 0x0B, 0x46, 0xE8, 0xA5, 0x72, 0x3F,
    0xCA, 0x87, 0x50, 0x1D, 0xB3, 0xFE, 0x29, 0x64, 0x38, 0x75, 0xA2, 0xEF, 0x41, 0x0C, 0xDB, 0x96,
    0x42, 0x0F, 0xD8, 0x95, 0x3B, 0x76, 0xA1, 0xEC, 0xB0, 0xFD, 0x2A, 0x67, 0xC9, 0x84, 0x53, 0x1E,
    0xEB, 0xA6, 0x71, 0x3C, 0x92, 0xDF, 0x08, 0x45, 0x19, 0x54, 0x83, 0xCE, 0x60, 0x2D, 0xFA, 0xB7,
    0x5D, 0x10, 0xC7, 0x8A, 0x24, 0x69, 0xBE, 0xF3, 0xAF, 0xE2, 0x35, 0x78, 0xD6, 0x9B, 0x4C, 0x01,
    0xF4, 0xB9, 0x6E, 0x23, 0x8D, 0xC0, 0x17, 0x5A, 0x06, 0x4B, 0x9C, 0xD1, 0x7F, 0x32, 0xE5, 0xA8};
}

LidarReader::LidarReader(float angleOffsetDeg)
    : angleOffsetDeg(angleOffsetDeg),
      running(false),
      buffer(),
      scanInProgress(),
      currentWorld(),
      hasLastPacketStartAngle(false),
      lastPacketStartAngle(0.0f),
      completedScanCount(0),
      readBuffer{} {
}

LidarReader::~LidarReader() {
    stop();
}

void LidarReader::stop() {
    if (!running) {
        return;
    }

    running = false;
    buffer.clear();
    scanInProgress.clear();
    hasLastPacketStartAngle = false;
    completedScanCount = 0;
}

int LidarReader::start() {
    printf("LidarReader: start() called\n");
    if (running) {
        return ESP_OK;
    }

    buffer.clear();
    scanInProgress.clear();
    currentWorld.clear();
    hasLastPacketStartAngle = false;
    completedScanCount = 0;
    running = true;
    return ESP_OK;
}

bool LidarReader::isRunning() const {
    return running;
}

bool LidarReader::update() {
    if (!running) {
        return false;
    }

    bool consumedBytes = false;

    TickType_t timeout = 10;
    while (true) {
        const int bytesRead = uart_read_bytes(LIDAR_UART_PORT, readBuffer.data(), readBuffer.size(), timeout);
        if (bytesRead <= 0) {
            break;
        }

        // printf("LidarReader: read %d bytes\n", bytesRead); // Commented out to avoid spam, uncomment if needed

        timeout = 0; // Only block on the first read, then drain whatever is left and return

        consumedBytes = true;
        processBytes(readBuffer.data(), static_cast<std::size_t>(bytesRead));
    }

    return consumedBytes;
}

void LidarReader::processBytes(const uint8_t* data, std::size_t len) {
    buffer.insert(buffer.end(), data, data + len);

    while (buffer.size() >= PACKET_SIZE) {
        const auto headerIt = std::find(buffer.begin(), buffer.end(), HEADER_BYTE);
        if (headerIt == buffer.end()) {
            buffer.clear();
            return;
        }

        if (headerIt != buffer.begin()) {
            buffer.erase(buffer.begin(), headerIt);
        }

        if (buffer.size() < PACKET_SIZE) {
            return;
        }

        consumePacket(buffer.data());
        buffer.erase(buffer.begin(), buffer.begin() + PACKET_SIZE);
    }
}

bool LidarReader::parsePacket(const uint8_t* packet, float* packetStartAngle, std::vector<LidarPoint>* outPoints) const {
    if (packet[0] != HEADER_BYTE || packet[1] != EXPECTED_VER_LEN) {
        // printf("LidarReader: Bad header %02x %02x\n", packet[0], packet[1]);
        return false;
    }

    const uint8_t expectedCrc = packet[PACKET_SIZE - 1];
    const uint8_t computedCrc = computeCrc8(packet, PACKET_SIZE - 1);
    if (expectedCrc != computedCrc) {
        // printf("LidarReader: CRC mismatch expected=%02x computed=%02x\n", expectedCrc, computedCrc);
        return false;
    }

    const uint16_t startAngleRaw = packet[4] | (static_cast<uint16_t>(packet[5]) << 8);
    const uint16_t endAngleRaw = packet[42] | (static_cast<uint16_t>(packet[43]) << 8);

    const float startAngle = static_cast<float>(startAngleRaw) / 100.0f;
    const float endAngle = static_cast<float>(endAngleRaw) / 100.0f;

    float angleDiff = endAngle - startAngle;
    if (angleDiff < 0.0f) {
        angleDiff += 360.0f;
    }

    const float angleStep = angleDiff / static_cast<float>(POINTS_PER_PACKET - 1);

    outPoints->clear();
    outPoints->reserve(POINTS_PER_PACKET);

    for (std::size_t i = 0; i < POINTS_PER_PACKET; ++i) {
        const std::size_t offset = 6 + i * 3;
        const uint16_t distanceMm = packet[offset] | (static_cast<uint16_t>(packet[offset + 1]) << 8);
        const uint8_t intensity = packet[offset + 2];
        const float distanceMeters = static_cast<float>(distanceMm) / 1000.0f;

        if (distanceMeters <= MIN_RANGE_METERS || distanceMeters >= MAX_RANGE_METERS || intensity == 0) {
            continue;
        }

        const float angle = normalizeAngle(startAngle + static_cast<float>(i) * angleStep + angleOffsetDeg);
        outPoints->push_back({angle, distanceMeters, intensity});
    }

    *packetStartAngle = normalizeAngle(startAngle + angleOffsetDeg);
    return true;
}

void LidarReader::consumePacket(const uint8_t* packet) {
    std::vector<LidarPoint> packetPoints;
    float packetStartAngle = 0.0f;
    if (!parsePacket(packet, &packetStartAngle, &packetPoints)) {
        return;
    }

    // Start a new 360-degree frame when packet start angle wraps around.
    if (hasLastPacketStartAngle && packetStartAngle < lastPacketStartAngle) {
        std::sort(scanInProgress.begin(), scanInProgress.end(), [](const LidarPoint& a, const LidarPoint& b) {
            return a.angleDeg < b.angleDeg;
        });
        currentWorld = scanInProgress;
        ++completedScanCount;
        scanInProgress.clear();
    }

    scanInProgress.insert(scanInProgress.end(), packetPoints.begin(), packetPoints.end());
    hasLastPacketStartAngle = true;
    lastPacketStartAngle = packetStartAngle;
}

float LidarReader::normalizeAngle(float angleDeg) {
    angleDeg = std::fmod(angleDeg, 360.0f);
    if (angleDeg < 0.0f) {
        angleDeg += 360.0f;
    }
    return angleDeg;
}

uint8_t LidarReader::computeCrc8(const uint8_t* data, std::size_t len) {
    uint8_t crc = 0;
    for (std::size_t i = 0; i < len; ++i) {
        crc = LD19_CRC8_TABLE[crc ^ data[i]];
    }
    return crc;
}
