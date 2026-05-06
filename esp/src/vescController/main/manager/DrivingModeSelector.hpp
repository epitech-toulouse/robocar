#pragma once

#include <atomic>
#include <cstdint>

enum class AutonomousDrivingMode : uint8_t {
    CorridorLidar = 0,
    Fusion = 1,
};

class DrivingModeSelector {
public:
    DrivingModeSelector() = default;
    ~DrivingModeSelector() = default;

    AutonomousDrivingMode getMode() const
    {
        return this->isFusionMode() ? AutonomousDrivingMode::Fusion
                                    : AutonomousDrivingMode::CorridorLidar;
    }

    bool isFusionMode() const
    {
        return this->fusionMode_.load(std::memory_order_relaxed);
    }

    bool isManualDriveEnabled() const
    {
        return this->isFusionMode();
    }

    void setMode(AutonomousDrivingMode mode)
    {
        this->setFusionMode(mode == AutonomousDrivingMode::Fusion);
    }

    void setFusionMode(bool enabled)
    {
        this->fusionMode_.store(enabled, std::memory_order_relaxed);
    }

    const char *modeString() const
    {
        return modeToString(this->getMode());
    }

    static const char *modeToString(AutonomousDrivingMode mode)
    {
        return mode == AutonomousDrivingMode::Fusion ? "FUSION" : "CORRIDOR_LIDAR";
    }

private:
    std::atomic<bool> fusionMode_{true};
};
