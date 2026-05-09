#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "config.h"

enum class SelectableAlgorithm : uint8_t {
    Manual = 0,
    CloseObstacle = 1,
    LidarCorridor = 2,
    Gps = 3,
    Camera = 4,
    Count,
};

struct AlgorithmDescriptor {
    SelectableAlgorithm id;
    const char *key;
    const char *label;
    float weight;
    bool implemented;
};

inline constexpr uint32_t algorithmBit(SelectableAlgorithm id)
{
    return 1u << static_cast<uint8_t>(id);
}

inline constexpr size_t kSelectableAlgorithmCount =
    static_cast<size_t>(SelectableAlgorithm::Count);

inline const std::array<AlgorithmDescriptor, kSelectableAlgorithmCount> kSelectableAlgorithms = {{
    {SelectableAlgorithm::Manual, "manual", "Manual", MANUAL_WEIGHT, true},
    {SelectableAlgorithm::CloseObstacle, "close_obstacle", "Close obstacle", LIDAR_AVOIDANCE_WEIGHT, true},
    {SelectableAlgorithm::LidarCorridor, "lidar_corridor", "Corridor LiDAR", LIDAR_CORRIDOR_WEIGHT, true},
    {SelectableAlgorithm::Gps, "gps", "GPS", GPS_WEIGHT, true},
    {SelectableAlgorithm::Camera, "camera", "Camera", CAMEDAR_WEIGHT, false},
}};

inline constexpr uint32_t selectableAlgorithmKnownMask()
{
    return algorithmBit(SelectableAlgorithm::Manual)
        | algorithmBit(SelectableAlgorithm::CloseObstacle)
        | algorithmBit(SelectableAlgorithm::LidarCorridor)
        | algorithmBit(SelectableAlgorithm::Gps)
        | algorithmBit(SelectableAlgorithm::Camera);
}

inline constexpr uint32_t selectableAlgorithmImplementedMask()
{
    return algorithmBit(SelectableAlgorithm::Manual)
        | algorithmBit(SelectableAlgorithm::CloseObstacle)
        | algorithmBit(SelectableAlgorithm::LidarCorridor)
        | algorithmBit(SelectableAlgorithm::Gps);
}

inline constexpr uint32_t selectableAlgorithmDefaultMask()
{
    return algorithmBit(SelectableAlgorithm::Manual)
        | algorithmBit(SelectableAlgorithm::CloseObstacle)
        | algorithmBit(SelectableAlgorithm::LidarCorridor);
}

inline const AlgorithmDescriptor *algorithmDescriptor(SelectableAlgorithm id)
{
    const size_t index = static_cast<size_t>(id);

    if (index >= kSelectableAlgorithmCount) {
        return nullptr;
    }
    return &kSelectableAlgorithms[index];
}

inline const AlgorithmDescriptor *findAlgorithmDescriptorByKey(const char *key)
{
    if (key == nullptr) {
        return nullptr;
    }
    for (const AlgorithmDescriptor &descriptor : kSelectableAlgorithms) {
        if (std::strcmp(descriptor.key, key) == 0) {
            return &descriptor;
        }
    }
    return nullptr;
}

class AlgorithmSelector {
public:
    AlgorithmSelector()
        : selectedMask_(selectableAlgorithmDefaultMask())
    {
    }

    uint32_t getSelectedMask() const
    {
        return this->selectedMask_.load(std::memory_order_relaxed);
    }

    void setSelectedMask(uint32_t mask)
    {
        this->selectedMask_.store(mask & selectableAlgorithmImplementedMask(),
                                  std::memory_order_relaxed);
    }

    bool isEnabled(SelectableAlgorithm id) const
    {
        return (this->getSelectedMask() & algorithmBit(id)) != 0;
    }

    bool isManualDriveEnabled() const
    {
        return this->isEnabled(SelectableAlgorithm::Manual);
    }

private:
    std::atomic<uint32_t> selectedMask_;
};
