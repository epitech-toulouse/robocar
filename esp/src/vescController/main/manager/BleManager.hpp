#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "freertos/FreeRTOS.h"

struct ble_gap_event;
struct ble_gatt_access_ctxt;

enum class BleEndpoint : uint8_t {
    Control,
    Camera,
};

struct BleMessage {
    uint32_t sequence = 0;
    TickType_t tick = 0;
    uint16_t connHandle = 0;
    BleEndpoint endpoint = BleEndpoint::Control;
    std::string payload;
};

class BleManager {
public:
    static BleManager &instance();

    bool start();
    void stop();
    bool isStarted() const;
    bool isConnected() const;
    int connectedClientCount() const;

    uint32_t latestSequence() const;
    std::vector<BleMessage> messagesSince(uint32_t sequence, BleEndpoint endpoint) const;

    void setControlResponse(const std::string &payload);
    std::string controlResponse() const;
    void notifyControlSubscribers(const std::string &payload);

    static int gapEventHandler(struct ble_gap_event *event, void *arg);
    static int gattAccessHandler(uint16_t connHandle,
                                 uint16_t attrHandle,
                                 struct ble_gatt_access_ctxt *ctxt,
                                 void *arg);
    static void onSync();
    static void onReset(int reason);
    static void hostTask(void *arg);

private:
    static constexpr size_t kMaxMessages = 128;
    static constexpr size_t kMaxValueSize = 512;
    static constexpr uint16_t kAppearanceGenericRemote = 384;

    BleManager() = default;

    void restartAdvertising();
    void addConnection(uint16_t connHandle);
    void removeConnection(uint16_t connHandle);
    void appendMessage(BleEndpoint endpoint, uint16_t connHandle, const std::string &payload);
    int handleAccess(uint16_t connHandle, uint16_t attrHandle, struct ble_gatt_access_ctxt *ctxt);

    mutable std::mutex mutex_;
    bool started_ = false;
    uint32_t sequence_ = 0;
    uint16_t controlCharacteristicHandle_ = 0;
    uint16_t cameraCharacteristicHandle_ = 0;
    std::string controlValue_ = "STATUS:{\"ok\":true,\"service\":\"robocar_ble\"}";
    std::vector<uint16_t> connHandles_;
    std::vector<BleMessage> messages_;
};
