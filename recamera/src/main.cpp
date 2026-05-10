#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cvi_sys.h>
#include <sscma.h>
#include <video.h>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

using namespace ma;

#define TAG "autonomous_detector"

namespace {

constexpr int kStopSignClassId = 11;
constexpr float kStopSignDistanceConstant = 250.0f;
constexpr float kStopDistanceCm = 1300.0f;
constexpr int kLaneClassId = 1;
constexpr int kDefaultLaneTick = 1;
constexpr int kDefaultStopTick = 5;
constexpr int kDefaultLaneCaptureSize = 128;
constexpr int kDefaultStopCaptureSize = 320;
constexpr int kStopTimeSeconds = 4;
constexpr int kStopDebounceSeconds = 3;
constexpr float kDefaultStopThreshold = 0.5f;
constexpr float kBarricadeThreshold = 0.40f;
constexpr float kLaneSweepEndAt512 = 100.0f;
constexpr int kLaneMinCenteredRowsMin = 4;
constexpr float kLaneMinCenteredRowsPercent = 0.08f;
constexpr float kLaneFallbackDeadbandPercent = 12.5f;
constexpr float kLaneFallbackScale = 0.85f;
constexpr float kLaneBoundaryFallbackCenterFactor = 0.30f;
constexpr float kLaneFallbackMaxSteerPercent = 60.0f;
constexpr float kLaneSignalLossRememberScale = 0.92f;
constexpr float kLaneSignalLossMaxRememberedSteerPercent = 55.0f;
constexpr float kDefaultLaneWidthAt512 = 280.0f;
constexpr float kLaneWeightBoundaryZonePercent = 0.35f;
constexpr float kSingleLineCentralZonePercent = 0.25f;
constexpr float kSingleLineWideRunPercent = 0.18f;
constexpr float kRaycastMinScoreMarginPercent = 0.10f;
constexpr int kRaycastRaysPerSide = 5;
constexpr size_t kLaneCenterHistorySize = 5;
constexpr int kLaneSignalLossHoldFrames = 6;
constexpr float kLaneMean[3] = {123.675f, 116.28f, 103.53f};
constexpr float kLaneScale[3] = {0.01712475f, 0.01750700f, 0.01742919f};
constexpr float kLaneMaskThreshold = 0.25f;
int g_current_frame_count = 0;
int g_last_known_lane_width = 0;
float g_last_confident_steering_percent = 0.0f;
int g_frames_since_confident_lane = 1000000;

struct LaneCenterPoint {
    float x = 0.0f;
    float y = 0.0f;
};

std::vector<LaneCenterPoint> g_lane_center_history;

struct LoadedModel {
    ma::engine::EngineCVI* engine = nullptr;
    ma::Model* model = nullptr;
    int input_width = 0;
    int input_height = 0;
};

 struct LaneDecision {
     float steering_percent = 0.0f;
     int weight = 0;
     const char* status = "SEARCHING";
     bool confident = false;
 };

int min_centered_lane_rows(int height) {
    return std::max(kLaneMinCenteredRowsMin, static_cast<int>(height * kLaneMinCenteredRowsPercent));
}

struct CpuReadableFrame {
    ma_img_t frame{};
    std::vector<uint8_t> owned_buffer;
    void* mapped_ptr = nullptr;
    uint32_t mapped_size = 0;

    ~CpuReadableFrame() {
        if (mapped_ptr != nullptr && mapped_size != 0) {
            CVI_SYS_Munmap(mapped_ptr, mapped_size);
        }
    }
};

bool dump_mat_to_path(const ::cv::Mat& rgb, const char* dump_path, const char* label);

bool model_disabled(const char* path) {
    return path == nullptr || std::strcmp(path, "-") == 0 || std::strcmp(path, "none") == 0 ||
           std::strcmp(path, "NONE") == 0;
}

size_t cvi_shared_memory_bytes() {
    const char* value = std::getenv("CVI_SHARED_MEMORY_MB");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }

    char* end = nullptr;
    const unsigned long mb = std::strtoul(value, &end, 10);
    if (end == value) {
        MA_LOGW(TAG, "invalid CVI_SHARED_MEMORY_MB=%s, using model default", value);
        return 0;
    }

    return static_cast<size_t>(mb) * 1024 * 1024;
}

int env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0 || parsed > 4096) {
        MA_LOGW(TAG, "invalid %s=%s, using %d", name, value, fallback);
        return fallback;
    }

    return static_cast<int>(parsed);
}

bool env_flag(const char* name, bool fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    if (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 || std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "yes") == 0 || std::strcmp(value, "YES") == 0) {
        return true;
    }
    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 || std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "no") == 0 || std::strcmp(value, "NO") == 0) {
        return false;
    }

    MA_LOGW(TAG, "invalid %s=%s, using %d", name, value, fallback ? 1 : 0);
    return fallback;
}

int env_nonneg_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed < 0 || parsed > 60000) {
        MA_LOGW(TAG, "invalid %s=%s, using %d", name, value, fallback);
        return fallback;
    }

    return static_cast<int>(parsed);
}

int setup_serial(const char* device) {
    int fd = open(device, O_RDWR | O_NOCTTY);
    if (fd == -1) {
        perror("Unable to open UART");
        return -1;
    }

    termios options;
    if (tcgetattr(fd, &options) != 0) {
        perror("Unable to read UART options");
        close(fd);
        return -1;
    }

    cfsetispeed(&options, B115200);
    cfsetospeed(&options, B115200);

    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    options.c_cflag |= (CLOCAL | CREAD);
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_oflag &= ~OPOST;
    options.c_iflag &= ~(IXON | IXOFF | IXANY);

    if (tcsetattr(fd, TCSANOW, &options) != 0) {
        perror("Unable to apply UART options");
        close(fd);
        return -1;
    }

    return fd;
}

void send_uart(int uart_fd, const std::string& command) {
    if (uart_fd < 0) {
        return;
    }
    write(uart_fd, command.c_str(), command.size());
}

LoadedModel load_model(const char* path, const char* label, size_t algorithm_id = 0) {
    LoadedModel loaded;
    if (model_disabled(path)) {
        return loaded;
    }
    if (access(path, R_OK) != 0) {
        MA_LOGE(TAG, "%s model file not found: %s", label, path);
        return loaded;
    }
    loaded.engine = new ma::engine::EngineCVI();
    const size_t shared_memory = cvi_shared_memory_bytes();
    ma_err_t ret = loaded.engine->init(shared_memory);
    if (ret != MA_OK) {
        MA_LOGE(TAG, "%s engine init failed", label);
        return loaded;
    }
    ret = loaded.engine->load(path);
    if (ret != MA_OK) {
        MA_LOGE(TAG, "%s engine load model failed", label);
        return loaded;
    }
    loaded.model = ma::ModelFactory::create(loaded.engine, algorithm_id);
    if (loaded.model == nullptr) {
        MA_LOGE(TAG, "%s model not supported", label);
        return loaded;
    }
    if (loaded.model->getInputType() != MA_INPUT_TYPE_IMAGE) {
        MA_LOGE(TAG, "%s model input type not supported", label);
        ma::ModelFactory::remove(loaded.model);
        loaded.model = nullptr;
        return loaded;
    }
    const ma_img_t* model_input = static_cast<const ma_img_t*>(loaded.model->getInput());
    loaded.input_width = model_input->width;
    loaded.input_height = model_input->height;
    return loaded;
}

void release_model(LoadedModel& loaded) {
    if (loaded.model != nullptr) {
        ma::ModelFactory::remove(loaded.model);
        loaded.model = nullptr;
    }
    delete loaded.engine;
    loaded.engine = nullptr;
}

bool is_loaded(const LoadedModel& loaded) {
    return loaded.engine != nullptr && loaded.model != nullptr;
}

bool expected_rgb888_size(const ma_img_t& frame, size_t* expected_size) {
    if (frame.width == 0 || frame.height == 0) {
        return false;
    }

    *expected_size = static_cast<size_t>(frame.width) * static_cast<size_t>(frame.height) * 3;
    return frame.size >= *expected_size;
}

bool make_cpu_readable_frame(const ma_img_t& src, int target_width, int target_height, CpuReadableFrame* prepared) {
    if (prepared == nullptr) {
        return false;
    }
    if (src.format != MA_PIXEL_FORMAT_RGB888) {
        MA_LOGW(TAG, "unsupported frame format for CPU preprocess: %d", static_cast<int>(src.format));
        return false;
    }
    if (src.data == nullptr) {
        MA_LOGW(TAG, "frame data is null");
        return false;
    }

    size_t source_bytes = 0;
    if (!expected_rgb888_size(src, &source_bytes)) {
        MA_LOGW(TAG,
                "invalid RGB frame geometry width=%u height=%u size=%u",
                src.width,
                src.height,
                src.size);
        return false;
    }

    prepared->frame = src;

    uint8_t* cpu_ptr = src.data;
    if (src.physical) {
        prepared->mapped_size = src.size;
        prepared->mapped_ptr = CVI_SYS_Mmap(reinterpret_cast<uint64_t>(src.data), prepared->mapped_size);
        if (prepared->mapped_ptr == nullptr) {
            MA_LOGW(TAG, "CVI_SYS_Mmap failed for physical frame addr=%p size=%u", src.data, src.size);
            return false;
        }
        cpu_ptr = static_cast<uint8_t*>(prepared->mapped_ptr);
    }

    const bool needs_copy = src.physical ||
                            src.width != static_cast<uint16_t>(target_width) ||
                            src.height != static_cast<uint16_t>(target_height);
    if (!needs_copy) {
        prepared->frame.physical = false;
        prepared->frame.data = cpu_ptr;
        prepared->frame.size = static_cast<uint32_t>(source_bytes);
        return true;
    }

    ::cv::Mat src_mat(src.height, src.width, CV_8UC3, cpu_ptr);
    ::cv::Mat dst_mat;
    if (src.width != static_cast<uint16_t>(target_width) ||
        src.height != static_cast<uint16_t>(target_height)) {
        ::cv::resize(src_mat, dst_mat, ::cv::Size(target_width, target_height));
    } else {
        dst_mat = src_mat;
    }

    prepared->owned_buffer.assign(dst_mat.data, dst_mat.data + dst_mat.total() * dst_mat.elemSize());
    prepared->frame.data = prepared->owned_buffer.data();
    prepared->frame.size = static_cast<uint32_t>(prepared->owned_buffer.size());
    prepared->frame.width = static_cast<uint16_t>(target_width);
    prepared->frame.height = static_cast<uint16_t>(target_height);
    prepared->frame.physical = false;
    return true;
}

bool fill_lane_input_tensor(const ma_tensor_t& input_tensor, const ma_img_t& frame) {
    CpuReadableFrame prepared;
    if (!make_cpu_readable_frame(frame, input_tensor.shape.dims[3], input_tensor.shape.dims[2], &prepared)) {
        return false;
    }

    const int height = prepared.frame.height;
    const int width = prepared.frame.width;
    const bool is_nhwc = input_tensor.shape.size == 4 &&
                         (input_tensor.shape.dims[3] == 3 || input_tensor.shape.dims[3] == 1);
    const bool is_nchw = input_tensor.shape.size == 4 &&
                         (input_tensor.shape.dims[1] == 3 || input_tensor.shape.dims[1] == 1);
    if (!is_nhwc && !is_nchw) {
        MA_LOGW(TAG, "unsupported lane input shape");
        return false;
    }

    const uint8_t* src = prepared.frame.data;
    const float quant_scale = input_tensor.quant_param.scale > 0.0f ? input_tensor.quant_param.scale : 1.0f;
    const int zero_point = input_tensor.quant_param.zero_point;

    static bool dumped_lane_resized = false;
    if (!dumped_lane_resized) {
        const char* dump_path = std::getenv("DUMP_LANE_RESIZED_PATH");
        if (dump_path != nullptr && dump_path[0] != '\0') {
            const int dump_after_frames =
                env_nonneg_int("DUMP_LANE_RESIZED_AFTER_FRAMES", env_nonneg_int("DUMP_FRAME_AFTER_FRAMES", 0));
            if (g_current_frame_count >= dump_after_frames) {
                ::cv::Mat rgb(height, width, CV_8UC3, prepared.frame.data);
                dumped_lane_resized = true;
                dump_mat_to_path(rgb, dump_path, "lane_resized");
            }
        }
    }

    auto normalized_value = [&](int pixel_index, int channel) {
        return (static_cast<float>(src[pixel_index * 3 + channel]) - kLaneMean[channel]) * kLaneScale[channel];
    };

    if (input_tensor.type == MA_TENSOR_TYPE_F32) {
        float* dst = input_tensor.data.f32;
        if (is_nchw) {
            const int plane = width * height;
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const int pixel_index = y * width + x;
                    for (int c = 0; c < 3; ++c) {
                        dst[c * plane + pixel_index] = normalized_value(pixel_index, c);
                    }
                }
            }
        } else {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const int pixel_index = y * width + x;
                    const int base = pixel_index * 3;
                    for (int c = 0; c < 3; ++c) {
                        dst[base + c] = normalized_value(pixel_index, c);
                    }
                }
            }
        }
        return true;
    }

    if (input_tensor.type == MA_TENSOR_TYPE_S8) {
        int8_t* dst = input_tensor.data.s8;
        if (is_nchw) {
            const int plane = width * height;
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const int pixel_index = y * width + x;
                    for (int c = 0; c < 3; ++c) {
                        const float q = normalized_value(pixel_index, c) / quant_scale + zero_point;
                        dst[c * plane + pixel_index] = static_cast<int8_t>(std::clamp<int>(std::lround(q), -128, 127));
                    }
                }
            }
        } else {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const int pixel_index = y * width + x;
                    const int base = pixel_index * 3;
                    for (int c = 0; c < 3; ++c) {
                        const float q = normalized_value(pixel_index, c) / quant_scale + zero_point;
                        dst[base + c] = static_cast<int8_t>(std::clamp<int>(std::lround(q), -128, 127));
                    }
                }
            }
        }
        return true;
    }

    MA_LOGW(TAG, "unsupported lane tensor type=%d", static_cast<int>(input_tensor.type));
    return false;
}

bool build_lane_mask_from_output(LoadedModel& lane, std::vector<uint8_t>* lane_mask, int* mask_width, int* mask_height) {
    const ma_tensor_t output = lane.engine->getOutput(0);
    if (output.shape.size != 4 || output.shape.dims[0] != 1 || output.shape.dims[1] < 2) {
        MA_LOGW(TAG, "unexpected lane output shape");
        return false;
    }

    *mask_height = output.shape.dims[2];
    *mask_width = output.shape.dims[3];
    const int pixel_count = (*mask_width) * (*mask_height);
    lane_mask->assign(pixel_count, 0);
    auto class1_probability = [](float class0, float class1) {
        const float max_logit = std::max(class0, class1);
        const float exp0 = std::exp(class0 - max_logit);
        const float exp1 = std::exp(class1 - max_logit);
        return exp1 / (exp0 + exp1);
    };

    int lane_pixels = 0;
    if (output.type == MA_TENSOR_TYPE_F32) {
        const float* data = output.data.f32;
        const float* class0 = data;
        const float* class1 = data + pixel_count;
        for (int i = 0; i < pixel_count; ++i) {
            if (class1_probability(class0[i], class1[i]) > kLaneMaskThreshold) {
                (*lane_mask)[i] = 255;
                ++lane_pixels;
            }
        }
    } else if (output.type == MA_TENSOR_TYPE_S8) {
        const int8_t* data = output.data.s8;
        const int8_t* class0 = data;
        const int8_t* class1 = data + pixel_count;
        const float quant_scale = output.quant_param.scale > 0.0f ? output.quant_param.scale : 1.0f;
        const int zero_point = output.quant_param.zero_point;
        for (int i = 0; i < pixel_count; ++i) {
            const float class0_value = (static_cast<int>(class0[i]) - zero_point) * quant_scale;
            const float class1_value = (static_cast<int>(class1[i]) - zero_point) * quant_scale;
            if (class1_probability(class0_value, class1_value) > kLaneMaskThreshold) {
                (*lane_mask)[i] = 255;
                ++lane_pixels;
            }
        }
    } else {
        MA_LOGW(TAG, "unsupported lane output type=%d", static_cast<int>(output.type));
        return false;
    }

    return lane_pixels > 0;
}

bool dump_rgb_frame_to_path(const ma_img_t& frame, const char* dump_path, const char* label) {
    CpuReadableFrame prepared;
    if (!make_cpu_readable_frame(frame, frame.width, frame.height, &prepared)) {
        MA_LOGW(TAG, "%s dump prepare failed", label);
        return false;
    }

    ::cv::Mat rgb(prepared.frame.height, prepared.frame.width, CV_8UC3, prepared.frame.data);
    ::cv::Mat bgr;
    ::cv::cvtColor(rgb, bgr, ::cv::COLOR_RGB2BGR);
    if (!::cv::imwrite(dump_path, bgr)) {
        MA_LOGW(TAG, "failed to dump %s to %s", label, dump_path);
        return false;
    }

    std::filesystem::path base_path(dump_path);
    const std::string raw_bgr_path =
        (base_path.parent_path() / (base_path.stem().string() + "_raw_bgr" + base_path.extension().string())).string();
    if (!::cv::imwrite(raw_bgr_path, rgb)) {
        MA_LOGW(TAG, "failed to dump %s raw-bgr view to %s", label, raw_bgr_path.c_str());
    }
    return true;
}

bool dump_mat_to_path(const ::cv::Mat& rgb, const char* dump_path, const char* label) {
    if (rgb.empty() || rgb.type() != CV_8UC3) {
        MA_LOGW(TAG, "%s dump mat invalid", label);
        return false;
    }

    ::cv::Mat bgr;
    ::cv::cvtColor(rgb, bgr, ::cv::COLOR_RGB2BGR);
    if (!::cv::imwrite(dump_path, bgr)) {
        MA_LOGW(TAG, "failed to dump %s to %s", label, dump_path);
        return false;
    }

    std::filesystem::path base_path(dump_path);
    const std::string raw_bgr_path =
        (base_path.parent_path() / (base_path.stem().string() + "_raw_bgr" + base_path.extension().string())).string();
    if (!::cv::imwrite(raw_bgr_path, rgb)) {
        MA_LOGW(TAG, "failed to dump %s raw-bgr view to %s", label, raw_bgr_path.c_str());
    }
    return true;
}

void dump_frame_once_if_requested(const ma_img_t& frame, int frame_count) {
    static bool attempted = false;
    if (attempted) {
        return;
    }

    const char* dump_path = std::getenv("DUMP_FRAME_PATH");
    if (dump_path == nullptr || dump_path[0] == '\0') {
        return;
    }

    const int dump_after_frames = env_nonneg_int("DUMP_FRAME_AFTER_FRAMES", 0);
    if (frame_count < dump_after_frames) {
        return;
    }

    attempted = true;
    dump_rgb_frame_to_path(frame, dump_path, "lane_frame");
}

void dump_stop_frame_once_if_requested(const ma_img_t& frame, int frame_count) {
    static bool attempted = false;
    if (attempted) {
        return;
    }

    const char* dump_path = std::getenv("DUMP_STOP_FRAME_PATH");
    if (dump_path == nullptr || dump_path[0] == '\0') {
        return;
    }

    const int dump_after_frames = env_nonneg_int("DUMP_STOP_FRAME_AFTER_FRAMES", 0);
    if (frame_count < dump_after_frames) {
        return;
    }

    attempted = true;
    dump_rgb_frame_to_path(frame, dump_path, "stop_frame");
}

void dump_lane_mask_once_if_requested(const std::vector<uint8_t>& lane_mask, int width, int height) {
    static bool attempted = false;
    if (attempted) {
        return;
    }

    const char* dump_path = std::getenv("DUMP_LANE_MASK_PATH");
    if (dump_path == nullptr || dump_path[0] == '\0') {
        return;
    }

    const int dump_after_frames =
        env_nonneg_int("DUMP_LANE_MASK_AFTER_FRAMES", env_nonneg_int("DUMP_FRAME_AFTER_FRAMES", 0));
    if (g_current_frame_count < dump_after_frames) {
        return;
    }

    attempted = true;
    ::cv::Mat mask(height, width, CV_8UC1, const_cast<uint8_t*>(lane_mask.data()));
    if (!::cv::imwrite(dump_path, mask)) {
        MA_LOGW(TAG, "failed to dump lane_mask to %s", dump_path);
        return;
    }

}

void dump_lane_overlay_once_if_requested(const ma_img_t& frame,
                                         const std::vector<uint8_t>& lane_mask,
                                         int width,
                                         int height) {
    static bool attempted = false;
    if (attempted) {
        return;
    }

    const char* dump_path = std::getenv("DUMP_LANE_OVERLAY_PATH");
    if (dump_path == nullptr || dump_path[0] == '\0') {
        return;
    }

    const int dump_after_frames =
        env_nonneg_int("DUMP_LANE_OVERLAY_AFTER_FRAMES", env_nonneg_int("DUMP_FRAME_AFTER_FRAMES", 0));
    if (g_current_frame_count < dump_after_frames) {
        return;
    }

    CpuReadableFrame prepared;
    if (!make_cpu_readable_frame(frame, width, height, &prepared)) {
        MA_LOGW(TAG, "lane_overlay dump prepare failed");
        return;
    }

    attempted = true;
    ::cv::Mat overlay(height, width, CV_8UC3, prepared.frame.data);
    overlay = overlay.clone();
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (lane_mask[y * width + x] == 0) {
                continue;
            }
            ::cv::Vec3b& px = overlay.at<::cv::Vec3b>(y, x);
            px[0] = static_cast<uint8_t>(std::min(255, static_cast<int>(px[0]) / 2));
            px[1] = static_cast<uint8_t>(std::min(255, static_cast<int>(px[1]) / 2 + 127));
            px[2] = static_cast<uint8_t>(std::min(255, static_cast<int>(px[2]) / 2));
        }
    }

    dump_mat_to_path(overlay, dump_path, "lane_overlay");
}

void dump_encoded_frame_once_if_requested(const ma_img_t& frame) {
    static bool attempted = false;
    if (attempted) {
        return;
    }

    const char* dump_path = std::getenv("DUMP_ENCODED_FRAME_PATH");
    if (dump_path == nullptr || dump_path[0] == '\0') {
        return;
    }

    attempted = true;
    if (frame.data == nullptr || frame.size == 0) {
        MA_LOGW(TAG, "encoded frame dump skipped: empty frame");
        return;
    }

    FILE* fp = std::fopen(dump_path, "wb");
    if (fp == nullptr) {
        MA_LOGW(TAG, "failed to open encoded dump path %s", dump_path);
        return;
    }
    const size_t written = std::fwrite(frame.data, 1, frame.size, fp);
    std::fclose(fp);
    if (written != frame.size) {
        MA_LOGW(TAG, "failed to write full encoded frame to %s", dump_path);
        return;
    }

}

bool mask_get(const std::vector<uint8_t>& mask, int width, int x, int y) {
    return mask[y * width + x] != 0;
}

int count_mask_pixels(const ma_segm2f_t& segment) {
    const int pixel_count = segment.mask.width * segment.mask.height;
    int count = 0;
    for (int i = 0; i < pixel_count; ++i) {
        const int byte_index = i / 8;
        const int bit_offset = i % 8;
        if (segment.mask.data[byte_index] & (1 << bit_offset)) {
            ++count;
        }
    }
    return count;
}

bool is_valid_lane_row(const std::vector<uint8_t>& mask, int width, int y, int* lx, int* rx);

struct LaneMaskStats {
    int total_pixels = 0;
    int lane_pixels = 0;
    int bottom_half_pixels = 0;
    int valid_rows = 0;
    int first_valid_y = -1;
};

struct LaneEvidence {
    bool barricaded = false;
    int max_consecutive_valid_rows = 0;
    int lower_band_valid_rows = 0;
    int lower_band_lane_rows = 0;
    int boundary_rows = 0;
};

struct SingleLineCandidate {
    bool detected = false;
    int lane_rows = 0;
    int single_run_rows = 0;
    int multi_run_rows = 0;
    int centered_single_run_rows = 0;
    int wide_single_run_rows = 0;
};

struct RaycastDecision {
    bool has_preference = false;
    bool blocked = false;
    int left_score = 0;
    int right_score = 0;
};

struct LaneCenterCandidate {
    int lx = -1;
    int rx = -1;
    int start_y = -1;
    int row_count = 0;
    float center_x = 0.0f;
    float center_y = 0.0f;
};

LaneCenterCandidate make_lane_center_candidate(int lx, int rx, int start_y, int row_count) {
    LaneCenterCandidate candidate;
    candidate.lx = lx;
    candidate.rx = rx;
    candidate.start_y = start_y;
    candidate.row_count = row_count;
    candidate.center_x = (lx + rx) / 2.0f;
    candidate.center_y = static_cast<float>(start_y);
    return candidate;
}

float lane_center_history_distance_sq(const LaneCenterCandidate& candidate) {
    if (g_lane_center_history.empty()) {
        return std::numeric_limits<float>::infinity();
    }

    float best_distance_sq = std::numeric_limits<float>::infinity();
    for (const LaneCenterPoint& point : g_lane_center_history) {
        const float dx = candidate.center_x - point.x;
        const float dy = candidate.center_y - point.y;
        const float distance_sq = dx * dx + dy * dy;
        if (distance_sq < best_distance_sq) {
            best_distance_sq = distance_sq;
        }
    }
    return best_distance_sq;
}

void remember_lane_center(float center_x, float center_y) {
    if (g_lane_center_history.size() >= kLaneCenterHistorySize) {
        g_lane_center_history.erase(g_lane_center_history.begin());
    }
    g_lane_center_history.push_back({center_x, center_y});
}

bool looks_like_lane_signal_loss(const LaneMaskStats& stats,
                                 const LaneEvidence& evidence,
                                 int width,
                                 int height,
                                 const LaneDecision& decision) {
    if (decision.confident || evidence.barricaded || g_frames_since_confident_lane > kLaneSignalLossHoldFrames) {
        return false;
    }

    const int min_partial_pixels = std::max(6, width * height / 120);
    const int min_partial_rows = std::max(3, height / 10);
    const int min_boundary_rows = std::max(3, height / 12);
    const bool sparse_but_present = stats.lane_pixels >= min_partial_pixels;
    const bool fragmented_near_car =
        evidence.lower_band_lane_rows >= min_partial_rows && evidence.lower_band_valid_rows == 0;
    const bool edge_only = evidence.boundary_rows >= min_boundary_rows;
    const bool upper_only = stats.first_valid_y >= 0 && stats.first_valid_y < height / 2 &&
                            evidence.lower_band_valid_rows == 0 && evidence.lower_band_lane_rows <= min_partial_rows;
    const bool abrupt_dropout = stats.lane_pixels == 0 && g_frames_since_confident_lane <= 2;
    return abrupt_dropout || (sparse_but_present && (fragmented_near_car || edge_only || upper_only));
}

void apply_signal_loss_recovery(const LaneMaskStats& stats,
                                const LaneEvidence& evidence,
                                int width,
                                int height,
                                LaneDecision* decision) {
    if (decision == nullptr) {
        return;
    }
    if (std::strcmp(decision->status, "SEARCHING") != 0 && std::strcmp(decision->status, "LOST") != 0) {
        return;
    }
    if (!looks_like_lane_signal_loss(stats, evidence, width, height, *decision)) {
        return;
    }

    const float remembered_steer =
        std::clamp(g_last_confident_steering_percent * kLaneSignalLossRememberScale,
                   -kLaneSignalLossMaxRememberedSteerPercent,
                   kLaneSignalLossMaxRememberedSteerPercent);
    if (std::fabs(remembered_steer) < 1.0f) {
        return;
    }

    decision->steering_percent = remembered_steer;
    decision->status = "SIGNAL_LOSS_HOLD";
    decision->weight = 0;
}

SingleLineCandidate analyze_single_line_candidate(const std::vector<uint8_t>& mask, int width, int height) {
    SingleLineCandidate candidate;
    const int y0 = static_cast<int>(height * 0.4f);
    const int min_run_width = std::max(2, static_cast<int>(width * 8.0f / 512.0f));
    const int wide_run_width = std::max(min_run_width, static_cast<int>(width * kSingleLineWideRunPercent));
    const float left_central_limit = width * kSingleLineCentralZonePercent;
    const float right_central_limit = width * (1.0f - kSingleLineCentralZonePercent);
    for (int y = y0; y < height; ++y) {
        int filtered_run_count = 0;
        int longest_run = 0;
        int longest_run_start = -1;
        int run_start = -1;
        for (int x = 0; x < width; ++x) {
            if (mask_get(mask, width, x, y)) {
                if (run_start < 0) {
                    run_start = x;
                }
                continue;
            }
            if (run_start >= 0) {
                const int run_width = x - run_start;
                if (run_width >= min_run_width) {
                    ++filtered_run_count;
                    if (run_width > longest_run) {
                        longest_run = run_width;
                        longest_run_start = run_start;
                    }
                }
                run_start = -1;
            }
        }
        if (run_start >= 0) {
            const int run_width = width - run_start;
            if (run_width >= min_run_width) {
                ++filtered_run_count;
                if (run_width > longest_run) {
                    longest_run = run_width;
                    longest_run_start = run_start;
                }
            }
        }
        if (filtered_run_count == 0) {
            continue;
        }

        ++candidate.lane_rows;
        if (filtered_run_count == 1) {
            ++candidate.single_run_rows;
            const float run_center = longest_run_start + longest_run / 2.0f;
            if (run_center >= left_central_limit && run_center <= right_central_limit) {
                ++candidate.centered_single_run_rows;
            }
            if (longest_run >= wide_run_width) {
                ++candidate.wide_single_run_rows;
            }
        } else {
            ++candidate.multi_run_rows;
        }
    }

    const int min_single_rows = std::max(3, static_cast<int>((height - y0) * 0.08f));
    candidate.detected = candidate.lane_rows >= min_single_rows &&
                         candidate.single_run_rows >= min_single_rows &&
                         candidate.single_run_rows > candidate.multi_run_rows &&
                         candidate.centered_single_run_rows >= std::max(2, min_single_rows - 1) &&
                         candidate.wide_single_run_rows >= std::max(2, min_single_rows - 1);
    return candidate;
}

int trace_raycast_clearance(const std::vector<uint8_t>& mask,
                            int width,
                            int height,
                            int start_x,
                            int start_y,
                            int end_x,
                            int end_y) {
    const int dx = end_x - start_x;
    const int dy = end_y - start_y;
    const int steps = std::max(std::abs(dx), std::abs(dy));
    if (steps <= 0) {
        return 0;
    }

    for (int step = 1; step <= steps; ++step) {
        const float t = static_cast<float>(step) / static_cast<float>(steps);
        const int x = std::clamp(static_cast<int>(std::lround(start_x + dx * t)), 0, width - 1);
        const int y = std::clamp(static_cast<int>(std::lround(start_y + dy * t)), 0, height - 1);
        if (mask_get(mask, width, x, y)) {
            return step;
        }
    }

    return steps;
}

RaycastDecision choose_raycast_turn(const std::vector<uint8_t>& mask, int width, int height) {
    RaycastDecision decision;
    const int start_x = width / 2;
    const int start_y = height - 2;
    const int top_y = static_cast<int>(height * 0.35f);
    const int side_y = static_cast<int>(height * 0.55f);
    const int left_inner_x = static_cast<int>(width * 0.18f);
    const int right_inner_x = static_cast<int>(width * 0.82f);

    for (int i = 0; i < kRaycastRaysPerSide; ++i) {
        const float blend = (kRaycastRaysPerSide == 1) ? 0.0f : static_cast<float>(i) / (kRaycastRaysPerSide - 1);
        const int left_target_x = static_cast<int>(std::lround(left_inner_x * (1.0f - blend)));
        const int right_target_x =
            static_cast<int>(std::lround((width - 1) * blend + right_inner_x * (1.0f - blend)));
        const int target_y = static_cast<int>(std::lround(top_y * blend + side_y * (1.0f - blend)));
        decision.left_score += trace_raycast_clearance(mask, width, height, start_x, start_y, left_target_x, target_y);
        decision.right_score +=
            trace_raycast_clearance(mask, width, height, start_x, start_y, right_target_x, target_y);
    }

    const int best_score = std::max(decision.left_score, decision.right_score);
    const int score_margin = std::abs(decision.left_score - decision.right_score);
    const int min_margin = std::max(2, static_cast<int>(best_score * kRaycastMinScoreMarginPercent));
    decision.blocked = best_score <= kRaycastRaysPerSide * 3;
    decision.has_preference = !decision.blocked && score_margin >= min_margin;
    return decision;
}

LaneMaskStats analyze_lane_mask(const std::vector<uint8_t>& mask, int width, int height) {
    LaneMaskStats stats;
    stats.total_pixels = width * height;

    for (int y = 0; y < height; ++y) {
        bool row_valid = false;
        int lx = -1;
        int rx = -1;
        if (is_valid_lane_row(mask, width, y, &lx, &rx)) {
            row_valid = true;
        }

        for (int x = 0; x < width; ++x) {
            if (!mask_get(mask, width, x, y)) {
                continue;
            }
            ++stats.lane_pixels;
            if (y >= height / 2) {
                ++stats.bottom_half_pixels;
            }
        }

        if (row_valid) {
            ++stats.valid_rows;
            if (stats.first_valid_y < 0) {
                stats.first_valid_y = y;
            }
        }
    }

    return stats;
}

LaneEvidence analyze_lane_evidence(const std::vector<uint8_t>& mask, int width, int height) {
    LaneEvidence evidence;
    const int wall_y = std::min(height - 1, static_cast<int>(height * 0.70f));
    int wall_pixels = 0;
    for (int x = 0; x < width; ++x) {
        if (mask_get(mask, width, x, wall_y)) {
            ++wall_pixels;
        }
    }
    evidence.barricaded = wall_pixels > static_cast<int>(width * kBarricadeThreshold);

    const int y0 = static_cast<int>(height * 0.4f);
    const int min_boundary_run_width = std::max(2, static_cast<int>(width * 10.0f / 512.0f));
    const float left_boundary_limit = width * kLaneWeightBoundaryZonePercent;
    const float right_boundary_limit = width * (1.0f - kLaneWeightBoundaryZonePercent);
    int consecutive_valid_rows = 0;

    for (int y = y0; y < height; ++y) {
        int lane_pixels_in_row = 0;
        int longest_run = 0;
        int longest_run_start = -1;
        int current_run_start = -1;

        for (int x = 0; x < width; ++x) {
            if (mask_get(mask, width, x, y)) {
                ++lane_pixels_in_row;
                if (current_run_start < 0) {
                    current_run_start = x;
                }
                continue;
            }
            if (current_run_start >= 0) {
                const int run_width = x - current_run_start;
                if (run_width > longest_run) {
                    longest_run = run_width;
                    longest_run_start = current_run_start;
                }
                current_run_start = -1;
            }
        }
        if (current_run_start >= 0) {
            const int run_width = width - current_run_start;
            if (run_width > longest_run) {
                longest_run = run_width;
                longest_run_start = current_run_start;
            }
        }

        if (lane_pixels_in_row > 0) {
            ++evidence.lower_band_lane_rows;
        }

        int lx = -1;
        int rx = -1;
        if (is_valid_lane_row(mask, width, y, &lx, &rx)) {
            ++evidence.lower_band_valid_rows;
            ++consecutive_valid_rows;
            evidence.max_consecutive_valid_rows = std::max(evidence.max_consecutive_valid_rows, consecutive_valid_rows);
            continue;
        }

        consecutive_valid_rows = 0;
        if (longest_run < min_boundary_run_width || longest_run_start < 0) {
            continue;
        }

        const float run_center = longest_run_start + longest_run / 2.0f;
        if (run_center <= left_boundary_limit || run_center >= right_boundary_limit) {
            ++evidence.boundary_rows;
        }
    }

    return evidence;
}

void dilate_5x5(const std::vector<uint8_t>& src, int width, int height, std::vector<uint8_t>& dst) {
    std::fill(dst.begin(), dst.end(), 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bool any = false;
            for (int ky = -2; ky <= 2 && !any; ++ky) {
                const int yy = y + ky;
                if (yy < 0 || yy >= height) {
                    continue;
                }
                for (int kx = -2; kx <= 2; ++kx) {
                    const int xx = x + kx;
                    if (xx >= 0 && xx < width && mask_get(src, width, xx, yy)) {
                        any = true;
                        break;
                    }
                }
            }
            dst[y * width + x] = any ? 255 : 0;
        }
    }
}

void erode_5x5(const std::vector<uint8_t>& src, int width, int height, std::vector<uint8_t>& dst) {
    std::fill(dst.begin(), dst.end(), 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bool all = true;
            for (int ky = -2; ky <= 2 && all; ++ky) {
                const int yy = y + ky;
                if (yy < 0 || yy >= height) {
                    all = false;
                    break;
                }
                for (int kx = -2; kx <= 2; ++kx) {
                    const int xx = x + kx;
                    if (xx < 0 || xx >= width || !mask_get(src, width, xx, yy)) {
                        all = false;
                        break;
                    }
                }
            }
            dst[y * width + x] = all ? 255 : 0;
        }
    }
}

void close_5x5(std::vector<uint8_t>& mask, int width, int height) {
    std::vector<uint8_t> tmp(mask.size());
    dilate_5x5(mask, width, height, tmp);
    mask.swap(tmp);
}

bool is_valid_lane_row(const std::vector<uint8_t>& mask, int width, int y, int* lx, int* rx) {
    struct Run {
        int start = 0;
        int end = 0;
    };

    std::vector<Run> runs;
    int run_start = -1;
    for (int x = 0; x < width; ++x) {
        if (mask_get(mask, width, x, y)) {
            if (run_start < 0) {
                run_start = x;
            }
            continue;
        }
        if (run_start >= 0) {
            runs.push_back({run_start, x - 1});
            run_start = -1;
        }
    }
    if (run_start >= 0) {
        runs.push_back({run_start, width - 1});
    }

    const int min_run_width = std::max(2, static_cast<int>(width * 8.0f / 512.0f));
    std::vector<Run> filtered_runs;
    filtered_runs.reserve(runs.size());
    for (const Run& run : runs) {
        if (run.end - run.start + 1 >= min_run_width) {
            filtered_runs.push_back(run);
        }
    }

    if (filtered_runs.size() < 2) {
        return false;
    }

    const int min_gap = std::max(20, static_cast<int>(width * 140.0f / 512.0f));
    const int max_gap = std::max(min_gap + 1, static_cast<int>(width * 470.0f / 512.0f));
    const Run& left_run = filtered_runs.front();
    const Run& right_run = filtered_runs.back();
    if (left_run.end >= right_run.start) {
        return false;
    }

    const int inner_left = left_run.end;
    const int inner_right = right_run.start;
    const int gap = inner_right - inner_left;
    if (gap <= min_gap || gap >= max_gap) {
        return false;
    }

    const int mid = (inner_left + inner_right) / 2;
    if (mask_get(mask, width, mid, y)) {
        return false;
    }

    *lx = inner_left;
    *rx = inner_right;
    return true;
}

LaneDecision decide_lane(const std::vector<uint8_t>& mask, int width, int height) {
    LaneDecision decision;
    const int mid_x = width / 2;
    const int wall_y = std::min(height - 1, static_cast<int>(height * 0.70f));
    const int min_centered_rows = min_centered_lane_rows(height);

    int wall_pixels = 0;
    for (int x = 0; x < width; ++x) {
        if (mask_get(mask, width, x, wall_y)) {
            ++wall_pixels;
        }
    }
    const bool barricaded = wall_pixels > static_cast<int>(width * kBarricadeThreshold);

    int lx = -1;
    int rx = -1;
    int found_y = -1;
    std::vector<LaneCenterCandidate> center_candidates;
    const int sweep_start = std::max(0, height - 15);
    const int sweep_end = std::min(height - 1, static_cast<int>(height * kLaneSweepEndAt512 / 512.0f));
    int current_count = 0;
    int current_sum_lx = 0;
    int current_sum_rx = 0;
    int current_start_y = -1;
    for (int y = sweep_start; y > sweep_end; --y) {
        int cur_lx = -1;
        int cur_rx = -1;
        const bool valid = is_valid_lane_row(mask, width, y, &cur_lx, &cur_rx);
        if (!valid) {
            if (current_count >= min_centered_rows) {
                const int candidate_lx = current_sum_lx / current_count;
                const int candidate_rx = current_sum_rx / current_count;
                center_candidates.push_back(
                    make_lane_center_candidate(candidate_lx, candidate_rx, current_start_y, current_count));
            }
            current_count = 0;
            current_sum_lx = 0;
            current_sum_rx = 0;
            current_start_y = -1;
            continue;
        }
        if (current_count == 0) {
            current_start_y = y;
        }
        current_sum_lx += cur_lx;
        current_sum_rx += cur_rx;
        ++current_count;
    }
    if (current_count >= min_centered_rows) {
        const int candidate_lx = current_sum_lx / current_count;
        const int candidate_rx = current_sum_rx / current_count;
        center_candidates.push_back(
            make_lane_center_candidate(candidate_lx, candidate_rx, current_start_y, current_count));
    }

    if (!center_candidates.empty()) {
        size_t best_index = 0;
        if (!g_lane_center_history.empty()) {
            float best_distance_sq = lane_center_history_distance_sq(center_candidates.front());
            for (size_t i = 1; i < center_candidates.size(); ++i) {
                const float distance_sq = lane_center_history_distance_sq(center_candidates[i]);
                if (distance_sq < best_distance_sq ||
                    (distance_sq == best_distance_sq &&
                     center_candidates[i].row_count > center_candidates[best_index].row_count)) {
                    best_distance_sq = distance_sq;
                    best_index = i;
                }
            }
        }

        const LaneCenterCandidate& best_candidate = center_candidates[best_index];
        lx = best_candidate.lx;
        rx = best_candidate.rx;
        found_y = best_candidate.start_y;
        g_last_known_lane_width = rx - lx;
    }

    float target_x = static_cast<float>(mid_x);
    if (barricaded) {
        const RaycastDecision raycast = choose_raycast_turn(mask, width, height);
        if (raycast.left_score > raycast.right_score) {
            decision.steering_percent = -100.0f;
            decision.status = "BARRICADE_LEFT";
        } else if (raycast.right_score > raycast.left_score) {
            decision.steering_percent = 100.0f;
            decision.status = "BARRICADE_RIGHT";
        } else {
            decision.steering_percent = 100.0f;
            decision.status = "BARRICADE_EVASIVE_RIGHT";
        }
        decision.confident = true;
        return decision;
    }

    if (lx >= 0 && rx >= 0 && found_y >= 0) {
        target_x = static_cast<float>((lx + rx) / 2);
        remember_lane_center(target_x, static_cast<float>(found_y));
        decision.status = "CENTERED";
        decision.confident = true;
    } else {
        long long sum_x = 0;
        int count = 0;
        const int y0 = static_cast<int>(height * 0.4f);
        for (int y = y0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (mask_get(mask, width, x, y)) {
                    sum_x += x;
                    ++count;
                }
            }
        }

        const SingleLineCandidate single_line = analyze_single_line_candidate(mask, width, height);
        if (single_line.detected) {
            const RaycastDecision raycast = choose_raycast_turn(mask, width, height);
            if (raycast.has_preference) {
                decision.steering_percent = (raycast.left_score > raycast.right_score) ? -100.0f : 100.0f;
                decision.status = (raycast.left_score > raycast.right_score) ? "RAYCAST_LEFT" : "RAYCAST_RIGHT";
                decision.confident = true;
                return decision;
            }
            if (raycast.blocked) {
                decision.steering_percent = 0.0f;
                decision.status = "RAYCAST_BLOCKED";
                return decision;
            }
        }

        if (count > 0) {
            const float avg_x = static_cast<float>(sum_x) / static_cast<float>(count);
            const int default_lane_width = std::max(1, static_cast<int>(width * kDefaultLaneWidthAt512 / 512.0f));
            const int fallback_lane_width =
                (g_last_known_lane_width > 0 && g_last_known_lane_width < width) ? g_last_known_lane_width
                                                                                  : default_lane_width;
            float target_from_boundary = avg_x;
            if (avg_x < mid_x) {
                target_from_boundary =
                    std::min(static_cast<float>(width - 1), avg_x + fallback_lane_width * kLaneBoundaryFallbackCenterFactor);
                decision.status = "LEFT_BOUNDARY_TRACK";
            } else {
                target_from_boundary =
                    std::max(0.0f, avg_x - fallback_lane_width * kLaneBoundaryFallbackCenterFactor);
                decision.status = "RIGHT_BOUNDARY_TRACK";
            }

            const float denom = std::max(1.0f, static_cast<float>(mid_x));
            const float deviation = ((target_from_boundary - mid_x) / denom) * 100.0f;
            if (std::fabs(deviation) <= kLaneFallbackDeadbandPercent) {
                decision.steering_percent = 0.0f;
                decision.status = "SEARCHING";
                return decision;
            }

            decision.steering_percent =
                std::clamp(deviation * kLaneFallbackScale, -kLaneFallbackMaxSteerPercent, kLaneFallbackMaxSteerPercent);
            decision.confident = true;
            return decision;
        }

        decision.steering_percent = 0.0f;
        decision.status = "LOST";
        return decision;
    }

    const float denom = std::max(1.0f, static_cast<float>(mid_x));
    decision.steering_percent = std::clamp(((target_x - mid_x) / denom) * 100.0f, -100.0f, 100.0f);
    return decision;
}

void unpack_segment_mask(const ma_segm2f_t& segment, std::vector<uint8_t>& mask, int* width, int* height) {
    *width = segment.mask.width;
    *height = segment.mask.height;
    mask.assign((*width) * (*height), 0);

    for (int y = 0; y < *height; ++y) {
        for (int x = 0; x < *width; ++x) {
            const int bit_index = y * (*width) + x;
            const int byte_index = bit_index / 8;
            const int bit_offset = bit_index % 8;
            if (segment.mask.data[byte_index] & (1 << bit_offset)) {
                mask[bit_index] = 255;
            }
        }
    }
}

bool run_lane_model(LoadedModel& lane, ma_img_t& frame, LaneDecision* decision) {
    const ma_output_type_t output_type = lane.model->getOutputType();
    if (output_type != MA_OUTPUT_TYPE_SEGMENT) {
        MA_LOGW(TAG, "Lane model output type must be MA_OUTPUT_TYPE_SEGMENT");
        return false;
    }

    const ma_tensor_t input_tensor = lane.engine->getInput(0);
    if (!fill_lane_input_tensor(input_tensor, frame)) {
        MA_LOGW(TAG, "Lane model input preprocess failed");
        return false;
    }

    if (lane.engine->run() != MA_OK) {
        MA_LOGW(TAG, "Lane model engine run failed");
        return false;
    }

    std::vector<uint8_t> lane_mask;
    int mask_width = 0;
    int mask_height = 0;
    if (!build_lane_mask_from_output(lane, &lane_mask, &mask_width, &mask_height)) {
        *decision = LaneDecision{};
        printf("[LANE_TICK] decision=%s steer=%+.1f weight=%d no_lane_mask\n",
               decision->status,
               decision->steering_percent,
               decision->weight);
        std::fflush(stdout);
        return true;
    }

    close_5x5(lane_mask, mask_width, mask_height);
    dump_lane_mask_once_if_requested(lane_mask, mask_width, mask_height);
    dump_lane_overlay_once_if_requested(frame, lane_mask, mask_width, mask_height);
    const LaneMaskStats stats = analyze_lane_mask(lane_mask, mask_width, mask_height);
    const LaneEvidence evidence = analyze_lane_evidence(lane_mask, mask_width, mask_height);
    *decision = decide_lane(lane_mask, mask_width, mask_height);
    apply_signal_loss_recovery(stats, evidence, mask_width, mask_height, decision);
    decision->weight = decision->confident ? 1 : 0;
    if (decision->confident) {
        g_last_confident_steering_percent = decision->steering_percent;
        g_frames_since_confident_lane = 0;
    } else if (g_frames_since_confident_lane < 1000000) {
        ++g_frames_since_confident_lane;
    }
    printf("[LANE_TICK] mask=%dx%d lane_pixels=%d/%d bottom=%d valid_rows=%d first_valid_y=%d decision=%s steer=%+.1f weight=%d barricade=%d max_valid_run=%d boundary_rows=%d lane_rows=%d\n",
           mask_width,
           mask_height,
           stats.lane_pixels,
           stats.total_pixels,
           stats.bottom_half_pixels,
           stats.valid_rows,
           stats.first_valid_y,
           decision->status,
           decision->steering_percent,
           decision->weight,
           evidence.barricaded ? 1 : 0,
           evidence.max_consecutive_valid_rows,
           evidence.boundary_rows,
           evidence.lower_band_lane_rows);
    std::fflush(stdout);
    return true;
}

bool run_stop_model(LoadedModel& stop, ma_img_t& frame, float threshold, bool* should_stop, int frame_count) {
    *should_stop = false;
    if (stop.model->getOutputType() != MA_OUTPUT_TYPE_BBOX) {
        MA_LOGW(TAG, "Stop model output type must be MA_OUTPUT_TYPE_BBOX");
        return false;
    }

    CpuReadableFrame prepared;
    if (!make_cpu_readable_frame(frame, stop.input_width, stop.input_height, &prepared)) {
        MA_LOGW(TAG, "Stop model frame prepare failed");
        return false;
    }
    dump_stop_frame_once_if_requested(prepared.frame, frame_count);

    ma::model::Detector* detector = static_cast<ma::model::Detector*>(stop.model);
    detector->setConfig(MA_MODEL_CFG_OPT_THRESHOLD, threshold);
    if (detector->run(&prepared.frame) != MA_OK) {
        MA_LOGW(TAG, "Stop model inference failed");
        return false;
    }

    auto results = detector->getResults();
    int detection_count = 0;
    for (auto result : results) {
        ++detection_count;
        if (result.target != kStopSignClassId || result.h <= 0.0f) {
            continue;
        }

        const float distance_cm = kStopSignDistanceConstant / result.h;
        printf("[STOP_TICK] detections=%d stop_distance_cm=%.1f\n", detection_count, distance_cm);
        std::fflush(stdout);
        if (distance_cm <= kStopDistanceCm) {
            *should_stop = true;
            printf("[STOP_TICK] detections=%d should_stop=1 stop_distance_cm=%.1f\n", detection_count, distance_cm);
            std::fflush(stdout);
            return true;
        }
    }

    printf("[STOP_TICK] detections=%d should_stop=0\n", detection_count);
    std::fflush(stdout);
    return true;
}

bool decode_jpeg_frame(const ma_img_t& jpeg_frame, std::vector<uint8_t>* rgb_buffer, ma_img_t* rgb_frame) {
    std::vector<uint8_t> encoded(jpeg_frame.data, jpeg_frame.data + jpeg_frame.size);
    ::cv::Mat decoded = ::cv::imdecode(encoded, ::cv::IMREAD_COLOR);
    if (decoded.empty()) {
        MA_LOGE(TAG, "JPEG decode failed");
        return false;
    }

    ::cv::cvtColor(decoded, decoded, ::cv::COLOR_BGR2RGB);
    rgb_buffer->assign(decoded.data, decoded.data + decoded.total() * decoded.elemSize());

    rgb_frame->size = static_cast<uint32_t>(rgb_buffer->size());
    rgb_frame->width = static_cast<uint16_t>(decoded.cols);
    rgb_frame->height = static_cast<uint16_t>(decoded.rows);
    rgb_frame->format = MA_PIXEL_FORMAT_RGB888;
    rgb_frame->rotate = MA_PIXEL_ROTATE_0;
    rgb_frame->timestamp = jpeg_frame.timestamp;
    rgb_frame->key = jpeg_frame.key;
    rgb_frame->index = jpeg_frame.index;
    rgb_frame->count = jpeg_frame.count;
    rgb_frame->physical = false;
    rgb_frame->data = rgb_buffer->data();
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage:\n");
        printf("   %s stop_sign.cvimodel lane.cvimodel [stop_threshold] [lane_tick] [stop_tick] [uart_device]\n", argv[0]);
        printf("   Use '-' as stop_sign.cvimodel to run lane-only and reduce CVI memory use.\n");
        printf("ex: %s yolo11n_cv181x_int8.cvimodel lane.cvimodel 0.5 1 5 /dev/ttyS0\n", argv[0]);
        printf("ex: %s - lane.cvimodel 0.5 1 5 /dev/ttyS0\n", argv[0]);
        return 1;
    }

    const char* stop_model_path = argv[1];
    const char* lane_model_path = argv[2];
    const float stop_threshold = (argc >= 4) ? std::atof(argv[3]) : kDefaultStopThreshold;
    const int lane_tick = std::max(1, (argc >= 5) ? std::atoi(argv[4]) : kDefaultLaneTick);
    const int stop_tick = std::max(1, (argc >= 6) ? std::atoi(argv[5]) : kDefaultStopTick);
    const char* uart_device = (argc >= 7) ? argv[6] : "/dev/ttyS0";

    const bool stop_enabled = !model_disabled(stop_model_path);
    if (stop_enabled) {
        MA_LOGW(TAG, "Stop model will be lazy-loaded on stop ticks to avoid CVI ION OOM");
    }

    const bool use_jpeg_path = env_flag("CAMERA_USE_JPEG", false);
    const int camera_channel = env_int("CAMERA_CHANNEL", use_jpeg_path ? 1 : 0);
    const bool use_physical = env_flag("CAMERA_PHYSICAL", false);
    const int debug_tick_delay_ms = env_nonneg_int("DEBUG_TICK_DELAY_MS", 200);

    Device* device = Device::getInstance();
    Camera* camera = nullptr;

    Signal::install({SIGINT, SIGSEGV, SIGABRT, SIGTRAP, SIGTERM, SIGHUP, SIGQUIT, SIGPIPE}, [device](int sig) {
        std::cout << "Caught signal " << sig << std::endl;
        for (auto& sensor : device->getSensors()) {
            sensor->deInit();
        }
        exit(0);
    });

    for (auto& sensor : device->getSensors()) {
        if (sensor->getType() == ma::Sensor::Type::kCamera) {
            camera = static_cast<Camera*>(sensor);
            camera->init(0);

            Camera::CtrlValue value;
            value.i32 = camera_channel;
            camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, value);

            const int default_capture_size = stop_enabled ? kDefaultStopCaptureSize : kDefaultLaneCaptureSize;
            const int capture_width = env_int("CAMERA_WIDTH", default_capture_size);
            const int capture_height = env_int("CAMERA_HEIGHT", default_capture_size);
            MA_LOGI(TAG, "camera capture size: %dx%d", capture_width, capture_height);

            value.u16s[0] = static_cast<uint16_t>(capture_width);
            value.u16s[1] = static_cast<uint16_t>(capture_height);
            camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, value);

            value.i32 = use_physical ? 1 : 0;
            camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, value);
            break;
        }
    }

    if (!camera) {
        MA_LOGE(TAG, "No camera found");
        return 1;
    }

    const int uart_fd = setup_serial(uart_device);
    if (uart_fd < 0) {
        MA_LOGW(TAG, "UART unavailable; continuing with stdout only");
    }

    MA_LOGI(TAG,
            "lane_tick=%d stop_tick=%d stop_threshold=%.2f camera_use_jpeg=%d camera_channel=%d camera_physical=%d",
            lane_tick,
            stop_tick,
            stop_threshold,
            use_jpeg_path ? 1 : 0,
            camera_channel,
            use_physical ? 1 : 0);
    if (camera->startStream(Camera::StreamMode::kRefreshOnReturn) != MA_OK) {
        MA_LOGE(TAG, "Camera stream failed to start");
        if (uart_fd >= 0) {
            close(uart_fd);

        }
        return 1;
    }

    LoadedModel lane = load_model(lane_model_path, "lane", MA_MODEL_TYPE_BISENETV2);
    if (!is_loaded(lane)) {
        MA_LOGE(TAG, "Lane model failed to load after camera start");
        camera->stopStream();
        if (uart_fd >= 0) {
            close(uart_fd);
        }
        return 1;
    }

    int frame_count = 0;
    int retrieve_fail_count = 0;
    bool stop_hold_active = false;
    std::chrono::steady_clock::time_point stop_hold_until;
    std::chrono::steady_clock::time_point stop_rearm_until{};
    while (true) {
        ma_img_t frame;
        const ma_pixel_format_t frame_format = use_jpeg_path ? MA_PIXEL_FORMAT_JPEG : MA_PIXEL_FORMAT_RGB888;
        if (camera->retrieveFrame(frame, frame_format) != MA_OK) {
            ++retrieve_fail_count;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        retrieve_fail_count = 0;
        ++frame_count;
        g_current_frame_count = frame_count;

        if (stop_hold_active) {
            const auto now = std::chrono::steady_clock::now();
            if (now < stop_hold_until) {
                camera->returnFrame(frame);
                continue;
            }

            send_uart(uart_fd, "GO\n");
            stop_hold_active = false;
            stop_rearm_until = now + std::chrono::seconds(kStopDebounceSeconds);
            camera->returnFrame(frame);
            continue;
        }

        std::vector<uint8_t> rgb_buffer;
        ma_img_t rgb_frame{};
        ma_img_t* model_frame = &frame;
        if (use_jpeg_path) {
            dump_encoded_frame_once_if_requested(frame);
            if (!decode_jpeg_frame(frame, &rgb_buffer, &rgb_frame)) {
                camera->returnFrame(frame);
                continue;
            }
            model_frame = &rgb_frame;
        }

        dump_frame_once_if_requested(*model_frame, frame_count);

        bool ran_tick = false;
        if (frame_count % lane_tick == 0) {
            ran_tick = true;
            LaneDecision lane_decision;
            if (run_lane_model(lane, *model_frame, &lane_decision)) {
                char command[32];
                std::snprintf(command, sizeof(command), "STEER:%+.0f\n", lane_decision.steering_percent);
                send_uart(uart_fd, command);
                char weight_command[32];
                std::snprintf(weight_command, sizeof(weight_command), "STEER_WEIGHT:%d\n", lane_decision.weight);
                send_uart(uart_fd, weight_command);
            }
        }

        const bool stop_detection_armed = std::chrono::steady_clock::now() >= stop_rearm_until;
        if (stop_enabled && stop_detection_armed && frame_count % stop_tick == 0) {
            ran_tick = true;
            LoadedModel stop = load_model(stop_model_path, "stop");
            bool should_stop = false;
            if (is_loaded(stop) && run_stop_model(stop, *model_frame, stop_threshold, &should_stop, frame_count) && should_stop) {
                send_uart(uart_fd, "STOP\n");
                send_uart(uart_fd, "STOP_WEIGHT:1\n");
                stop_hold_active = true;
                stop_hold_until = std::chrono::steady_clock::now() + std::chrono::seconds(kStopTimeSeconds);
            }
            release_model(stop);
        }

        camera->returnFrame(frame);
        if (ran_tick && debug_tick_delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(debug_tick_delay_ms));
        }
    }

    if (uart_fd >= 0) {
        close(uart_fd);
    }
    release_model(lane);
    camera->stopStream();
    return 0;
}
