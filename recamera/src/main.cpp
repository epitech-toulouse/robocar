#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
constexpr float kStopDistanceCm = 100.0f;
constexpr int kLaneClassId = 1;
constexpr int kDefaultLaneTick = 1;
constexpr int kDefaultStopTick = 5;
constexpr float kDefaultStopThreshold = 0.5f;
constexpr float kBarricadeThreshold = 0.40f;

struct LoadedModel {
    ma::engine::EngineCVI* engine = nullptr;
    ma::Model* model = nullptr;
    int input_width = 0;
    int input_height = 0;
};

struct LaneDecision {
    float steering_percent = 0.0f;
    const char* status = "SEARCHING";
};

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
        MA_LOGI(TAG, "%s model disabled", label);
        return loaded;
    }
    if (access(path, R_OK) != 0) {
        MA_LOGE(TAG, "%s model file not found: %s", label, path);
        return loaded;
    }
    loaded.engine = new ma::engine::EngineCVI();
    const size_t shared_memory = cvi_shared_memory_bytes();
    if (shared_memory > 0) {
        MA_LOGI(TAG, "%s engine shared memory request: %zu bytes", label, shared_memory);
    } else {
        MA_LOGI(TAG, "%s engine using cvimodel default shared memory", label);
    }
    ma_err_t ret = loaded.engine->init(shared_memory);
    if (ret != MA_OK) {
        MA_LOGE(TAG, "%s engine init failed", label);
        return loaded;
    }
    ret = loaded.engine->load(path);
    MA_LOGI(TAG, "%s engine load model %s", label, path);
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
    MA_LOGI(TAG, "%s model type: %d", label, loaded.model->getType());
    MA_LOGI(TAG, "%s input size: %dx%d", label, loaded.input_width, loaded.input_height);
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

bool mask_get(const std::vector<uint8_t>& mask, int width, int x, int y) {
    return mask[y * width + x] != 0;
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
    std::vector<uint8_t> closed(mask.size());
    dilate_5x5(mask, width, height, tmp);
    erode_5x5(tmp, width, height, closed);
    mask.swap(closed);
}

bool is_valid_lane_row(const std::vector<uint8_t>& mask, int width, int y, int* lx, int* rx) {
    int first = -1;
    int last = -1;
    int count = 0;
    for (int x = 0; x < width; ++x) {
        if (mask_get(mask, width, x, y)) {
            if (first < 0) {
                first = x;
            }
            last = x;
            ++count;
        }
    }

    if (count < 5) {
        return false;
    }

    const int min_gap = std::max(20, static_cast<int>(width * 140.0f / 512.0f));
    const int max_gap = std::max(min_gap + 1, static_cast<int>(width * 470.0f / 512.0f));
    const int gap = last - first;
    if (gap <= min_gap || gap >= max_gap) {
        return false;
    }

    const int mid = (first + last) / 2;
    if (mask_get(mask, width, mid, y)) {
        return false;
    }

    *lx = first;
    *rx = last;
    return true;
}

LaneDecision decide_lane(const std::vector<uint8_t>& mask, int width, int height) {
    LaneDecision decision;
    const int mid_x = width / 2;
    const int wall_y = std::min(height - 1, static_cast<int>(height * 0.70f));

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
    const int sweep_start = std::max(0, height - 15);
    const int sweep_end = std::min(height - 1, static_cast<int>(height * 100.0f / 512.0f));
    for (int y = sweep_start; y > sweep_end; --y) {
        int cur_lx = -1;
        int cur_rx = -1;
        if (!is_valid_lane_row(mask, width, y, &cur_lx, &cur_rx)) {
            continue;
        }

        int confirm_count = 0;
        for (int check_y = y - 1; check_y >= y - 10 && check_y >= 0; --check_y) {
            int ignored_lx = -1;
            int ignored_rx = -1;
            if (is_valid_lane_row(mask, width, check_y, &ignored_lx, &ignored_rx)) {
                ++confirm_count;
            }
        }

        if (confirm_count >= 8) {
            lx = cur_lx;
            rx = cur_rx;
            found_y = y;
            break;
        }
    }

    float target_x = static_cast<float>(mid_x);
    if (barricaded) {
        int left_mass = 0;
        int right_mass = 0;
        const int y0 = static_cast<int>(height * 0.1f);
        const int y1 = static_cast<int>(height * 0.5f);
        for (int y = y0; y < y1; ++y) {
            for (int x = 0; x < mid_x; ++x) {
                left_mass += mask_get(mask, width, x, y) ? 1 : 0;
            }
            for (int x = mid_x; x < width; ++x) {
                right_mass += mask_get(mask, width, x, y) ? 1 : 0;
            }
        }
        decision.steering_percent = (left_mass > right_mass) ? -100.0f : 100.0f;
        decision.status = "BARRICADE";
        return decision;
    }

    if (lx >= 0 && rx >= 0 && found_y >= 0) {
        target_x = static_cast<float>((lx + rx) / 2);
        decision.status = "CENTERED";
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

        if (count > 0) {
            const float avg_x = static_cast<float>(sum_x) / static_cast<float>(count);
            decision.steering_percent = (avg_x < mid_x) ? 100.0f : -100.0f;
            decision.status = (avg_x < mid_x) ? "FULL_RIGHT_TURN" : "FULL_LEFT_TURN";
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
    printf("[LANE_TICK] begin frame_size=%zu\n", frame.size);
    std::fflush(stdout);

    const ma_output_type_t output_type = lane.model->getOutputType();
    if (output_type != MA_OUTPUT_TYPE_SEGMENT) {
        printf("[LANE_TICK] wrong output type=%d expected=%d\n",
               static_cast<int>(output_type),
               static_cast<int>(MA_OUTPUT_TYPE_SEGMENT));
        std::fflush(stdout);
        MA_LOGW(TAG, "Lane model output type must be MA_OUTPUT_TYPE_SEGMENT");
        return false;
    }

    CpuReadableFrame prepared;
    if (!make_cpu_readable_frame(frame, lane.input_width, lane.input_height, &prepared)) {
        MA_LOGW(TAG, "Lane model frame prepare failed");
        return false;
    }
    if (frame.physical ||
        frame.width != static_cast<uint16_t>(lane.input_width) ||
        frame.height != static_cast<uint16_t>(lane.input_height)) {
        printf("[LANE_TICK] prepared cpu frame %ux%u -> %dx%d physical=%d\n",
               frame.width,
               frame.height,
               lane.input_width,
               lane.input_height,
               frame.physical ? 1 : 0);
        std::fflush(stdout);
    }

    ma::model::Segmentor* segmentor = static_cast<ma::model::Segmentor*>(lane.model);
    if (segmentor->run(&prepared.frame) != MA_OK) {
        MA_LOGW(TAG, "Lane model inference failed");
        return false;
    }
    printf("[LANE_TICK] inference done\n");
    std::fflush(stdout);

    auto results = segmentor->getResults();
    std::vector<uint8_t> lane_mask;
    int mask_width = 0;
    int mask_height = 0;
    bool found_lane_mask = false;

    for (auto result : results) {
        if (result.box.target != kLaneClassId) {
            continue;
        }
        unpack_segment_mask(result, lane_mask, &mask_width, &mask_height);
        found_lane_mask = true;
        break;
    }

    if (!found_lane_mask) {
        printf("[LANE_TICK] no lane mask\n");
        std::fflush(stdout);
        *decision = LaneDecision{};
        return true;
    }

    close_5x5(lane_mask, mask_width, mask_height);
    *decision = decide_lane(lane_mask, mask_width, mask_height);
    printf("[LANE_TICK] mask=%dx%d decision=%s steer=%+.1f\n",
           mask_width,
           mask_height,
           decision->status,
           decision->steering_percent);
    std::fflush(stdout);
    return true;
}

bool run_stop_model(LoadedModel& stop, ma_img_t& frame, float threshold, bool* should_stop) {
    printf("[STOP_TICK] begin frame_size=%zu threshold=%.2f\n", frame.size, threshold);
    std::fflush(stdout);

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
    if (frame.physical ||
        frame.width != static_cast<uint16_t>(stop.input_width) ||
        frame.height != static_cast<uint16_t>(stop.input_height)) {
        printf("[STOP_TICK] resized %dx%d -> %dx%d\n",
               frame.width, frame.height, stop.input_width, stop.input_height);
        std::fflush(stdout);
    }

    ma::model::Detector* detector = static_cast<ma::model::Detector*>(stop.model);
    detector->setConfig(MA_MODEL_CFG_OPT_THRESHOLD, threshold);
    if (detector->run(&prepared.frame) != MA_OK) {
        MA_LOGW(TAG, "Stop model inference failed");
        return false;
    }
    printf("[STOP_TICK] inference done\n");
    std::fflush(stdout);

    auto results = detector->getResults();
    int detection_count = 0;
    for (auto result : results) {
        ++detection_count;
        if (result.target != kStopSignClassId || result.h <= 0.0f) {
            continue;
        }

        const float distance_cm = kStopSignDistanceConstant / result.h;
        printf("[STOP_SIGN] score=%.2f distance=%.2fcm\n", result.score, distance_cm);
        if (distance_cm < kStopDistanceCm) {
            *should_stop = true;
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

            const int capture_width = env_int("CAMERA_WIDTH", 128);
            const int capture_height = env_int("CAMERA_HEIGHT", 128);
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
    while (true) {
        ma_img_t frame;
        const ma_pixel_format_t frame_format = use_jpeg_path ? MA_PIXEL_FORMAT_JPEG : MA_PIXEL_FORMAT_RGB888;
        if (camera->retrieveFrame(frame, frame_format) != MA_OK) {
            ++retrieve_fail_count;
            if (retrieve_fail_count == 1 || retrieve_fail_count % 100 == 0) {
                printf("[LOOP] retrieveFrame failed count=%d\n", retrieve_fail_count);
                std::fflush(stdout);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        retrieve_fail_count = 0;
        ++frame_count;
        printf("[LOOP] frame=%d fmt=%s size=%zu\n", frame_count, use_jpeg_path ? "jpeg" : "rgb", frame.size);
        std::fflush(stdout);

        std::vector<uint8_t> rgb_buffer;
        ma_img_t rgb_frame{};
        ma_img_t* model_frame = &frame;
        if (use_jpeg_path) {
            if (!decode_jpeg_frame(frame, &rgb_buffer, &rgb_frame)) {
                camera->returnFrame(frame);
                continue;
            }
            model_frame = &rgb_frame;
            printf("[LOOP] frame=%d jpeg decoded rgb=%ux%u size=%u\n",
                   frame_count,
                   rgb_frame.width,
                   rgb_frame.height,
                   rgb_frame.size);
            std::fflush(stdout);
        }

        if (frame_count % lane_tick == 0) {
            printf("[LOOP] frame=%d lane tick\n", frame_count);
            std::fflush(stdout);
            LaneDecision lane_decision;
            if (run_lane_model(lane, *model_frame, &lane_decision)) {
                char command[32];
                std::snprintf(command, sizeof(command), "STEER:%+.0f\n", lane_decision.steering_percent);
                send_uart(uart_fd, command);
                printf("[LANE] frame=%d status=%s steer=%+.1f\n", frame_count, lane_decision.status, lane_decision.steering_percent);
            }
        }

        if (stop_enabled && frame_count % stop_tick == 0) {
            printf("[LOOP] frame=%d stop tick\n", frame_count);
            std::fflush(stdout);

            LoadedModel stop = load_model(stop_model_path, "stop");
            bool should_stop = false;
            if (is_loaded(stop) && run_stop_model(stop, *model_frame, stop_threshold, &should_stop) && should_stop) {
                send_uart(uart_fd, "STOP\n");
                printf("[STOP_SIGN] Sent STOP\n");
            }
            release_model(stop);
            printf("[STOP_TICK] stop model released\n");
            std::fflush(stdout);
        }

        camera->returnFrame(frame);
        printf("[LOOP] frame=%d processed\n", frame_count);
        std::fflush(stdout);
    }

    if (uart_fd >= 0) {
        close(uart_fd);
    }
    release_model(lane);
    camera->stopStream();
    return 0;
}
