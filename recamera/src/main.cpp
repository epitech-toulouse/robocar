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

LoadedModel load_model(const char* path, const char* label) {
    LoadedModel loaded;
    loaded.engine = new ma::engine::EngineCVI();

    ma_err_t ret = loaded.engine->init();
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

    loaded.model = ma::ModelFactory::create(loaded.engine);
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
    if (lane.model->getOutputType() != MA_OUTPUT_TYPE_SEGMENT) {
        MA_LOGW(TAG, "Lane model output type must be MA_OUTPUT_TYPE_SEGMENT");
        return false;
    }

    ma_tensor_t tensor = {
        .size = frame.size,
        .is_physical = true,
        .is_variable = false,
    };
    tensor.data.data = reinterpret_cast<void*>(frame.data);
    lane.engine->setInput(0, tensor);

    ma::model::Segmentor* segmentor = static_cast<ma::model::Segmentor*>(lane.model);
    if (segmentor->run(nullptr) != MA_OK) {
        MA_LOGW(TAG, "Lane model inference failed");
        return false;
    }

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
        *decision = LaneDecision{};
        return true;
    }

    close_5x5(lane_mask, mask_width, mask_height);
    *decision = decide_lane(lane_mask, mask_width, mask_height);
    return true;
}

bool run_stop_model(LoadedModel& stop, ma_img_t& frame, float threshold, bool* should_stop) {
    *should_stop = false;
    if (stop.model->getOutputType() != MA_OUTPUT_TYPE_BBOX) {
        MA_LOGW(TAG, "Stop model output type must be MA_OUTPUT_TYPE_BBOX");
        return false;
    }

    ma_tensor_t tensor = {
        .size = frame.size,
        .is_physical = true,
        .is_variable = false,
    };
    tensor.data.data = reinterpret_cast<void*>(frame.data);
    stop.engine->setInput(0, tensor);

    ma::model::Detector* detector = static_cast<ma::model::Detector*>(stop.model);
    detector->setConfig(MA_MODEL_CFG_OPT_THRESHOLD, threshold);
    if (detector->run(nullptr) != MA_OK) {
        MA_LOGW(TAG, "Stop model inference failed");
        return false;
    }

    auto results = detector->getResults();
    for (auto result : results) {
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

    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage:\n");
        printf("   %s stop_sign.cvimodel lane.cvimodel [stop_threshold] [lane_tick] [stop_tick] [uart_device]\n", argv[0]);
        printf("ex: %s yolo11n_cv181x_int8.cvimodel lane.cvimodel 0.5 1 5 /dev/ttyS0\n", argv[0]);
        return 1;
    }

    const char* stop_model_path = argv[1];
    const char* lane_model_path = argv[2];
    const float stop_threshold = (argc >= 4) ? std::atof(argv[3]) : kDefaultStopThreshold;
    const int lane_tick = std::max(1, (argc >= 5) ? std::atoi(argv[4]) : kDefaultLaneTick);
    const int stop_tick = std::max(1, (argc >= 6) ? std::atoi(argv[5]) : kDefaultStopTick);
    const char* uart_device = (argc >= 7) ? argv[6] : "/dev/ttyS0";

    LoadedModel stop = load_model(stop_model_path, "stop");
    LoadedModel lane = load_model(lane_model_path, "lane");
    if (stop.model == nullptr || lane.model == nullptr) {
        release_model(stop);
        release_model(lane);
        return 1;
    }

    if (stop.input_width != lane.input_width || stop.input_height != lane.input_height) {
        MA_LOGW(TAG,
                "Model input sizes differ: stop=%dx%d lane=%dx%d. Camera will use lane size; convert both models to the same input size for best results.",
                stop.input_width,
                stop.input_height,
                lane.input_width,
                lane.input_height);
    }

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
            value.i32 = 0;
            camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, value);

            value.u16s[0] = lane.input_width;
            value.u16s[1] = lane.input_height;
            camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, value);

            value.i32 = 1;
            camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, value);
            break;
        }
    }

    if (!camera) {
        MA_LOGE(TAG, "No camera found");
        release_model(stop);
        release_model(lane);
        return 1;
    }

    const int uart_fd = setup_serial(uart_device);
    if (uart_fd < 0) {
        MA_LOGW(TAG, "UART unavailable; continuing with stdout only");
    }

    MA_LOGI(TAG, "lane_tick=%d stop_tick=%d stop_threshold=%.2f", lane_tick, stop_tick, stop_threshold);
    camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    int frame_count = 0;
    while (true) {
        ma_img_t frame;
        if (camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) != MA_OK) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        ++frame_count;

        if (frame_count % lane_tick == 0) {
            LaneDecision lane_decision;
            if (run_lane_model(lane, frame, &lane_decision)) {
                char command[32];
                std::snprintf(command, sizeof(command), "STEER:%+.0f\n", lane_decision.steering_percent);
                send_uart(uart_fd, command);
                printf("[LANE] frame=%d status=%s steer=%+.1f\n", frame_count, lane_decision.status, lane_decision.steering_percent);
            }
        }

        if (frame_count % stop_tick == 0) {
            bool should_stop = false;
            if (run_stop_model(stop, frame, stop_threshold, &should_stop) && should_stop) {
                send_uart(uart_fd, "STOP\n");
                printf("[STOP_SIGN] Sent STOP\n");
            }
        }

        camera->returnFrame(frame);
    }

    if (uart_fd >= 0) {
        close(uart_fd);
    }
    camera->stopStream();
    release_model(stop);
    release_model(lane);
    return 0;
}
