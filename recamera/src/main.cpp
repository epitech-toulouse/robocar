#include <iostream>
#include <chrono>
#include <thread>
#include <iterator>

#include <sscma.h>
#include <video.h>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

using namespace ma;

#define TAG "model_detector"

int setup_serial(const char* device) {
    // Open the device file in Read/Write mode
    // O_NOCTTY: Prevents the port from becoming the "controlling terminal" for this process
    int fd = open(device, O_RDWR | O_NOCTTY);
    
    if (fd == -1) {
        perror("Unable to open UART");
        return -1;
    }

    struct termios options;
    
    // Get the current configuration of the serial port and store it in 'options'
    tcgetattr(fd, &options);

    // Set the Output Baud Rate (speed at which the reCamera sends data to your car)
    cfsetospeed(&options, B115200);

    // PARENB: Disable Parity bit generation (Standard for 8N1)
    options.c_cflag &= ~PARENB;
    
    // CSTOPB: Use 1 Stop bit (clearing this bit ensures it's not 2 bits)
    options.c_cflag &= ~CSTOPB;
    
    // CSIZE: A mask to clear the current character size bits
    options.c_cflag &= ~CSIZE;
    
    // CS8: Set the character size to 8 bits per byte
    options.c_cflag |= CS8;
    
    // CLOCAL: Ignore modem control lines (use only TX/RX/GND)
    // CREAD: Enable the receiver so you can actually read incoming data
    options.c_cflag |= (CLOCAL | CREAD);
    
    // ICANON: Disable "Canonical" mode (disables line-by-line editing/buffering)
    // ECHO/ECHOE: Disable echoing of characters back to the sender
    // ISIG: Disable interpretation of signal-generating characters (INTR, QUIT, SUSP)
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    
    // OPOST: Disable "Post-processing" of output (ensures data is sent exactly as written)
    options.c_oflag &= ~OPOST;

    // Apply these new settings immediately (TCSANOW) to the hardware
    tcsetattr(fd, TCSANOW, &options);
    
    return fd;
}

int main(int argc, char** argv) {
	if (argc < 2) {
		printf("Usage:\n");
		printf("   %s cvimodel [threshold]\n", argv[0]);
		printf("ex: %s yolo11.cvimodel 0.5\n", argv[0]);
		exit(-1);
	}

	float threshold = 0.5f; // default threshold
	if (argc >= 3) {
		threshold = atof(argv[2]);
	}

	ma_err_t ret = MA_OK;

	// EngineCVI is the hardware-accelerated inference engine for the Sophgo SG2002 RISC-V SoC
	auto* engine = new ma::engine::EngineCVI();

	// Initialize the engine and load the .cvimodel file
	ret          = engine->init();
	if (ret != MA_OK) {
		MA_LOGE(TAG, "engine init failed");
		return 1;
	}
	ret = engine->load(argv[1]);

	MA_LOGI(TAG, "engine load model %s", argv[1]);
	if (ret != MA_OK) {
		MA_LOGE(TAG, "engine load model failed");
		return 1;
	}

	// ModelFactory automatically creates the appropriate model class (e.g., Detector) based on the loaded model's metadata
	ma::Model* model = ma::ModelFactory::create(engine);

	if (model == nullptr) {
		MA_LOGE(TAG, "model not supported");
		return 1;
	}

	MA_LOGI(TAG, "model type: %d", model->getType());
	MA_LOGI(TAG, "threshold: %f", threshold);

	if (model->getInputType() != MA_INPUT_TYPE_IMAGE) {
		MA_LOGE(TAG, "model input type not supported");
		return 1;
	}

	// Get model input dimensions required by the neural network
	const ma_img_t* model_input = static_cast<const ma_img_t*>(model->getInput());
	int input_width = model_input->width;
	int input_height = model_input->height;
	MA_LOGI(TAG, "model input size: %dx%d", input_width, input_height);

	// Device is a singleton managing all sensors (Camera, Mic, etc.) on the reCamera
	Device* device = Device::getInstance();
	Camera* camera = nullptr;

	// Signal handler to ensure camera sensors are properly de-initialized on exit
	Signal::install({SIGINT, SIGSEGV, SIGABRT, SIGTRAP, SIGTERM, SIGHUP, SIGQUIT, SIGPIPE}, [device](int sig) {
			std::cout << "Caught signal " << sig << std::endl;
			for (auto& sensor : device->getSensors()) {
			sensor->deInit();
			}
			exit(0);
			});

	// Find and initialize the camera sensor
	for (auto& sensor : device->getSensors()) {
		if (sensor->getType() == ma::Sensor::Type::kCamera) {
			camera = static_cast<Camera*>(sensor);
			camera->init(0); // Initialize camera index 0

			Camera::CtrlValue value;

			// Set camera channel
			value.i32 = 0;
			camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, value);

			// Set window size to match model input requirements
			value.u16s[0] = input_width;
			value.u16s[1] = input_height;
			camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, value);

			// Set physical mode (DMA/Zero-copy if supported)
			value.i32 = 1;
			camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, value);
			break;
		}
	}

	if (!camera) {
		MA_LOGE(TAG, "No camera found");
		return 1;
	}

	// Start streaming frames. kRefreshOnReturn ensures we only get new frames when we're ready
	camera->startStream(Camera::StreamMode::kRefreshOnReturn);

	int frame_count = 0;
	long long total_time = 0;

	while (true) {
		ma_img_t frame;
		// Retrieve a frame from the camera buffer in RGB888 format
		if (camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) == MA_OK) {
			auto capture_start = std::chrono::high_resolution_clock::now();

			// Wrap the raw frame data into a tensor structure for the inference engine
			ma_tensor_t tensor = {
				.size        = frame.size,
				.is_physical = true,  // The reCamera uses physical addresses for memory-mapped I/O
				.is_variable = false,
			};
			tensor.data.data = reinterpret_cast<void*>(frame.data);

			// Pass the tensor to the hardware engine's first input index
			engine->setInput(0, tensor);
			auto capture_end = std::chrono::high_resolution_clock::now();

			// Set a callback to return the frame to the camera driver once preprocessing/inference is finished
			model->setPreprocessDone([camera, &frame](void* ctx) { camera->returnFrame(frame); });

			if (model->getOutputType() == MA_OUTPUT_TYPE_BBOX) {
				// Cast to Detector class for bounding box operations
				ma::model::Detector* detector = static_cast<ma::model::Detector*>(model);

				// Set the confidence threshold for detections
				detector->setConfig(MA_MODEL_CFG_OPT_THRESHOLD, threshold);

				auto inference_start = std::chrono::high_resolution_clock::now();
				// Execute the neural network inference
				detector->run(nullptr);

				// Retrieve the parsed results (bounding boxes, classes, scores)
				auto _results = detector->getResults();
				auto inference_end = std::chrono::high_resolution_clock::now();

				auto capture_duration = std::chrono::duration_cast<std::chrono::milliseconds>(capture_end - capture_start);
				auto inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - inference_start);
				auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - capture_start);

				total_time += total_duration.count();
				frame_count++;

				size_t num_detections = std::distance(_results.begin(), _results.end());
				const float STOP_SIGN_F_CONSTANT = 250.0f;
				int uart_fd = setup_serial("/dev/ttyS0");
				for (auto result : _results) {
					// Approximate distance based on the height of the bounding box
					// This is a rough estimation where larger box = closer object
					float distance = (result.h > 0) ? (1.0f / result.h) * 10.0f : 0.0f; 
					// Specific requested format for robot car logic
					if (result.target == 11) {
						// result.h is normalized (0.0 to 1.0), so we use it directly 
						// if your constant is tuned for normalized height.
						float distance_cm = STOP_SIGN_F_CONSTANT / result.h;

						printf("[STOP_SIGN] Distance: %.2f cm\n", distance_cm);
						if (distance_cm < 100.0f) {
							// Send a command to the TX pin
							const char* cmd = "STOP\n";
							write(uart_fd, cmd, 5); 
							printf("Sent to UART: %s", cmd);
						}

						// Robot logic based on real-world distance
						// I must send the data to rx and tx
					}
				}

				if (frame_count % 10 == 0) {
					double avg_time = static_cast<double>(total_time) / frame_count;
					printf("Average processing time per frame: %.2f ms (over %d frames)\n", avg_time, frame_count);
				}
			} else {
				MA_LOGW(TAG, "Model output type not supported for detection, only bbox supported in this example");
				// If we don't run the model, we MUST return the frame manually to avoid memory leaks
				camera->returnFrame(frame);
			}
		} else {
			// Wait a bit if no frame is available to reduce CPU usage
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
	}

	camera->stopStream();
	ma::ModelFactory::remove(model);

	return 0;
}
