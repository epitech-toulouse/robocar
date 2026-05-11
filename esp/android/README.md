# MyApplication

## Structure

- app/src/main/java/com/example/myapplication/navigation
  - AppRoot + bottom navigation
- app/src/main/java/com/example/myapplication/bluetooth
  - BLE client + BLE screen + BLE ViewModel
- app/src/main/java/com/example/myapplication/camera
  - Camera screen + CameraX pipeline + FrameProcessor
- app/src/main/java/com/example/myapplication/nativebridge
  - JNI bridge + C++ frame processor
- app/src/main/cpp
  - OpenCV native code entry point

## Data flow (high level)

CameraX -> FrameProcessor -> NativeCvBridge (JNI) -> native_cv.cpp -> result code
result code -> BleClient.sendAlgorithmResult -> BLE characteristic

## Where to plug the C++ algo

- app/src/main/cpp/native_cv.cpp
  - JNI entry: Java_com_example_myapplication_nativebridge_NativeCvBridge_processFrame

## OpenCV SDK path

Set in local.properties (example):

opencv.dir=/path/to/OpenCV-android-sdk/sdk/native/jni
