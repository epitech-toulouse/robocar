#include <jni.h>
#include <android/log.h>
#include <opencv2/core.hpp>

#define LOG_TAG "NativeCv"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

extern "C" JNIEXPORT jint JNICALL
Java_com_example_myapplication_nativebridge_NativeCvBridge_processFrame(
        JNIEnv *env,
        jobject /*thiz*/,
        jint width,
        jint height,
        jobject yBuffer,
        jobject uBuffer,
        jobject vBuffer,
        jint yRowStride,
        jint uvRowStride,
        jint uvPixelStride,
        jint rotationDegrees) {
    if (yBuffer == nullptr || uBuffer == nullptr || vBuffer == nullptr) {
        return -1;
    }

    auto *yData = static_cast<uint8_t *>(env->GetDirectBufferAddress(yBuffer));
    auto *uData = static_cast<uint8_t *>(env->GetDirectBufferAddress(uBuffer));
    auto *vData = static_cast<uint8_t *>(env->GetDirectBufferAddress(vBuffer));

    if (yData == nullptr || uData == nullptr || vData == nullptr) {
        return -2;
    }

    cv::Mat yPlane(height, width, CV_8UC1, yData, yRowStride);
    (void) yPlane;

    LOGI("Frame %dx%d rot=%d yStride=%d uvStride=%d uvPixel=%d", width, height,
         rotationDegrees, yRowStride, uvRowStride, uvPixelStride);

    return 0;
}
