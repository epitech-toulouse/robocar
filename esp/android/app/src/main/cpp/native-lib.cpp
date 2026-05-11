#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <android/log.h>

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++ (avec support OpenCV)";
    return env->NewStringUTF(hello.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_processImageNative(
        JNIEnv* env,
        jobject /* this */,
        jlong matAddrRgba,
        jlong matAddrGray) {
    
    cv::Mat& mRgb = *(cv::Mat*)matAddrRgba;
    cv::Mat& mGray = *(cv::Mat*)matAddrGray;
    
    cv::cvtColor(mRgb, mGray, cv::COLOR_RGBA2GRAY);
    // Exemple d'analyse (détection de contours de base)
    cv::Canny(mGray, mGray, 50, 150);
}