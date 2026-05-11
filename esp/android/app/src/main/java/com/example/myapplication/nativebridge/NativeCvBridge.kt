package com.example.myapplication.nativebridge

import java.nio.ByteBuffer

object NativeCvBridge {
    private val nativeAvailable: Boolean = try {
        System.loadLibrary("native_cv")
        true
    } catch (_: UnsatisfiedLinkError) {
        false
    }

    fun isAvailable(): Boolean = nativeAvailable

    fun processFrame(
        width: Int,
        height: Int,
        yBuffer: ByteBuffer,
        uBuffer: ByteBuffer,
        vBuffer: ByteBuffer,
        yRowStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int,
        rotationDegrees: Int
    ): Int {
        if (!nativeAvailable) {
            return 0
        }
        return processFrameNative(
            width = width,
            height = height,
            yBuffer = yBuffer,
            uBuffer = uBuffer,
            vBuffer = vBuffer,
            yRowStride = yRowStride,
            uvRowStride = uvRowStride,
            uvPixelStride = uvPixelStride,
            rotationDegrees = rotationDegrees
        )
    }

    private external fun processFrameNative(
        width: Int,
        height: Int,
        yBuffer: ByteBuffer,
        uBuffer: ByteBuffer,
        vBuffer: ByteBuffer,
        yRowStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int,
        rotationDegrees: Int
    ): Int
}
