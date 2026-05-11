package com.example.myapplication.nativebridge

import androidx.camera.core.ImageProxy
import com.example.myapplication.camera.FrameProcessor

class NativeCvFrameProcessor(
    private val onResult: (Int) -> Unit
) : FrameProcessor {
    override fun process(image: ImageProxy) {
        if (!NativeCvBridge.isAvailable()) {
            onResult(0)
            return
        }

        val planes = image.planes
        if (planes.size < 3) {
            return
        }

        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]

        val resultCode = NativeCvBridge.processFrame(
            image.width,
            image.height,
            yPlane.buffer,
            uPlane.buffer,
            vPlane.buffer,
            yPlane.rowStride,
            uPlane.rowStride,
            uPlane.pixelStride,
            image.imageInfo.rotationDegrees
        )
        onResult(resultCode)
    }
}
