package com.example.myapplication

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

data class StopSignDetectionResult(
    val detected: Boolean,
    val boundingRect: Rect?,
    val confidence: Int,
    val polygon: Array<Point> = emptyArray()
)

data class StopSignDetectorParams(
    val minAreaRatio: Double = 0.003,
    val maxAreaRatio: Double = 0.35
)

class StopSignDetector(
    initialParams: StopSignDetectorParams = StopSignDetectorParams()
) {

    private val hsv = Mat()
    private val mask1 = Mat()
    private val mask2 = Mat()
    private val redMask = Mat()
    private val morphed = Mat()
    private val hierarchy = Mat()
    @Volatile
    private var params: StopSignDetectorParams = initialParams

    fun detect(rgbaFrame: Mat): StopSignDetectionResult {
        val params = params
        Imgproc.cvtColor(rgbaFrame, hsv, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(hsv, hsv, Imgproc.COLOR_RGB2HSV)

        Core.inRange(hsv, Scalar(0.0, 90.0, 60.0), Scalar(12.0, 255.0, 255.0), mask1)
        Core.inRange(hsv, Scalar(168.0, 90.0, 60.0), Scalar(180.0, 255.0, 255.0), mask2)
        Core.bitwise_or(mask1, mask2, redMask)

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        Imgproc.morphologyEx(redMask, morphed, Imgproc.MORPH_CLOSE, kernel)
        Imgproc.morphologyEx(morphed, morphed, Imgproc.MORPH_OPEN, kernel)
        kernel.release()

        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(
            morphed,
            contours,
            hierarchy,
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        val imageArea = rgbaFrame.width() * rgbaFrame.height().toDouble()
        var bestRect: Rect? = null
        var bestPolygon: Array<Point> = emptyArray()
        var bestScore = 0.0

        for (contour in contours) {
            val area = Imgproc.contourArea(contour)
            if (area < imageArea * params.minAreaRatio || area > imageArea * params.maxAreaRatio) {
                contour.release()
                continue
            }

            val contour2f = MatOfPoint2f(*contour.toArray())
            val perimeter = Imgproc.arcLength(contour2f, true)
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(contour2f, approx, perimeter * 0.03, true)
            val polygon = approx.toArray()
            val vertexCount = polygon.size

            if (vertexCount in 6..10) {
                val rect = Imgproc.boundingRect(MatOfPoint(*polygon))
                val aspectRatio = rect.width.toDouble() / max(rect.height, 1)
                val rectArea = max(rect.width * rect.height, 1).toDouble()
                val fillRatio = area / rectArea

                if (aspectRatio in 0.75..1.25 && fillRatio in 0.45..0.95) {
                    val vertexScore = 1.0 - min(abs(vertexCount - 8), 3) / 3.0
                    val aspectScore = 1.0 - min(abs(aspectRatio - 1.0), 0.25) / 0.25
                    val fillScore = 1.0 - min(abs(fillRatio - 0.68), 0.30) / 0.30
                    val areaScore = min(area / (imageArea * 0.04), 1.0)
                    val score = vertexScore * 0.35 + aspectScore * 0.25 + fillScore * 0.20 + areaScore * 0.20

                    if (score > bestScore) {
                        bestScore = score
                        bestRect = rect
                        bestPolygon = polygon
                    }
                }
            }

            approx.release()
            contour2f.release()
            contour.release()
        }

        contours.forEach { if (!it.empty()) it.release() }

        val confidence = (bestScore * 100.0).toInt().coerceIn(0, 100)
        return StopSignDetectionResult(
            detected = bestRect != null && confidence >= 45,
            boundingRect = bestRect,
            confidence = confidence,
            polygon = bestPolygon
        )
    }

    fun release() {
        hsv.release()
        mask1.release()
        mask2.release()
        redMask.release()
        morphed.release()
        hierarchy.release()
    }

    fun updateParams(newParams: StopSignDetectorParams) {
        params = newParams
    }
}
