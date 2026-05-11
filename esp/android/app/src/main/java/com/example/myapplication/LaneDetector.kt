package com.example.myapplication

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

data class RoadLine(
    val bottom: Point,
    val top: Point,
    val slope: Double,
    val intercept: Double,
    val score: Double,
    val estimated: Boolean
)

data class LaneDetectionResult(
    val leftLine: RoadLine?,
    val rightLine: RoadLine?,
    val centerBottom: Point?,
    val centerTop: Point?,
    val confidence: Int,
    val roiPolygon: Array<Point>,
    val usedEstimatedLane: Boolean
)

data class LaneDetectorParams(
    val blurKernelSize: Double = 5.0,
    val cannyLowThreshold: Double = 60.0,
    val cannyHighThreshold: Double = 180.0,
    val roiTopRatio: Double = 0.40,
    val roiBottomRatio: Double = 0.99,
    val roiCenterXRatio: Double = 0.50,
    val roiTopHalfWidthRatio: Double = 0.45,
    val roiBottomHalfWidthRatio: Double = 0.46,
    val whiteValueMin: Double = 170.0,
    val whiteSaturationMax: Double = 80.0,
    val morphologyKernelSize: Int = 7,
    val minSlopeAbs: Double = 0.35,
    val maxSlopeAbs: Double = 3.2,
    val houghThreshold: Int = 45,
    val minLineLength: Double = 45.0,
    val maxLineGap: Double = 70.0,
    val defaultLaneWidthBottomRatio: Double = 0.42,
    val defaultLaneWidthTopRatio: Double = 0.18
)

class LaneDetector(
    initialParams: LaneDetectorParams = LaneDetectorParams()
) {

    private enum class LaneSide { LEFT, RIGHT }

    private data class LineCandidate(
        val slope: Double,
        val intercept: Double,
        val weight: Double
    )

    private val gray = Mat()
    private val hsv = Mat()
    private val blurred = Mat()
    private val brightMask = Mat()
    private val whiteMask = Mat()
    private val laneMask = Mat()
    private val edges = Mat()
    private val roiMask = Mat()
    private val maskedLaneMask = Mat()
    private val maskedEdges = Mat()
    private val lineSegments = Mat()

    @Volatile
    private var params: LaneDetectorParams = initialParams

    private var lastKnownBottomLaneWidthPx: Double? = null
    private var lastKnownTopLaneWidthPx: Double? = null

    fun detect(rgbaFrame: Mat): LaneDetectionResult {
        val params = params
        val width = rgbaFrame.width()
        val height = rgbaFrame.height()
        val bottomY = (height * params.roiBottomRatio).coerceIn(0.0, (height - 1).toDouble())
        val topY = height * params.roiTopRatio
        val verticalSpan = max(bottomY - topY, 1.0)

        Imgproc.cvtColor(rgbaFrame, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.cvtColor(rgbaFrame, hsv, Imgproc.COLOR_RGBA2BGR)
        Imgproc.cvtColor(hsv, hsv, Imgproc.COLOR_BGR2HSV)
        Imgproc.GaussianBlur(
            gray,
            blurred,
            Size(params.blurKernelSize, params.blurKernelSize),
            0.0
        )

        Imgproc.threshold(
            blurred,
            brightMask,
            0.0,
            255.0,
            Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU
        )
        Core.inRange(
            hsv,
            Scalar(0.0, 0.0, params.whiteValueMin),
            Scalar(180.0, params.whiteSaturationMax, 255.0),
            whiteMask
        )
        Core.bitwise_and(brightMask, whiteMask, laneMask)

        val morphologyKernel = Imgproc.getStructuringElement(
            Imgproc.MORPH_RECT,
            Size(params.morphologyKernelSize.toDouble(), params.morphologyKernelSize.toDouble())
        )
        Imgproc.morphologyEx(laneMask, laneMask, Imgproc.MORPH_CLOSE, morphologyKernel)
        Imgproc.morphologyEx(laneMask, laneMask, Imgproc.MORPH_OPEN, morphologyKernel)
        morphologyKernel.release()

        val roiPolygon = buildRoiPolygon(width, height)
        applyRoiMask(laneMask, roiPolygon, maskedLaneMask)
        Imgproc.Canny(maskedLaneMask, maskedEdges, params.cannyLowThreshold, params.cannyHighThreshold)

        Imgproc.HoughLinesP(
            maskedEdges,
            lineSegments,
            1.0,
            Math.PI / 180.0,
            params.houghThreshold,
            params.minLineLength,
            params.maxLineGap
        )

        val leftCandidates = mutableListOf<LineCandidate>()
        val rightCandidates = mutableListOf<LineCandidate>()

        for (index in 0 until lineSegments.rows()) {
            val values = lineSegments.get(index, 0) ?: continue
            if (values.size < 4) {
                continue
            }

            val x1 = values[0]
            val y1 = values[1]
            val x2 = values[2]
            val y2 = values[3]
            val dx = x2 - x1
            val dy = y2 - y1

            if (abs(dx) < 1.0) {
                continue
            }

            val slope = dy / dx
            val absSlope = abs(slope)
            if (absSlope < params.minSlopeAbs || absSlope > params.maxSlopeAbs) {
                continue
            }

            val midpointX = (x1 + x2) / 2.0
            val midpointY = (y1 + y2) / 2.0
            if (midpointY < topY) {
                continue
            }

            val intercept = y1 - slope * x1
            val length = hypot(dx, dy)
            val bottomWeight = ((max(y1, y2) - topY) / verticalSpan).coerceIn(0.0, 1.0)
            val weight = length * (0.65 + 0.35 * bottomWeight)

            if (slope < 0.0 && midpointX < width * 0.97) {
                leftCandidates += LineCandidate(slope, intercept, weight)
            } else if (slope > 0.0 && midpointX > width * 0.03) {
                rightCandidates += LineCandidate(slope, intercept, weight)
            }
        }

        var leftLine = buildAverageLine(leftCandidates, bottomY, topY, width, LaneSide.LEFT)
        var rightLine = buildAverageLine(rightCandidates, bottomY, topY, width, LaneSide.RIGHT)

        if (leftLine != null && rightLine != null && leftLine.bottom.x >= rightLine.bottom.x) {
            if (leftLine.score >= rightLine.score) {
                rightLine = estimateParallelLine(leftLine, LaneSide.RIGHT, width, bottomY, topY)
            } else {
                leftLine = estimateParallelLine(rightLine, LaneSide.LEFT, width, bottomY, topY)
            }
        }

        if (leftLine != null && rightLine != null && !leftLine.estimated && !rightLine.estimated) {
            lastKnownBottomLaneWidthPx = (rightLine.bottom.x - leftLine.bottom.x).coerceAtLeast(1.0)
            lastKnownTopLaneWidthPx = (rightLine.top.x - leftLine.top.x).coerceAtLeast(1.0)
        } else if (leftLine == null && rightLine != null) {
            leftLine = estimateParallelLine(rightLine, LaneSide.LEFT, width, bottomY, topY)
        } else if (rightLine == null && leftLine != null) {
            rightLine = estimateParallelLine(leftLine, LaneSide.RIGHT, width, bottomY, topY)
        }

        val centerBottom = if (leftLine != null && rightLine != null) {
            Point((leftLine.bottom.x + rightLine.bottom.x) / 2.0, bottomY)
        } else {
            null
        }

        val centerTop = if (leftLine != null && rightLine != null) {
            Point((leftLine.top.x + rightLine.top.x) / 2.0, topY)
        } else {
            null
        }

        val usedEstimatedLane = (leftLine?.estimated == true) || (rightLine?.estimated == true)
        val confidence = computeConfidence(
            leftLine = leftLine,
            rightLine = rightLine,
            leftCandidates = leftCandidates.size,
            rightCandidates = rightCandidates.size,
            imageWidth = width
        )

        return LaneDetectionResult(
            leftLine = leftLine,
            rightLine = rightLine,
            centerBottom = centerBottom,
            centerTop = centerTop,
            confidence = confidence,
            roiPolygon = roiPolygon,
            usedEstimatedLane = usedEstimatedLane
        )
    }

    fun release() {
        gray.release()
        hsv.release()
        blurred.release()
        brightMask.release()
        whiteMask.release()
        laneMask.release()
        edges.release()
        roiMask.release()
        maskedLaneMask.release()
        maskedEdges.release()
        lineSegments.release()
    }

    fun updateParams(newParams: LaneDetectorParams) {
        params = newParams
    }

    fun renderMaskedPreview(targetRgba: Mat) {
        if (maskedLaneMask.empty()) {
            return
        }
        Imgproc.cvtColor(maskedLaneMask, targetRgba, Imgproc.COLOR_GRAY2RGBA)
    }

    private fun buildRoiPolygon(width: Int, height: Int): Array<Point> {
        val params = params
        val bottomY = (height * params.roiBottomRatio).coerceIn(0.0, (height - 1).toDouble())
        val topY = height * params.roiTopRatio
        val centerX = width * params.roiCenterXRatio
        val topHalfWidthPx = width * params.roiTopHalfWidthRatio
        val bottomHalfWidthPx = width * params.roiBottomHalfWidthRatio
        return arrayOf(
            Point((centerX - bottomHalfWidthPx).coerceIn(0.0, (width - 1).toDouble()), bottomY),
            Point((centerX - topHalfWidthPx).coerceIn(0.0, (width - 1).toDouble()), topY),
            Point((centerX + topHalfWidthPx).coerceIn(0.0, (width - 1).toDouble()), topY),
            Point((centerX + bottomHalfWidthPx).coerceIn(0.0, (width - 1).toDouble()), bottomY)
        )
    }

    private fun applyRoiMask(source: Mat, roiPolygon: Array<Point>, destination: Mat) {
        roiMask.create(source.size(), CvType.CV_8UC1)
        roiMask.setTo(Scalar(0.0))

        val roiPoints = MatOfPoint(*roiPolygon)
        Imgproc.fillPoly(roiMask, listOf(roiPoints), Scalar(255.0))
        Core.bitwise_and(source, roiMask, destination)
        roiPoints.release()
    }

    private fun buildAverageLine(
        candidates: List<LineCandidate>,
        bottomY: Double,
        topY: Double,
        imageWidth: Int,
        laneSide: LaneSide
    ): RoadLine? {
        if (candidates.isEmpty()) {
            return null
        }

        var sumWeight = 0.0
        var slopeSum = 0.0
        var interceptSum = 0.0

        for (candidate in candidates) {
            sumWeight += candidate.weight
            slopeSum += candidate.slope * candidate.weight
            interceptSum += candidate.intercept * candidate.weight
        }

        if (sumWeight <= 0.0) {
            return null
        }

        val slope = slopeSum / sumWeight
        if (abs(slope) < 1e-3) {
            return null
        }

        val intercept = interceptSum / sumWeight
        val bottomX = ((bottomY - intercept) / slope).coerceIn(0.0, (imageWidth - 1).toDouble())
        val topX = ((topY - intercept) / slope).coerceIn(0.0, (imageWidth - 1).toDouble())

        if (laneSide == LaneSide.LEFT && bottomX > imageWidth * 0.92) {
            return null
        }
        if (laneSide == LaneSide.RIGHT && bottomX < imageWidth * 0.08) {
            return null
        }

        return RoadLine(
            bottom = Point(bottomX, bottomY),
            top = Point(topX, topY),
            slope = slope,
            intercept = intercept,
            score = sumWeight / candidates.size,
            estimated = false
        )
    }

    private fun estimateParallelLine(
        referenceLine: RoadLine,
        missingSide: LaneSide,
        imageWidth: Int,
        bottomY: Double,
        topY: Double
    ): RoadLine {
        val bottomLaneWidth = (lastKnownBottomLaneWidthPx ?: (imageWidth * params.defaultLaneWidthBottomRatio))
            .coerceIn(imageWidth * 0.20, imageWidth * 0.70)
        val topLaneWidth = (lastKnownTopLaneWidthPx ?: (imageWidth * params.defaultLaneWidthTopRatio))
            .coerceIn(imageWidth * 0.08, imageWidth * 0.45)

        val shiftDirection = if (missingSide == LaneSide.RIGHT) 1.0 else -1.0
        val estimatedBottom = Point(
            (referenceLine.bottom.x + shiftDirection * bottomLaneWidth).coerceIn(0.0, imageWidth - 1.0),
            bottomY
        )
        val estimatedTop = Point(
            (referenceLine.top.x + shiftDirection * topLaneWidth).coerceIn(0.0, imageWidth - 1.0),
            topY
        )

        val dx = estimatedTop.x - estimatedBottom.x
        val dy = estimatedTop.y - estimatedBottom.y
        val safeDx = if (abs(dx) < 1e-3) {
            if (dx >= 0.0) 1e-3 else -1e-3
        } else {
            dx
        }
        val slope = dy / safeDx
        val intercept = estimatedBottom.y - slope * estimatedBottom.x

        return RoadLine(
            bottom = estimatedBottom,
            top = estimatedTop,
            slope = slope,
            intercept = intercept,
            score = referenceLine.score * 0.55,
            estimated = true
        )
    }

    private fun computeConfidence(
        leftLine: RoadLine?,
        rightLine: RoadLine?,
        leftCandidates: Int,
        rightCandidates: Int,
        imageWidth: Int
    ): Int {
        if (leftLine == null && rightLine == null) {
            return 0
        }

        val directLines = listOfNotNull(leftLine, rightLine).count { !it.estimated }
        val candidateBoost = min(leftCandidates + rightCandidates, 8) * 3
        val geometryBoost = if (leftLine != null && rightLine != null) {
            val laneWidthRatio = abs(rightLine.bottom.x - leftLine.bottom.x) / imageWidth
            if (laneWidthRatio in 0.20..0.75) 12 else 4
        } else {
            0
        }

        val base = when (directLines) {
            2 -> 65
            1 -> 45
            else -> 25
        }

        val estimationPenalty = listOfNotNull(leftLine, rightLine).count { it.estimated } * 10

        return (base + candidateBoost + geometryBoost - estimationPenalty).coerceIn(0, 100)
    }
}
