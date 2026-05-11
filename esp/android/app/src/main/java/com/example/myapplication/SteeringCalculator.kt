package com.example.myapplication

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.roundToInt

enum class DirectionCommand {
    LEFT,
    RIGHT,
    STRAIGHT,
    STOP
}

data class SteeringDecision(
    val angleDegrees: Double,
    val steeringPercent: Int,
    val confidence: Int,
    val command: DirectionCommand,
    val lateralErrorPixels: Double,
    val usedFallback: Boolean
)

class SteeringCalculator(
    private val angleMaxDegrees: Double = 45.0,
    private val smoothingAlpha: Double = 0.25,
    private val lateralCorrectionMaxDegrees: Double = 12.0,
    private val holdLastFrames: Int = 5
) {

    private var smoothedAngleDegrees = 0.0
    private var smoothedSteeringPercent = 0.0
    private var lastStableDecision: SteeringDecision? = null
    private var missingFrameCount = 0

    fun compute(detection: LaneDetectionResult, imageWidth: Int): SteeringDecision {
        val centerBottom = detection.centerBottom
        val centerTop = detection.centerTop
        val cameraCenterX = imageWidth / 2.0

        if (centerBottom == null || centerTop == null) {
            missingFrameCount += 1
            val fallback = lastStableDecision
            if (fallback != null && missingFrameCount <= holdLastFrames) {
                return fallback.copy(
                    confidence = (fallback.confidence - missingFrameCount * 15).coerceAtLeast(10),
                    usedFallback = true
                )
            }

            return SteeringDecision(
                angleDegrees = 0.0,
                steeringPercent = 0,
                confidence = 0,
                command = DirectionCommand.STOP,
                lateralErrorPixels = 0.0,
                usedFallback = true
            )
        }

        missingFrameCount = 0

        val lateralErrorPixels = centerBottom.x - cameraCenterX
        val lateralErrorNormalized = (lateralErrorPixels / (imageWidth / 2.0)).coerceIn(-1.0, 1.0)

        val trajectoryAngleDegrees = Math.toDegrees(
            atan2(
                centerTop.x - centerBottom.x,
                centerBottom.y - centerTop.y
            )
        )
        val lateralCorrectionDegrees = lateralErrorNormalized * lateralCorrectionMaxDegrees
        val combinedAngleDegrees = trajectoryAngleDegrees + lateralCorrectionDegrees

        smoothedAngleDegrees = smooth(smoothedAngleDegrees, combinedAngleDegrees)
        val rawSteeringPercent = clampToRange(combinedAngleDegrees / angleMaxDegrees * 100.0, -100.0, 100.0)
        smoothedSteeringPercent = smooth(smoothedSteeringPercent, rawSteeringPercent)

        val steeringPercent = smoothedSteeringPercent.roundToInt().coerceIn(-100, 100)
        val command = when {
            detection.confidence <= 0 -> DirectionCommand.STOP
            abs(steeringPercent) < 10 -> DirectionCommand.STRAIGHT
            steeringPercent < 0 -> DirectionCommand.LEFT
            else -> DirectionCommand.RIGHT
        }

        return SteeringDecision(
            angleDegrees = smoothedAngleDegrees,
            steeringPercent = steeringPercent,
            confidence = detection.confidence,
            command = command,
            lateralErrorPixels = lateralErrorPixels,
            usedFallback = detection.usedEstimatedLane
        ).also { decision ->
            if (decision.confidence > 0) {
                lastStableDecision = decision
            }
        }
    }

    private fun smooth(previous: Double, current: Double): Double {
        return previous * (1.0 - smoothingAlpha) + current * smoothingAlpha
    }

    private fun clampToRange(value: Double, minValue: Double, maxValue: Double): Double {
        return value.coerceIn(minValue, maxValue)
    }
}
