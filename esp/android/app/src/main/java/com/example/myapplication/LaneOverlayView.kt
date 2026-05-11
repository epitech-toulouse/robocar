package com.example.myapplication

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.util.AttributeSet
import android.view.View
import kotlin.math.max
import kotlin.math.min
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

class LaneOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val roiPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.LTGRAY
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }
    private val leftPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.YELLOW
        style = Paint.Style.STROKE
        strokeWidth = 8f
    }
    private val rightPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.CYAN
        style = Paint.Style.STROKE
        strokeWidth = 8f
    }
    private val estimatedPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.MAGENTA
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }
    private val centerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        style = Paint.Style.FILL
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 34f
    }
    private val trajectoryPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }
    private val connectorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.MAGENTA
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }
    private val stopBoxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 8f
    }
    private val stopFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(130, 220, 20, 20)
        style = Paint.Style.FILL
    }

    private var detection: LaneDetectionResult? = null
    private var steering: SteeringDecision? = null
    private var stopSign: StopSignDetectionResult? = null
    private var sourceWidth = 0
    private var sourceHeight = 0

    fun update(
        detection: LaneDetectionResult,
        steering: SteeringDecision,
        stopSign: StopSignDetectionResult,
        sourceWidth: Int,
        sourceHeight: Int
    ) {
        this.detection = detection
        this.steering = steering
        this.stopSign = stopSign
        this.sourceWidth = sourceWidth
        this.sourceHeight = sourceHeight
        postInvalidateOnAnimation()
    }

    fun clear() {
        detection = null
        steering = null
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val detection = detection ?: return
        val steering = steering ?: return
        if (sourceWidth <= 0 || sourceHeight <= 0) return

        val imageCenter = mapPoint(Point(sourceWidth / 2.0, sourceHeight.toDouble() - 1.0))
        drawRoi(canvas, detection.roiPolygon)
        drawRoadLine(canvas, detection.leftLine, leftPaint)
        drawRoadLine(canvas, detection.rightLine, rightPaint)

        canvas.drawCircle(imageCenter.x, imageCenter.y, 10f, textPaint)

        detection.centerBottom?.let {
            val p = mapPoint(it)
            canvas.drawCircle(p.x, p.y, 12f, centerPaint)
            canvas.drawLine(imageCenter.x, imageCenter.y, p.x, p.y, connectorPaint)
        }
        if (detection.centerBottom != null && detection.centerTop != null) {
            val p1 = mapPoint(detection.centerBottom!!)
            val p2 = mapPoint(detection.centerTop!!)
            canvas.drawCircle(p2.x, p2.y, 10f, centerPaint)
            canvas.drawLine(p1.x, p1.y, p2.x, p2.y, trajectoryPaint)
        }
        drawStopSign(canvas, stopSign)

        val lines = listOf(
            "Angle: ${"%.1f".format(steering.angleDegrees)} deg",
            "Steering: ${steering.steeringPercent}%",
            "Confidence: ${steering.confidence}%",
            "Command: ${steering.command.name}",
            "Offset: ${"%.0f".format(steering.lateralErrorPixels)} px",
            "Stop: ${if (stopSign?.detected == true) "YES ${stopSign?.confidence}%" else "NO"}"
        )
        lines.forEachIndexed { index, text ->
            canvas.drawText(text, 24f, 42f + index * 40f, textPaint)
        }
    }

    fun drawOntoMat(
        frame: Mat,
        detection: LaneDetectionResult,
        steering: SteeringDecision,
        stopSign: StopSignDetectionResult
    ) {
        Imgproc.polylines(
            frame,
            listOf(org.opencv.core.MatOfPoint(*detection.roiPolygon)),
            true,
            Scalar(120.0, 120.0, 120.0, 255.0),
            2
        )
        drawMatLine(frame, detection.leftLine, Scalar(0.0, 255.0, 255.0, 255.0))
        drawMatLine(frame, detection.rightLine, Scalar(255.0, 255.0, 0.0, 255.0))

        val imageCenter = Point(frame.width() / 2.0, frame.height().toDouble() - 1.0)
        Imgproc.circle(frame, imageCenter, 8, Scalar(255.0, 255.0, 255.0, 255.0), -1)
        detection.centerBottom?.let {
            Imgproc.circle(frame, it, 10, Scalar(0.0, 255.0, 0.0, 255.0), -1)
            Imgproc.line(frame, imageCenter, it, Scalar(255.0, 0.0, 255.0, 255.0), 2)
        }
        if (detection.centerBottom != null && detection.centerTop != null) {
            Imgproc.circle(frame, detection.centerTop, 8, Scalar(0.0, 165.0, 255.0, 255.0), -1)
            Imgproc.arrowedLine(
                frame,
                detection.centerBottom,
                detection.centerTop,
                Scalar(0.0, 255.0, 0.0, 255.0),
                3
            )
        }
        drawMatStopSign(frame, stopSign)

        val lines = listOf(
            "Angle: ${"%.1f".format(steering.angleDegrees)} deg",
            "Steering: ${steering.steeringPercent}%",
            "Confidence: ${steering.confidence}%",
            "Command: ${steering.command.name}",
            "Offset: ${"%.0f".format(steering.lateralErrorPixels)} px",
            "Stop: ${if (stopSign.detected) "YES ${stopSign.confidence}%" else "NO"}"
        )
        lines.forEachIndexed { index, text ->
            Imgproc.putText(
                frame,
                text,
                Point(24.0, 36.0 + index * 32.0),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.8,
                Scalar(255.0, 255.0, 255.0, 255.0),
                2
            )
        }
    }

    private fun drawRoi(canvas: Canvas, polygon: Array<Point>) {
        for (i in polygon.indices) {
            val p1 = mapPoint(polygon[i])
            val p2 = mapPoint(polygon[(i + 1) % polygon.size])
            canvas.drawLine(p1.x, p1.y, p2.x, p2.y, roiPaint)
        }
    }

    private fun drawRoadLine(canvas: Canvas, roadLine: RoadLine?, basePaint: Paint) {
        if (roadLine == null) return
        val p1 = mapPoint(roadLine.bottom)
        val p2 = mapPoint(roadLine.top)
        canvas.drawLine(p1.x, p1.y, p2.x, p2.y, if (roadLine.estimated) estimatedPaint else basePaint)
    }

    private fun drawStopSign(canvas: Canvas, stopSign: StopSignDetectionResult?) {
        val rect = stopSign?.boundingRect ?: return
        if (!stopSign.detected) return

        val tl = mapPoint(Point(rect.x.toDouble(), rect.y.toDouble()))
        val br = mapPoint(Point((rect.x + rect.width).toDouble(), (rect.y + rect.height).toDouble()))
        canvas.drawRect(tl.x, tl.y, br.x, br.y, stopFillPaint)
        canvas.drawRect(tl.x, tl.y, br.x, br.y, stopBoxPaint)
        canvas.drawText("STOP", tl.x, tl.y - 12f, textPaint)
    }

    private fun drawMatLine(frame: Mat, roadLine: RoadLine?, color: Scalar) {
        if (roadLine == null) return
        Imgproc.line(
            frame,
            roadLine.bottom,
            roadLine.top,
            if (roadLine.estimated) Scalar(255.0, 0.0, 255.0, 255.0) else color,
            if (roadLine.estimated) 3 else 5
        )
    }

    private fun drawMatStopSign(frame: Mat, stopSign: StopSignDetectionResult) {
        val rect = stopSign.boundingRect ?: return
        if (!stopSign.detected) return
        Imgproc.rectangle(
            frame,
            Point(rect.x.toDouble(), rect.y.toDouble()),
            Point((rect.x + rect.width).toDouble(), (rect.y + rect.height).toDouble()),
            Scalar(0.0, 0.0, 255.0, 255.0),
            4
        )
        Imgproc.putText(
            frame,
            "STOP ${stopSign.confidence}%",
            Point(rect.x.toDouble(), max(24.0, rect.y - 10.0)),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            0.9,
            Scalar(255.0, 255.0, 255.0, 255.0),
            2
        )
    }

    private fun mapPoint(point: Point): PointF {
        val scale = min(width / sourceWidth.toFloat(), height / sourceHeight.toFloat())
        val dx = (width - sourceWidth * scale) / 2f
        val dy = (height - sourceHeight * scale) / 2f
        return PointF(dx + point.x.toFloat() * scale, dy + point.y.toFloat() * scale)
    }
}
