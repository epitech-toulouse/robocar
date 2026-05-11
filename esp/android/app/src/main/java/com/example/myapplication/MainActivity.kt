package com.example.myapplication

import android.Manifest
import android.app.AlertDialog
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.View
import android.view.WindowManager
import android.widget.EditText
import android.widget.SeekBar
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.Observer
import com.example.myapplication.bluetooth.BleClient
import com.example.myapplication.bluetooth.BleClientProvider
import com.example.myapplication.databinding.ActivityMainBinding
import java.util.ArrayDeque
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

class MainActivity : AppCompatActivity() {

    companion object {
        private const val PRESET_PREFS = "opencv_presets"
        private const val PRESET_NAMES_KEY = "__names__"
    }

    private enum class StopProtocolState {
        IDLE,
        STOP_HOLD,
        STOP_COOLDOWN
    }

    private enum class DisplayMode {
        NORMAL,
        MASKED
    }

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService

    private lateinit var laneDetector: LaneDetector
    private lateinit var steeringCalculator: SteeringCalculator
    private lateinit var stopSignDetector: StopSignDetector
    private lateinit var bleClient: BleClient

    private var openCvReady = false
    private var lastStatusUpdateMs = 0L
    private var lastBleSendMs = 0L
    private var isSyncingControls = false
    private var isSyncingZoomControls = false
    private var laneParams = LaneDetectorParams()
    private var stopSignParams = StopSignDetectorParams()
    private var imageShiftRatio = 0.0
    private var displayMode = DisplayMode.NORMAL
    private var wideAngleAvailable = false
    private var cameraProvider: ProcessCameraProvider? = null
    private var boundCamera: Camera? = null
    private var minAvailableZoomRatio = 1f
    private var maxAvailableZoomRatio = 1f
    private var desiredZoomRatio = 1f
    private var zoomInitialized = false
    private var activeCameraId = "?"
    private var physicalCameraCount = 0
    private var tuningPanelVisible = true
    private val steeringHistory = ArrayDeque<Float>()
    private var lastBleSendOk: Boolean? = null
    private var lastBlePayload = ""
    private var stopProtocolState = StopProtocolState.IDLE
    private var stopHoldUntilMs = 0L
    private var stopCooldownUntilMs = 0L
    private lateinit var shiftedFrame: Mat

    private val zoomStateObserver = Observer<androidx.camera.core.ZoomState> {
        updateWideAngleAvailability(boundCamera)
    }

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startCameraIfReady()
            } else {
                binding.statusText.text = getString(R.string.camera_permission_required)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (!packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY)) {
            binding.statusText.text = getString(R.string.camera_unavailable)
            return
        }

        bleClient = BleClientProvider.get(this)
        cameraExecutor = Executors.newSingleThreadExecutor()
        binding.previewView.implementationMode = androidx.camera.view.PreviewView.ImplementationMode.COMPATIBLE
        binding.previewView.scaleType = androidx.camera.view.PreviewView.ScaleType.FIT_CENTER
        binding.processedImageView.scaleType = android.widget.ImageView.ScaleType.FIT_CENTER
        applyVisualImageShift()

        binding.buttonToggleMask.setOnClickListener {
            displayMode = if (displayMode == DisplayMode.NORMAL) {
                DisplayMode.MASKED
            } else {
                DisplayMode.NORMAL
            }
            updateDisplayModeButton()
            updatePreviewMode()
        }
        binding.buttonToggleLens.setOnClickListener {
            desiredZoomRatio = 1.0f.coerceIn(minAvailableZoomRatio, maxAvailableZoomRatio)
            applyZoomRatio()
        }
        binding.buttonBackBluetooth.setOnClickListener {
            finish()
        }
        binding.buttonOpenCommandCenter.setOnClickListener {
            startActivity(Intent(this, CommandActivity::class.java))
        }

        setupZoomControls()
        updateDisplayModeButton()
        updateLensButton()
        updatePreviewMode()
        setupPanelControls()
        setupTuningControls()
        initCameraProvider()
        ensureCameraPermission()
    }

    override fun onResume() {
        super.onResume()
        openCvReady = OpenCVLoader.initLocal()
        if (!openCvReady) {
            binding.statusText.text = getString(R.string.opencv_init_failed)
            return
        }
        if (!::laneDetector.isInitialized) {
            laneDetector = LaneDetector(laneParams)
        } else {
            laneDetector.updateParams(laneParams)
        }
        if (!::steeringCalculator.isInitialized) {
            steeringCalculator = SteeringCalculator()
        }
        if (!::stopSignDetector.isInitialized) {
            stopSignDetector = StopSignDetector(stopSignParams)
        } else {
            stopSignDetector.updateParams(stopSignParams)
        }
        if (!::shiftedFrame.isInitialized) {
            shiftedFrame = Mat()
        }
        startCameraIfReady()
    }

    override fun onPause() {
        super.onPause()
        stopCamera()
    }

    override fun onDestroy() {
        stopCamera()
        if (::laneDetector.isInitialized) {
            laneDetector.release()
        }
        if (::stopSignDetector.isInitialized) {
            stopSignDetector.release()
        }
        if (::shiftedFrame.isInitialized) {
            shiftedFrame.release()
        }
        cameraExecutor.shutdown()
        super.onDestroy()
    }

    private fun initCameraProvider() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener(
            {
                cameraProvider = providerFuture.get()
                updateWideAngleAvailability()
                startCameraIfReady()
            },
            ContextCompat.getMainExecutor(this)
        )
    }

    private fun ensureCameraPermission() {
        if (hasCameraPermission()) {
            startCameraIfReady()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
    }

    private fun startCameraIfReady() {
        if (!openCvReady || !hasCameraPermission()) {
            return
        }
        val provider = cameraProvider ?: return

        val preview = Preview.Builder()
            .setTargetResolution(Size(1280, 720))
            .build()
            .also { it.setSurfaceProvider(binding.previewView.surfaceProvider) }

        val analyzer = ImageAnalysis.Builder()
            .setTargetResolution(Size(1280, 720))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    analyzeFrame(imageProxy)
                }
            }

        provider.unbindAll()
        boundCamera = provider.bindToLifecycle(
            this,
            CameraSelector.DEFAULT_BACK_CAMERA,
            preview,
            analyzer
        )
        boundCamera?.cameraInfo?.zoomState?.removeObservers(this)
        boundCamera?.cameraInfo?.zoomState?.observe(this, zoomStateObserver)
        updateWideAngleAvailability(boundCamera)
        applyZoomRatio()
    }

    private fun stopCamera() {
        boundCamera?.cameraInfo?.zoomState?.removeObservers(this)
        cameraProvider?.unbindAll()
        boundCamera = null
    }

    private fun updateWideAngleAvailability(camera: Camera? = boundCamera) {
        val bound = camera
        if (bound == null) {
            minAvailableZoomRatio = 1f
            maxAvailableZoomRatio = 1f
            activeCameraId = "?"
            physicalCameraCount = 0
            wideAngleAvailable = false
            updateLensButton()
            return
        }

        val info = Camera2CameraInfo.from(bound.cameraInfo)
        val zoomState = bound.cameraInfo.zoomState.value
        val zoomRange = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            info.getCameraCharacteristic(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE)
        } else {
            null
        }

        activeCameraId = info.cameraId
        physicalCameraCount = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
            val cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
            cameraManager.getCameraCharacteristics(activeCameraId).physicalCameraIds.size
        } else {
            0
        }

        minAvailableZoomRatio = min(
            zoomState?.minZoomRatio ?: 1f,
            zoomRange?.lower ?: 1f
        )
        maxAvailableZoomRatio = max(
            zoomState?.maxZoomRatio ?: 1f,
            zoomRange?.upper ?: 1f
        )
        wideAngleAvailable = minAvailableZoomRatio < 0.99f

        if (!zoomInitialized) {
            desiredZoomRatio = minAvailableZoomRatio
            zoomInitialized = true
        } else {
            desiredZoomRatio = desiredZoomRatio.coerceIn(minAvailableZoomRatio, maxAvailableZoomRatio)
        }

        Log.d(
            "MainActivity",
            "Camera $activeCameraId zoomRange=${"%.2f".format(minAvailableZoomRatio)}x..${"%.2f".format(maxAvailableZoomRatio)}x physicalCount=$physicalCameraCount"
        )

        syncZoomControls()
        updateLensButton()
    }

    private fun applyZoomRatio() {
        val camera = boundCamera ?: return
        val clampedZoom = desiredZoomRatio.coerceIn(effectiveZoomMin(), effectiveZoomMax())
        desiredZoomRatio = clampedZoom
        camera.cameraControl.setZoomRatio(clampedZoom)
        syncZoomControls()
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        if (!::laneDetector.isInitialized || !::steeringCalculator.isInitialized) {
            imageProxy.close()
            return
        }

        val rgbaFrame = imageProxy.toRgbaMat()
        applyImageShift(rgbaFrame)
        val detection = laneDetector.detect(rgbaFrame)
        val steering = steeringCalculator.compute(detection, rgbaFrame.width())
        val stopSign = stopSignDetector.detect(rgbaFrame)
        recordSteeringForBle(steering.steeringPercent.toFloat())
        updateStopProtocol(stopSign)
        maybeSendBleSteering()

        if (displayMode == DisplayMode.MASKED) {
            laneDetector.renderMaskedPreview(rgbaFrame)
            drawDebugOverlayOnMat(rgbaFrame, detection, steering, stopSign)
            val bitmap = rgbaFrame.toBitmap()
            runOnUiThread {
                binding.processedImageView.setImageBitmap(bitmap)
                binding.laneOverlayView.clear()
            }
        } else {
            runOnUiThread {
                binding.laneOverlayView.update(
                    detection,
                    steering,
                    stopSign,
                    rgbaFrame.width(),
                    rgbaFrame.height()
                )
            }
        }

        maybeUpdateStatusPanel(steering, stopSign)
        rgbaFrame.release()
        imageProxy.close()
    }

    private fun recordSteeringForBle(steeringPercent: Float) {
        if (steeringHistory.size >= 10) {
            steeringHistory.removeFirst()
        }
        steeringHistory.addLast(steeringPercent.coerceIn(-100f, 100f))
    }

    private fun maybeSendBleSteering() {
        val now = SystemClock.elapsedRealtime()
        if (stopProtocolState == StopProtocolState.STOP_HOLD) {
            return
        }
        if (now - lastBleSendMs < 100L || steeringHistory.isEmpty()) {
            return
        }
        lastBleSendMs = now

        val averageSteering = steeringHistory.average().toFloat()
        lastBlePayload = "STEER:${"%.1f".format(java.util.Locale.US, averageSteering)}"
        lastBleSendOk = bleClient.sendSteeringFrame(
            steeringPercent = averageSteering,
            weight = 1.0f
        )
    }

    private fun updateStopProtocol(stopSign: StopSignDetectionResult) {
        val now = SystemClock.elapsedRealtime()

        when (stopProtocolState) {
            StopProtocolState.STOP_HOLD -> {
                if (now >= stopHoldUntilMs) {
                    lastBlePayload = "GO"
                    lastBleSendOk = bleClient.sendGoCommand()
                    stopProtocolState = StopProtocolState.STOP_COOLDOWN
                    stopCooldownUntilMs = now + 3_000L
                }
                return
            }

            StopProtocolState.STOP_COOLDOWN -> {
                if (now >= stopCooldownUntilMs) {
                    stopProtocolState = StopProtocolState.IDLE
                } else {
                    return
                }
            }

            StopProtocolState.IDLE -> Unit
        }

        if (stopSign.detected) {
            lastBlePayload = "STOP"
            lastBleSendOk = bleClient.sendStopCommand()
            if (lastBleSendOk == true) {
                stopProtocolState = StopProtocolState.STOP_HOLD
                stopHoldUntilMs = now + 4_000L
                lastBleSendMs = now
            }
        }
    }

    private fun ImageProxy.toRgbaMat(): Mat {
        val nv21 = yuv420888ToNv21(this)
        val yuvMat = Mat(height + height / 2, width, CvType.CV_8UC1)
        yuvMat.put(0, 0, nv21)

        val rgba = Mat()
        Imgproc.cvtColor(yuvMat, rgba, Imgproc.COLOR_YUV2RGBA_NV21)
        yuvMat.release()

        return rotateMat(rgba, imageInfo.rotationDegrees)
    }

    private fun yuv420888ToNv21(image: ImageProxy): ByteArray {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)

        val chromaHeight = image.height / 2
        val chromaWidth = image.width / 2
        val uRowStride = image.planes[1].rowStride
        val vRowStride = image.planes[2].rowStride
        val uPixelStride = image.planes[1].pixelStride
        val vPixelStride = image.planes[2].pixelStride

        val uBytes = ByteArray(uSize)
        val vBytes = ByteArray(vSize)
        uBuffer.get(uBytes)
        vBuffer.get(vBytes)

        var offset = ySize
        for (row in 0 until chromaHeight) {
            for (col in 0 until chromaWidth) {
                nv21[offset++] = vBytes[row * vRowStride + col * vPixelStride]
                nv21[offset++] = uBytes[row * uRowStride + col * uPixelStride]
            }
        }

        return nv21
    }

    private fun rotateMat(source: Mat, rotationDegrees: Int): Mat {
        if (rotationDegrees == 0) {
            return source
        }

        val rotated = Mat()
        when (rotationDegrees) {
            90 -> Core.rotate(source, rotated, Core.ROTATE_90_CLOCKWISE)
            180 -> Core.rotate(source, rotated, Core.ROTATE_180)
            270 -> Core.rotate(source, rotated, Core.ROTATE_90_COUNTERCLOCKWISE)
            else -> {
                source.release()
                return rotated
            }
        }
        source.release()
        return rotated
    }

    private fun Mat.toBitmap(): Bitmap {
        val bitmap = Bitmap.createBitmap(cols(), rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(this, bitmap)
        return bitmap
    }

    private fun updateDisplayModeButton() {
        binding.buttonToggleMask.text = if (displayMode == DisplayMode.NORMAL) {
            getString(R.string.show_mask_view)
        } else {
            getString(R.string.show_normal_view)
        }
    }

    private fun updateLensButton() {
        binding.buttonToggleLens.text = "Reset zoom 1.0x"
        binding.buttonToggleLens.isEnabled = maxAvailableZoomRatio > minAvailableZoomRatio
    }

    private fun updatePreviewMode() {
        val normal = displayMode == DisplayMode.NORMAL
        binding.previewView.visibility = if (normal) View.VISIBLE else View.INVISIBLE
        binding.processedImageView.visibility = if (normal) View.GONE else View.VISIBLE
        binding.laneOverlayView.visibility = if (normal) View.VISIBLE else View.GONE
    }

    private fun setupPanelControls() {
        binding.buttonToggleTuningPanel.setOnClickListener {
            tuningPanelVisible = !tuningPanelVisible
            updatePanelVisibility()
        }
        binding.buttonSectionRoi.setOnClickListener {
            toggleSection(binding.sectionRoi)
        }
        binding.buttonSectionCamera.setOnClickListener {
            toggleSection(binding.sectionCamera)
        }
        binding.buttonSectionCanny.setOnClickListener {
            toggleSection(binding.sectionCanny)
        }
        binding.buttonSectionHough.setOnClickListener {
            toggleSection(binding.sectionHough)
        }
        binding.buttonSectionStop.setOnClickListener {
            toggleSection(binding.sectionStop)
        }
        updatePanelVisibility()
    }

    private fun setupZoomControls() {
        binding.seekZoomRatio.max = 200
        binding.seekZoomRatio.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (!fromUser || isSyncingZoomControls) {
                    return
                }
                val fraction = progress / binding.seekZoomRatio.max.toFloat()
                val zoomMin = effectiveZoomMin()
                val zoomMax = effectiveZoomMax()
                desiredZoomRatio = zoomMin + (zoomMax - zoomMin) * fraction
                applyZoomRatio()
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit

            override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
        })
        syncZoomControls()
    }

    private fun syncZoomControls() {
        isSyncingZoomControls = true
        val zoomMin = effectiveZoomMin()
        val zoomMax = effectiveZoomMax()
        val zoomRange = (zoomMax - zoomMin).coerceAtLeast(0.0001f)
        val progress = (((desiredZoomRatio - zoomMin) / zoomRange) * binding.seekZoomRatio.max)
            .roundToInt()
            .coerceIn(0, binding.seekZoomRatio.max)
        binding.seekZoomRatio.progress = progress
        binding.seekZoomRatio.isEnabled = zoomMax > zoomMin

        val zoomType = when {
            desiredZoomRatio < 1.0f -> "Grand angle"
            abs(desiredZoomRatio - 1.0f) < 0.05f -> "Optique normal"
            else -> "Zoom normal"
        }
        binding.textZoomRatio.text =
            "Zoom: ${"%.2f".format(desiredZoomRatio)}x\n$zoomType"
        isSyncingZoomControls = false
    }

    private fun effectiveZoomMin(): Float {
        return max(minAvailableZoomRatio, 0.5f)
    }

    private fun effectiveZoomMax(): Float {
        val maxTarget = min(maxAvailableZoomRatio, 2.0f)
        return if (maxTarget < effectiveZoomMin()) effectiveZoomMin() else maxTarget
    }

    private fun toggleSection(section: View) {
        section.visibility = if (section.visibility == View.VISIBLE) View.GONE else View.VISIBLE
    }

    private fun updatePanelVisibility() {
        binding.tuningPanel.visibility = if (tuningPanelVisible) View.VISIBLE else View.GONE
        binding.buttonToggleTuningPanel.text = if (tuningPanelVisible) {
            "Masquer reglages"
        } else {
            "Afficher reglages"
        }
    }

    private fun setupTuningControls() {
        configureSeekBar(
            seekBar = binding.seekImageShift,
            maxValue = 80,
            onChanged = { progress ->
                imageShiftRatio = (progress - 40) / 100.0
                applyVisualImageShift()
                syncControlsFromParams()
            }
        )
        configureSeekBar(
            seekBar = binding.seekCannyLow,
            maxValue = 200,
            onChanged = { progress ->
                var low = max(progress, 1)
                var high = laneParams.cannyHighThreshold.roundToInt()
                if (low >= high) {
                    high = min(low + 10, 300)
                    laneParams = laneParams.copy(cannyHighThreshold = high.toDouble())
                }
                laneParams = laneParams.copy(cannyLowThreshold = low.toDouble())
                syncControlsFromParams()
                applyLaneParams()
            }
        )
        configureSeekBar(
            seekBar = binding.seekCannyHigh,
            maxValue = 300,
            onChanged = { progress ->
                var high = max(progress, 10)
                var low = laneParams.cannyLowThreshold.roundToInt()
                if (high <= low) {
                    low = max(high - 10, 1)
                    laneParams = laneParams.copy(cannyLowThreshold = low.toDouble())
                }
                laneParams = laneParams.copy(cannyHighThreshold = high.toDouble())
                syncControlsFromParams()
                applyLaneParams()
            }
        )
        configureSeekBar(
            seekBar = binding.seekRoiTop,
            maxValue = 85,
            onChanged = { progress ->
                val top = 0.05 + progress / 100.0
                val bottom = max(laneParams.roiBottomRatio, top + 0.08)
                laneParams = laneParams.copy(
                    roiTopRatio = top,
                    roiBottomRatio = bottom.coerceAtMost(0.99)
                )
                syncControlsFromParams()
                applyLaneParams()
            }
        )
        configureSeekBar(
            seekBar = binding.seekRoiBottom,
            maxValue = 51,
            onChanged = { progress ->
                val bottom = 0.48 + progress / 100.0
                val top = min(laneParams.roiTopRatio, bottom - 0.08)
                laneParams = laneParams.copy(
                    roiTopRatio = top.coerceAtLeast(0.05),
                    roiBottomRatio = bottom.coerceAtMost(0.99)
                )
                syncControlsFromParams()
                applyLaneParams()
            }
        )
        configureSeekBar(
            seekBar = binding.seekRoiCenterX,
            maxValue = 80,
            onChanged = { progress ->
                laneParams = laneParams.copy(roiCenterXRatio = 0.10 + progress / 100.0)
                syncControlsFromParams()
                applyLaneParams()
            }
        )
        configureSeekBar(
            seekBar = binding.seekRoiTopWidth,
            maxValue = 40,
            onChanged = { progress ->
                laneParams = laneParams.copy(roiTopHalfWidthRatio = 0.05 + progress / 100.0)
                syncControlsFromParams()
                applyLaneParams()
            }
        )
        configureSeekBar(
            seekBar = binding.seekRoiBottomWidth,
            maxValue = 45,
            onChanged = { progress ->
                laneParams = laneParams.copy(roiBottomHalfWidthRatio = 0.05 + progress / 100.0)
                syncControlsFromParams()
                applyLaneParams()
            }
        )
        configureSeekBar(
            seekBar = binding.seekHoughThreshold,
            maxValue = 140,
            onChanged = { progress ->
                laneParams = laneParams.copy(houghThreshold = progress + 10)
                syncControlsFromParams()
                applyLaneParams()
            }
        )
        configureSeekBar(
            seekBar = binding.seekMinLineLength,
            maxValue = 190,
            onChanged = { progress ->
                laneParams = laneParams.copy(minLineLength = (progress + 10).toDouble())
                syncControlsFromParams()
                applyLaneParams()
            }
        )
        configureSeekBar(
            seekBar = binding.seekMaxLineGap,
            maxValue = 150,
            onChanged = { progress ->
                laneParams = laneParams.copy(maxLineGap = progress.toDouble())
                syncControlsFromParams()
                applyLaneParams()
            }
        )
        configureSeekBar(
            seekBar = binding.seekStopMinSize,
            maxValue = 97,
            onChanged = { progress ->
                val minRatio = 0.001 + progress / 1000.0
                val maxRatio = max(stopSignParams.maxAreaRatio, minRatio + 0.005)
                stopSignParams = stopSignParams.copy(
                    minAreaRatio = minRatio,
                    maxAreaRatio = maxRatio.coerceAtMost(0.60)
                )
                applyStopSignParams()
                syncControlsFromParams()
            }
        )
        configureSeekBar(
            seekBar = binding.seekStopMaxSize,
            maxValue = 55,
            onChanged = { progress ->
                val maxRatio = 0.05 + progress / 100.0
                val minRatio = min(stopSignParams.minAreaRatio, maxRatio - 0.005)
                stopSignParams = stopSignParams.copy(
                    minAreaRatio = minRatio.coerceAtLeast(0.001),
                    maxAreaRatio = maxRatio.coerceAtMost(0.60)
                )
                applyStopSignParams()
                syncControlsFromParams()
            }
        )

        binding.buttonResetTuning.setOnClickListener {
            laneParams = LaneDetectorParams()
            stopSignParams = StopSignDetectorParams()
            imageShiftRatio = 0.0
            applyVisualImageShift()
            syncControlsFromParams()
            applyLaneParams()
            applyStopSignParams()
        }
        binding.buttonSavePreset.setOnClickListener {
            showSavePresetDialog()
        }
        binding.buttonLoadPreset.setOnClickListener {
            showLoadPresetDialog()
        }

        syncControlsFromParams()
    }

    private fun configureSeekBar(
        seekBar: SeekBar,
        maxValue: Int,
        onChanged: (Int) -> Unit
    ) {
        seekBar.max = maxValue
        seekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (!fromUser || isSyncingControls) {
                    return
                }
                onChanged(progress)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit

            override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
        })
    }

    private fun syncControlsFromParams() {
        isSyncingControls = true

        binding.seekImageShift.progress = ((imageShiftRatio + 0.40) * 100.0).roundToInt().coerceIn(0, 80)
        binding.seekCannyLow.progress = laneParams.cannyLowThreshold.roundToInt().coerceIn(1, 200)
        binding.seekCannyHigh.progress = laneParams.cannyHighThreshold.roundToInt().coerceIn(10, 300)
        binding.seekRoiTop.progress = ((laneParams.roiTopRatio - 0.05) * 100.0).roundToInt().coerceIn(0, 85)
        binding.seekRoiBottom.progress = ((laneParams.roiBottomRatio - 0.48) * 100.0).roundToInt().coerceIn(0, 51)
        binding.seekRoiCenterX.progress =
            ((laneParams.roiCenterXRatio - 0.10) * 100.0).roundToInt().coerceIn(0, 80)
        binding.seekRoiTopWidth.progress =
            ((laneParams.roiTopHalfWidthRatio - 0.05) * 100.0).roundToInt().coerceIn(0, 40)
        binding.seekRoiBottomWidth.progress =
            ((laneParams.roiBottomHalfWidthRatio - 0.05) * 100.0).roundToInt().coerceIn(0, 45)
        binding.seekHoughThreshold.progress = (laneParams.houghThreshold - 10).coerceIn(0, 140)
        binding.seekMinLineLength.progress =
            (laneParams.minLineLength.roundToInt() - 10).coerceIn(0, 190)
        binding.seekMaxLineGap.progress = laneParams.maxLineGap.roundToInt().coerceIn(0, 150)
        binding.seekStopMinSize.progress =
            ((stopSignParams.minAreaRatio - 0.001) * 1000.0).roundToInt().coerceIn(0, 97)
        binding.seekStopMaxSize.progress =
            ((stopSignParams.maxAreaRatio - 0.05) * 100.0).roundToInt().coerceIn(0, 55)

        binding.textImageShift.text =
            "Decalage image: ${(imageShiftRatio * 100.0).roundToInt()} %"
        binding.textCannyLow.text = "Canny bas: ${laneParams.cannyLowThreshold.roundToInt()}"
        binding.textCannyHigh.text = "Canny haut: ${laneParams.cannyHighThreshold.roundToInt()}"
        binding.textRoiTop.text =
            "ROI haut: ${(laneParams.roiTopRatio * 100.0).roundToInt()} %"
        binding.textRoiBottom.text =
            "ROI bas: ${(laneParams.roiBottomRatio * 100.0).roundToInt()} %"
        binding.textRoiCenterX.text =
            "ROI centre X: ${(laneParams.roiCenterXRatio * 100.0).roundToInt()} %"
        binding.textRoiTopWidth.text =
            "ROI largeur haute: ${(laneParams.roiTopHalfWidthRatio * 200.0).roundToInt()} %"
        binding.textRoiBottomWidth.text =
            "ROI largeur basse: ${(laneParams.roiBottomHalfWidthRatio * 200.0).roundToInt()} %"
        binding.textHoughThreshold.text = "Hough seuil: ${laneParams.houghThreshold}"
        binding.textMinLineLength.text =
            "Hough longueur min: ${laneParams.minLineLength.roundToInt()} px"
        binding.textMaxLineGap.text =
            "Hough gap max: ${laneParams.maxLineGap.roundToInt()} px"
        binding.textStopMinSize.text =
            "STOP taille min: ${(stopSignParams.minAreaRatio * 100.0).formatPercent()} % image"
        binding.textStopMaxSize.text =
            "STOP taille max: ${(stopSignParams.maxAreaRatio * 100.0).formatPercent()} % image"

        isSyncingControls = false
    }

    private fun applyLaneParams() {
        if (::laneDetector.isInitialized) {
            laneDetector.updateParams(laneParams)
        }
    }

    private fun applyStopSignParams() {
        if (::stopSignDetector.isInitialized) {
            stopSignDetector.updateParams(stopSignParams)
        }
    }

    private fun applyVisualImageShift() {
        val applyShift = {
            val shiftPixels = binding.previewView.width * imageShiftRatio.toFloat()
            binding.previewView.translationX = shiftPixels
            binding.processedImageView.translationX = 0f
            binding.laneOverlayView.translationX = 0f
        }
        if (binding.previewView.width == 0) {
            binding.previewView.post { applyShift() }
        } else {
            applyShift()
        }
    }

    private fun applyImageShift(frame: Mat) {
        if (!::shiftedFrame.isInitialized || abs(imageShiftRatio) < 0.0001) {
            return
        }
        val dx = frame.width() * imageShiftRatio
        val transform = Mat(2, 3, CvType.CV_64F)
        transform.put(0, 0, 1.0, 0.0, dx, 0.0, 1.0, 0.0)
        Imgproc.warpAffine(
            frame,
            shiftedFrame,
            transform,
            frame.size(),
            Imgproc.INTER_LINEAR,
            Core.BORDER_REPLICATE,
            Scalar.all(0.0)
        )
        shiftedFrame.copyTo(frame)
        transform.release()
    }

    private fun showSavePresetDialog() {
        val input = EditText(this).apply {
            hint = "Nom du preset"
            setText("preset_${System.currentTimeMillis() % 100000}")
        }
        AlertDialog.Builder(this)
            .setTitle("Sauver preset")
            .setView(input)
            .setPositiveButton("Sauver") { _, _ ->
                val name = input.text?.toString()?.trim().orEmpty()
                if (name.isNotEmpty()) {
                    savePreset(name)
                }
            }
            .setNegativeButton("Annuler", null)
            .show()
    }

    private fun showLoadPresetDialog() {
        val names = getPresetNames()
        if (names.isEmpty()) {
            AlertDialog.Builder(this)
                .setTitle("Charger preset")
                .setMessage("Aucun preset sauvegarde.")
                .setPositiveButton("OK", null)
                .show()
            return
        }
        AlertDialog.Builder(this)
            .setTitle("Charger preset")
            .setItems(names.toTypedArray()) { _, which ->
                loadPreset(names[which])
            }
            .setNegativeButton("Annuler", null)
            .show()
    }

    private fun savePreset(name: String) {
        val prefs = getSharedPreferences(PRESET_PREFS, Context.MODE_PRIVATE)
        val preset = JSONObject().apply {
            put("imageShiftRatio", imageShiftRatio)
            put("desiredZoomRatio", desiredZoomRatio)
            put("roiTopRatio", laneParams.roiTopRatio)
            put("roiBottomRatio", laneParams.roiBottomRatio)
            put("roiCenterXRatio", laneParams.roiCenterXRatio)
            put("roiTopHalfWidthRatio", laneParams.roiTopHalfWidthRatio)
            put("roiBottomHalfWidthRatio", laneParams.roiBottomHalfWidthRatio)
            put("cannyLowThreshold", laneParams.cannyLowThreshold)
            put("cannyHighThreshold", laneParams.cannyHighThreshold)
            put("houghThreshold", laneParams.houghThreshold)
            put("minLineLength", laneParams.minLineLength)
            put("maxLineGap", laneParams.maxLineGap)
            put("stopMinAreaRatio", stopSignParams.minAreaRatio)
            put("stopMaxAreaRatio", stopSignParams.maxAreaRatio)
        }
        val names = getPresetNames().toMutableSet()
        names.add(name)
        prefs.edit()
            .putString(name, preset.toString())
            .putStringSet(PRESET_NAMES_KEY, names)
            .apply()
        binding.statusText.text = "Preset sauve: $name"
    }

    private fun loadPreset(name: String) {
        val prefs = getSharedPreferences(PRESET_PREFS, Context.MODE_PRIVATE)
        val raw = prefs.getString(name, null) ?: return
        val preset = JSONObject(raw)
        imageShiftRatio = preset.optDouble("imageShiftRatio", imageShiftRatio)
        desiredZoomRatio = preset.optDouble("desiredZoomRatio", desiredZoomRatio.toDouble()).toFloat()
        laneParams = laneParams.copy(
            roiTopRatio = preset.optDouble("roiTopRatio", laneParams.roiTopRatio),
            roiBottomRatio = preset.optDouble("roiBottomRatio", laneParams.roiBottomRatio),
            roiCenterXRatio = preset.optDouble("roiCenterXRatio", laneParams.roiCenterXRatio),
            roiTopHalfWidthRatio = preset.optDouble("roiTopHalfWidthRatio", laneParams.roiTopHalfWidthRatio),
            roiBottomHalfWidthRatio = preset.optDouble("roiBottomHalfWidthRatio", laneParams.roiBottomHalfWidthRatio),
            cannyLowThreshold = preset.optDouble("cannyLowThreshold", laneParams.cannyLowThreshold),
            cannyHighThreshold = preset.optDouble("cannyHighThreshold", laneParams.cannyHighThreshold),
            houghThreshold = preset.optInt("houghThreshold", laneParams.houghThreshold),
            minLineLength = preset.optDouble("minLineLength", laneParams.minLineLength),
            maxLineGap = preset.optDouble("maxLineGap", laneParams.maxLineGap)
        )
        stopSignParams = stopSignParams.copy(
            minAreaRatio = preset.optDouble("stopMinAreaRatio", stopSignParams.minAreaRatio),
            maxAreaRatio = preset.optDouble("stopMaxAreaRatio", stopSignParams.maxAreaRatio)
        )
        applyLaneParams()
        applyStopSignParams()
        applyVisualImageShift()
        applyZoomRatio()
        syncControlsFromParams()
        binding.statusText.text = "Preset charge: $name"
    }

    private fun getPresetNames(): List<String> {
        val prefs = getSharedPreferences(PRESET_PREFS, Context.MODE_PRIVATE)
        return prefs.getStringSet(PRESET_NAMES_KEY, emptySet())
            .orEmpty()
            .sorted()
    }

    private fun Double.formatPercent(): String {
        return String.format(java.util.Locale.US, "%.1f", this)
    }

    private fun maybeUpdateStatusPanel(
        steering: SteeringDecision,
        stopSign: StopSignDetectionResult
    ) {
        val now = SystemClock.elapsedRealtime()
        if (now - lastStatusUpdateMs < 150L) {
            return
        }
        lastStatusUpdateMs = now

        val summary = buildString {
            appendLine("Direction:")
            appendLine("angleDegrees: ${"%.1f".format(steering.angleDegrees)}")
            appendLine("steeringPercent: ${steering.steeringPercent}")
            appendLine("confidence: ${steering.confidence}")
            appendLine("command: ${steering.command.name}")
            appendLine("view: ${displayMode.name}")
            appendLine("zoom: ${"%.2f".format(boundCamera?.cameraInfo?.zoomState?.value?.zoomRatio ?: desiredZoomRatio)}x")
            appendLine("zoomRange: ${"%.2f".format(minAvailableZoomRatio)}x..${"%.2f".format(maxAvailableZoomRatio)}x")
            appendLine("cameraId: $activeCameraId / physical=$physicalCameraCount")
            appendLine("bleSend: ${when (lastBleSendOk) {
                true -> "OK"
                false -> "FAILED"
                null -> "WAITING"
            }}")
            appendLine("bleFrame: $lastBlePayload")
            appendLine("stopProto: $stopProtocolState")
            append("stopSign: ${if (stopSign.detected) "DETECTED ${stopSign.confidence}%" else "NONE"}")
        }

        runOnUiThread {
            binding.statusText.text = summary
        }
    }

    private fun drawDebugOverlayOnMat(
        frame: Mat,
        detection: LaneDetectionResult,
        steering: SteeringDecision,
        stopSign: StopSignDetectionResult
    ) {
        binding.laneOverlayView.drawOntoMat(frame, detection, steering, stopSign)
    }
}
