package com.example.myapplication

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.MotionEvent
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import com.example.myapplication.bluetooth.BleClient
import com.example.myapplication.bluetooth.BleClientProvider
import com.example.myapplication.databinding.ActivityCommandBinding

class CommandActivity : AppCompatActivity() {

    private lateinit var binding: ActivityCommandBinding
    private lateinit var bleClient: BleClient
    private val uiHandler = Handler(Looper.getMainLooper())

    private var lastBleSendOk: Boolean? = null
    private var lastBlePayload = ""

    private val blePanelRefresh = object : Runnable {
        override fun run() {
            updateBlePanel()
            uiHandler.postDelayed(this, 500L)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCommandBinding.inflate(layoutInflater)
        setContentView(binding.root)

        bleClient = BleClientProvider.get(this)

        binding.buttonBackHome.setOnClickListener {
            finish()
        }

        setupBleControls()
        updateBlePanel()
    }

    override fun onResume() {
        super.onResume()
        uiHandler.removeCallbacks(blePanelRefresh)
        uiHandler.post(blePanelRefresh)
    }

    override fun onPause() {
        super.onPause()
        uiHandler.removeCallbacks(blePanelRefresh)
    }

    private fun setupBleControls() {
        binding.buttonBleStatus.setOnClickListener {
            lastBlePayload = "STATUS?"
            lastBleSendOk = bleClient.requestStatus()
            updateBlePanel()
        }
        binding.buttonBleLogs.setOnClickListener {
            lastBlePayload = "LOGS:0"
            lastBleSendOk = bleClient.requestLogs(0)
            updateBlePanel()
        }
        binding.buttonBleArm.setOnClickListener {
            lastBlePayload = "A"
            lastBleSendOk = bleClient.sendProtocolChar('A')
            updateBlePanel()
        }
        binding.buttonBleStop.setOnClickListener {
            lastBlePayload = "S"
            lastBleSendOk = bleClient.sendProtocolChar('S')
            updateBlePanel()
        }
        binding.buttonApplyAlgorithms.setOnClickListener {
            val selected = mutableListOf<String>()
            if (binding.checkAlgoManual.isChecked) selected += "manual"
            if (binding.checkAlgoCloseObstacle.isChecked) selected += "close_obstacle"
            if (binding.checkAlgoLidarCorridor.isChecked) selected += "lidar_corridor"
            if (binding.checkAlgoGps.isChecked) selected += "gps"
            if (binding.checkAlgoCamera.isChecked) selected += "camera"
            lastBlePayload = "ALG:${selected.joinToString(",")}"
            lastBleSendOk = bleClient.sendAlgorithms(selected)
            updateBlePanel()
        }
        binding.buttonApplyGpsGoal.setOnClickListener {
            val lat = binding.editGpsLat.text?.toString().orEmpty()
            val lon = binding.editGpsLon.text?.toString().orEmpty()
            lastBlePayload = "GPS:${lat.trim()},${lon.trim()}"
            lastBleSendOk = bleClient.sendGpsGoal(lat, lon)
            updateBlePanel()
        }

        bindHoldButton(binding.buttonManualForward, 'F', 'f')
        bindHoldButton(binding.buttonManualBackward, 'B', 'b')
        bindHoldButton(binding.buttonManualLeft, 'L', 'l')
        bindHoldButton(binding.buttonManualRight, 'R', 'r')
    }

    private fun bindHoldButton(button: View, down: Char, up: Char) {
        var pressed = false
        button.setOnTouchListener { view, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    if (!pressed) {
                        pressed = true
                        view.isPressed = true
                        lastBlePayload = down.toString()
                        lastBleSendOk = bleClient.sendProtocolChar(down)
                        updateBlePanel()
                    }
                    true
                }

                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    if (pressed) {
                        pressed = false
                        view.isPressed = false
                        lastBlePayload = up.toString()
                        lastBleSendOk = bleClient.sendProtocolChar(up)
                        updateBlePanel()
                    }
                    if (event.actionMasked == MotionEvent.ACTION_UP) {
                        view.performClick()
                    }
                    true
                }

                else -> true
            }
        }
    }

    private fun updateBlePanel() {
        if (!::bleClient.isInitialized) {
            return
        }
        val sendState = when (lastBleSendOk) {
            true -> "OK"
            false -> "FAILED"
            null -> "WAITING"
        }
        binding.textBleStatus.text = buildString {
            appendLine("BLE: ${bleClient.connectionState.name}")
            appendLine(bleClient.statusText)
            appendLine("lastSend: $sendState $lastBlePayload")
        }.trimEnd()
        binding.textBleResponse.text = bleClient.lastValueText.ifBlank {
            bleClient.lastValueHex.ifBlank { "-" }
        }
    }
}
