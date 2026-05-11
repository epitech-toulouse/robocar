package com.example.myapplication.bluetooth

import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.le.BluetoothLeScanner
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanFilter
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.Context
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.ParcelUuid
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import java.util.UUID

private val SERVICE_UUID: UUID = UUID.fromString("0100aaaf-6d66-7b98-2f4d-60a8c0265631")
private val CHARACTERISTIC_UUID: UUID = UUID.fromString("0200aaaf-6d66-7b98-2f4d-60a8c0265631")
private val CLIENT_CONFIG_UUID: UUID = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")

data class BleDevice(
    val name: String,
    val address: String,
    val rssi: Int
)

enum class BleConnectionState {
    Idle,
    Scanning,
    Connecting,
    Connected,
    Disconnected,
    Error
}

class BleClient(private val context: Context) {
    companion object {
        private const val TAG = "BleClient"
        private const val CONNECT_DELAY_MS = 350L
        private const val CONNECT_TIMEOUT_MS = 10000L
        private const val RETRY_DELAY_MS = 800L
    }

    private val bluetoothManager = context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
    private val bluetoothAdapter: BluetoothAdapter? = bluetoothManager.adapter

    private var gatt: BluetoothGatt? = null
    private var targetCharacteristic: BluetoothGattCharacteristic? = null
    private val mainHandler = Handler(Looper.getMainLooper())
    private var pendingConnectAddress: String? = null
    private var activeGattAddress: String? = null
    private var connectionGeneration = 0
    private var connectRetryCount = 0

    val scanResults = mutableStateListOf<BleDevice>()
    var connectionState by mutableStateOf(BleConnectionState.Idle)
    var statusText by mutableStateOf("Idle")
    var lastValueHex by mutableStateOf("")
    var lastValueText by mutableStateOf("")
    var useServiceFilter by mutableStateOf(false)
    var systemConnectionSummary by mutableStateOf("System BLE: unknown")

    private val scanCallback = object : ScanCallback() {
        override fun onScanResult(callbackType: Int, result: ScanResult) {
            val device = result.device
            val name = device.name ?: "(no name)"
            val address = device.address
            val rssi = result.rssi
            Log.d(TAG, "onScanResult type=$callbackType name=$name address=$address rssi=$rssi")
            val existingIndex = scanResults.indexOfFirst { it.address == address }
            val entry = BleDevice(name = name, address = address, rssi = rssi)
            if (existingIndex >= 0) {
                scanResults[existingIndex] = entry
            } else {
                scanResults.add(entry)
            }
        }

        override fun onScanFailed(errorCode: Int) {
            connectionState = BleConnectionState.Error
            statusText = "Scan failed: ${scanErrorLabel(errorCode)}"
            Log.e(TAG, "onScanFailed code=$errorCode label=${scanErrorLabel(errorCode)}")
        }
    }

    private val gattCallback = object : BluetoothGattCallback() {
        override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
            Log.d(TAG, "onConnectionStateChange address=${gatt.device?.address} status=$status newState=$newState")
            if (status != BluetoothGatt.GATT_SUCCESS) {
                val failedAddress = gatt.device?.address ?: pendingConnectAddress
                mainHandler.removeCallbacksAndMessages(CONNECT_TIMEOUT_TOKEN)
                connectionState = BleConnectionState.Error
                statusText = "GATT error: $status"
                targetCharacteristic = null
                if (this@BleClient.gatt === gatt) {
                    this@BleClient.gatt = null
                }
                gatt.close()
                if (status == 133 && failedAddress != null && connectRetryCount < 1) {
                    connectRetryCount += 1
                    statusText = "GATT 133, retry ${connectRetryCount}/1"
                    Log.w(TAG, "Retrying connection after GATT 133 for $failedAddress")
                    scheduleConnect(failedAddress, retrying = true)
                } else {
                    pendingConnectAddress = null
                    activeGattAddress = null
                }
                return
            }

            when (newState) {
                BluetoothProfile.STATE_CONNECTED -> {
                    pendingConnectAddress = null
                    activeGattAddress = gatt.device?.address
                    mainHandler.removeCallbacksAndMessages(CONNECT_TIMEOUT_TOKEN)
                    connectionState = BleConnectionState.Connected
                    statusText = "Connected, discovering services"
                    Log.d(TAG, "discoverServices for $activeGattAddress")
                    gatt.discoverServices()
                }
                BluetoothProfile.STATE_DISCONNECTED -> {
                    mainHandler.removeCallbacksAndMessages(CONNECT_TIMEOUT_TOKEN)
                    connectionState = BleConnectionState.Disconnected
                    statusText = "Disconnected"
                    targetCharacteristic = null
                    if (this@BleClient.gatt === gatt) {
                        this@BleClient.gatt = null
                    }
                    activeGattAddress = null
                    gatt.close()
                }
            }
        }

        override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
            Log.d(TAG, "onServicesDiscovered address=${gatt.device?.address} status=$status")
            if (status != BluetoothGatt.GATT_SUCCESS) {
                connectionState = BleConnectionState.Error
                statusText = "Service discovery failed: $status"
                return
            }

            val discoveredServices = gatt.services.orEmpty()
            if (discoveredServices.isEmpty()) {
                connectionState = BleConnectionState.Error
                statusText = "Connected but no GATT services found"
                Log.e(TAG, "No services discovered on device ${gatt.device?.address}")
                return
            }

            for (service in discoveredServices) {
                Log.d(TAG, "Service ${service.uuid}")
                for (characteristic in service.characteristics.orEmpty()) {
                    Log.d(TAG, "  Characteristic ${characteristic.uuid}")
                }
            }

            val service = gatt.getService(SERVICE_UUID)
            val characteristic = service?.getCharacteristic(CHARACTERISTIC_UUID)
            if (service == null || characteristic == null) {
                connectionState = BleConnectionState.Error
                val availableServiceIds = discoveredServices.joinToString { it.uuid.toString() }
                statusText = if (service == null) {
                    "Service not found: $SERVICE_UUID"
                } else {
                    "Characteristic not found: $CHARACTERISTIC_UUID"
                }
                Log.e(
                    TAG,
                    "Expected service=$SERVICE_UUID characteristic=$CHARACTERISTIC_UUID not found. Available services=$availableServiceIds"
                )
                return
            }

            targetCharacteristic = characteristic
            statusText = "Service ready"
            connectRetryCount = 0
            setNotify(true)
        }

        override fun onCharacteristicChanged(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic
        ) {
            if (characteristic.uuid == CHARACTERISTIC_UUID) {
                lastValueHex = characteristic.value.toHexString()
                lastValueText = characteristic.value.toString(Charsets.UTF_8)
            }
        }

        override fun onCharacteristicRead(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic,
            status: Int
        ) {
            if (status == BluetoothGatt.GATT_SUCCESS && characteristic.uuid == CHARACTERISTIC_UUID) {
                lastValueHex = characteristic.value.toHexString()
                lastValueText = characteristic.value.toString(Charsets.UTF_8)
            }
        }

        override fun onCharacteristicWrite(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic,
            status: Int
        ) {
            if (status != BluetoothGatt.GATT_SUCCESS) {
                statusText = "Write failed: $status"
            }
        }
    }

    private object CONNECT_TIMEOUT_TOKEN

    private fun scanner(): BluetoothLeScanner? = bluetoothAdapter?.bluetoothLeScanner

    init {
        refreshSystemConnectionState()
    }

    private fun scanErrorLabel(errorCode: Int): String {
        return when (errorCode) {
            ScanCallback.SCAN_FAILED_ALREADY_STARTED -> "$errorCode already started"
            ScanCallback.SCAN_FAILED_APPLICATION_REGISTRATION_FAILED -> "$errorCode app registration failed"
            ScanCallback.SCAN_FAILED_FEATURE_UNSUPPORTED -> "$errorCode feature unsupported"
            ScanCallback.SCAN_FAILED_INTERNAL_ERROR -> "$errorCode internal error"
            ScanCallback.SCAN_FAILED_OUT_OF_HARDWARE_RESOURCES -> "$errorCode out of hardware resources"
            ScanCallback.SCAN_FAILED_SCANNING_TOO_FREQUENTLY -> "$errorCode scanning too frequently"
            else -> errorCode.toString()
        }
    }

    @SuppressLint("MissingPermission")
    fun startScan() {
        refreshSystemConnectionState()
        val adapter = bluetoothAdapter
        if (adapter == null) {
            connectionState = BleConnectionState.Error
            statusText = "Bluetooth adapter unavailable"
            Log.e(TAG, "startScan failed: adapter unavailable")
            return
        }
        if (!adapter.isEnabled) {
            connectionState = BleConnectionState.Error
            statusText = "Bluetooth disabled"
            Log.e(TAG, "startScan failed: bluetooth disabled")
            return
        }

        val scanner = scanner()
        if (scanner == null) {
            connectionState = BleConnectionState.Error
            statusText = "BLE scanner unavailable"
            Log.e(TAG, "startScan failed: scanner unavailable")
            return
        }

        scanResults.clear()
        connectionState = BleConnectionState.Scanning
        statusText = "Scanning"
        Log.d(TAG, "startScan requested")

        val settings = ScanSettings.Builder()
            .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
            .build()

        try {
            scanner.stopScan(scanCallback)
            if (useServiceFilter) {
                val filter = ScanFilter.Builder().setServiceUuid(ParcelUuid(SERVICE_UUID)).build()
                Log.d(TAG, "startScan with service filter $SERVICE_UUID")
                scanner.startScan(listOf(filter), settings, scanCallback)
            } else {
                Log.d(TAG, "startScan without service filter")
                scanner.startScan(null, settings, scanCallback)
            }
        } catch (securityException: SecurityException) {
            connectionState = BleConnectionState.Error
            statusText = "Missing BLE permission"
            Log.e(TAG, "startScan security exception", securityException)
        } catch (exception: Exception) {
            connectionState = BleConnectionState.Error
            statusText = "Scan exception: ${exception.javaClass.simpleName}"
            Log.e(TAG, "startScan exception", exception)
        }
    }

    @SuppressLint("MissingPermission")
    fun stopScan() {
        try {
            scanner()?.stopScan(scanCallback)
            Log.d(TAG, "stopScan requested")
        } catch (securityException: SecurityException) {
            Log.e(TAG, "stopScan security exception", securityException)
        }
        if (connectionState == BleConnectionState.Scanning) {
            connectionState = BleConnectionState.Idle
            statusText = "Idle"
        }
    }

    @SuppressLint("MissingPermission")
    fun connect(address: String) {
        refreshSystemConnectionState()
        val adapter = bluetoothAdapter
        if (adapter == null) {
            connectionState = BleConnectionState.Error
            statusText = "Bluetooth adapter unavailable"
            Log.e(TAG, "connect failed: adapter unavailable")
            return
        }
        if (!adapter.isEnabled) {
            connectionState = BleConnectionState.Error
            statusText = "Bluetooth disabled"
            Log.e(TAG, "connect failed: bluetooth disabled")
            return
        }

        val device = bluetoothAdapter?.getRemoteDevice(address)
        if (device == null) {
            connectionState = BleConnectionState.Error
            statusText = "Device not found"
            Log.e(TAG, "connect failed: device not found for $address")
            return
        }

        stopScan()
        closeCurrentGatt("connect($address)")
        pendingConnectAddress = address
        targetCharacteristic = null
        connectionGeneration += 1
        connectRetryCount = 0
        connectionState = BleConnectionState.Connecting
        statusText = "Connecting..."
        Log.d(TAG, "connect requested address=$address generation=$connectionGeneration")
        scheduleConnect(device.address, retrying = false)
    }

    @SuppressLint("MissingPermission")
    private fun scheduleConnect(address: String, retrying: Boolean) {
        val generation = connectionGeneration
        val delayMs = if (retrying) RETRY_DELAY_MS else CONNECT_DELAY_MS
        pendingConnectAddress = address
        mainHandler.postDelayed(
            {
                if (generation != connectionGeneration) {
                    Log.d(TAG, "skip stale connect generation=$generation current=$connectionGeneration")
                    return@postDelayed
                }
                performConnect(address, generation)
            },
            delayMs
        )
    }

    @SuppressLint("MissingPermission")
    private fun performConnect(address: String, generation: Int) {
        val adapter = bluetoothAdapter
        if (adapter == null || !adapter.isEnabled) {
            connectionState = BleConnectionState.Error
            statusText = "Bluetooth unavailable for connect"
            Log.e(TAG, "performConnect failed: bluetooth unavailable")
            return
        }
        val device = try {
            adapter.getRemoteDevice(address)
        } catch (exception: IllegalArgumentException) {
            connectionState = BleConnectionState.Error
            statusText = "Invalid device address"
            Log.e(TAG, "performConnect invalid address=$address", exception)
            return
        }

        closeCurrentGatt("performConnect($address)")
        connectionState = BleConnectionState.Connecting
        statusText = if (connectRetryCount > 0) "Retrying connection..." else "Connecting..."
        Log.d(TAG, "performConnect address=$address generation=$generation retry=$connectRetryCount")
        gatt = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            device.connectGatt(context, false, gattCallback, BluetoothDevice.TRANSPORT_LE)
        } else {
            device.connectGatt(context, false, gattCallback)
        }
        activeGattAddress = address
        mainHandler.removeCallbacksAndMessages(CONNECT_TIMEOUT_TOKEN)
        mainHandler.postAtTime({
            if (generation != connectionGeneration) {
                return@postAtTime
            }
            if (connectionState == BleConnectionState.Connecting) {
                Log.e(TAG, "connect timeout address=$address generation=$generation")
                statusText = "Connection timeout"
                connectionState = BleConnectionState.Error
                closeCurrentGatt("timeout")
            }
        }, CONNECT_TIMEOUT_TOKEN, System.currentTimeMillis() + CONNECT_TIMEOUT_MS)
    }

    @SuppressLint("MissingPermission")
    private fun closeCurrentGatt(reason: String) {
        val currentGatt = gatt ?: return
        Log.d(TAG, "closeCurrentGatt reason=$reason address=${currentGatt.device?.address}")
        try {
            currentGatt.disconnect()
        } catch (_: Exception) {
        }
        try {
            currentGatt.close()
        } catch (_: Exception) {
        }
        if (gatt === currentGatt) {
            gatt = null
        }
        targetCharacteristic = null
        activeGattAddress = null
    }

    @SuppressLint("MissingPermission")
    fun disconnect() {
        Log.d(TAG, "disconnect requested")
        connectionGeneration += 1
        pendingConnectAddress = null
        mainHandler.removeCallbacksAndMessages(CONNECT_TIMEOUT_TOKEN)
        closeCurrentGatt("disconnect()")
        connectionState = BleConnectionState.Disconnected
        statusText = "Disconnected"
        refreshSystemConnectionState()
    }

    @SuppressLint("MissingPermission")
    fun refreshSystemConnectionState() {
        val adapter = bluetoothAdapter
        if (adapter == null) {
            systemConnectionSummary = "System BLE: adapter unavailable"
            return
        }
        if (!adapter.isEnabled) {
            systemConnectionSummary = "System BLE: bluetooth disabled"
            return
        }

        try {
            val connectedGattDevices = bluetoothManager.getConnectedDevices(BluetoothProfile.GATT)
            if (connectedGattDevices.isNotEmpty()) {
                val summary = connectedGattDevices.joinToString { device ->
                    val name = device.name ?: "(no name)"
                    "$name ${device.address}"
                }
                systemConnectionSummary = "System BLE: connected to $summary"
                Log.d(TAG, "refreshSystemConnectionState connectedGatt=$summary")
            } else {
                val bondedSummary = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    adapter.bondedDevices
                } else {
                    @Suppress("DEPRECATION")
                    adapter.bondedDevices
                }.orEmpty().joinToString { device ->
                    val name = device.name ?: "(no name)"
                    "$name ${device.address}"
                }
                systemConnectionSummary = if (bondedSummary.isBlank()) {
                    "System BLE: no connected GATT device"
                } else {
                    "System BLE: no connected GATT device, paired: $bondedSummary"
                }
                Log.d(TAG, "refreshSystemConnectionState no connected GATT device")
            }
        } catch (securityException: SecurityException) {
            systemConnectionSummary = "System BLE: missing permission"
            Log.e(TAG, "refreshSystemConnectionState security exception", securityException)
        } catch (exception: Exception) {
            systemConnectionSummary = "System BLE: ${exception.javaClass.simpleName}"
            Log.e(TAG, "refreshSystemConnectionState exception", exception)
        }
    }

    @SuppressLint("MissingPermission")
    fun readValue() {
        val characteristic = targetCharacteristic ?: return
        gatt?.readCharacteristic(characteristic)
    }

    @SuppressLint("MissingPermission")
    fun writeValue(bytes: ByteArray): Boolean {
        return writePayload(bytes)
    }

    @SuppressLint("MissingPermission")
    fun sendAlgorithmResult(resultCode: Int): Boolean {
        val payload = "CV:$resultCode".encodeToByteArray()
        return writePayload(payload)
    }

    fun sendSteeringFrame(steeringPercent: Float, weight: Float = 1.0f): Boolean {
        val clampedSteering = steeringPercent.coerceIn(-100.0f, 100.0f)
        val payload = buildString {
            append("STEER:")
            append(String.format(java.util.Locale.US, "%.1f", clampedSteering))
            append('\n')
        }
        Log.d(TAG, "sendSteeringFrame payload=$payload")
        return writePayload(payload.encodeToByteArray())
    }

    fun sendStopCommand(): Boolean {
        Log.d(TAG, "sendStopCommand payload=STOP")
        return writePayload("STOP\n".encodeToByteArray())
    }

    fun sendGoCommand(): Boolean {
        Log.d(TAG, "sendGoCommand payload=GO")
        return writePayload("GO\n".encodeToByteArray())
    }

    fun sendProtocolChar(command: Char): Boolean {
        Log.d(TAG, "sendProtocolChar payload=$command")
        return writePayload(byteArrayOf(command.code.toByte()))
    }

    fun sendAlgorithms(selected: Collection<String>): Boolean {
        val payload = "ALG:${selected.joinToString(",")}\n"
        Log.d(TAG, "sendAlgorithms payload=$payload")
        return writePayload(payload.encodeToByteArray())
    }

    fun sendGpsGoal(latitude: String, longitude: String): Boolean {
        val payload = "GPS:${latitude.trim()},${longitude.trim()}\n"
        Log.d(TAG, "sendGpsGoal payload=$payload")
        return writePayload(payload.encodeToByteArray())
    }

    fun requestStatus(): Boolean {
        Log.d(TAG, "requestStatus payload=STATUS?")
        return writePayload("STATUS?\n".encodeToByteArray())
    }

    fun requestLogs(since: Int = 0): Boolean {
        val payload = "LOGS:$since\n"
        Log.d(TAG, "requestLogs payload=$payload")
        return writePayload(payload.encodeToByteArray())
    }

    @SuppressLint("MissingPermission")
    private fun writePayload(bytes: ByteArray): Boolean {
        val characteristic = targetCharacteristic
        if (connectionState != BleConnectionState.Connected || characteristic == null || gatt == null) {
            statusText = "Write skipped: not connected"
            return false
        }
        characteristic.value = bytes
        val accepted = gatt?.writeCharacteristic(characteristic) == true
        if (!accepted) {
            statusText = "Write failed to start"
        }
        return accepted
    }

    @SuppressLint("MissingPermission")
    fun setNotify(enabled: Boolean) {
        val characteristic = targetCharacteristic ?: return
        gatt?.setCharacteristicNotification(characteristic, enabled)

        val descriptor = characteristic.getDescriptor(CLIENT_CONFIG_UUID)
        if (descriptor != null) {
            descriptor.value = if (enabled) {
                BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
            } else {
                BluetoothGattDescriptor.DISABLE_NOTIFICATION_VALUE
            }
            gatt?.writeDescriptor(descriptor)
        }
    }

    fun close() {
        stopScan()
        disconnect()
    }
}

private fun ByteArray.toHexString(): String {
    return joinToString(separator = " ") { byte -> String.format("%02X", byte) }
}
