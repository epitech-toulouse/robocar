package com.example.myapplication.camera

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.myapplication.bluetooth.BleClient
import com.example.myapplication.nativebridge.NativeCvFrameProcessor

class CameraViewModel(
    bleClient: BleClient
) : ViewModel() {
    val frameProcessor: FrameProcessor = NativeCvFrameProcessor { resultCode ->
        bleClient.sendAlgorithmResult(resultCode)
    }
}

class CameraViewModelFactory(
    private val bleClient: BleClient
) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(CameraViewModel::class.java)) {
            @Suppress("UNCHECKED_CAST")
            return CameraViewModel(bleClient) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}
