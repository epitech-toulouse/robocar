package com.example.myapplication.bluetooth

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider

class BleViewModel(
    val bleClient: BleClient
) : ViewModel() {
    fun sendText(payload: String): Boolean {
        return bleClient.writeValue(payload.encodeToByteArray())
    }
}

class BleViewModelFactory(
    private val bleClient: BleClient
) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(BleViewModel::class.java)) {
            @Suppress("UNCHECKED_CAST")
            return BleViewModel(bleClient) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}
