package com.example.myapplication.bluetooth

import android.content.Context

object BleClientProvider {
    @Volatile
    private var sharedClient: BleClient? = null

    fun get(context: Context): BleClient {
        val existing = sharedClient
        if (existing != null) {
            return existing
        }

        return synchronized(this) {
            val doubleCheck = sharedClient
            if (doubleCheck != null) {
                doubleCheck
            } else {
                BleClient(context.applicationContext).also { created ->
                    sharedClient = created
                }
            }
        }
    }
}
