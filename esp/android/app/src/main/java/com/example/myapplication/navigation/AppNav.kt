package com.example.myapplication.navigation

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.myapplication.bluetooth.BleClient
import com.example.myapplication.bluetooth.BleClientProvider
import com.example.myapplication.bluetooth.BleScreen
import com.example.myapplication.bluetooth.BleViewModel
import com.example.myapplication.bluetooth.BleViewModelFactory

@Composable
fun AppRoot(
    modifier: Modifier = Modifier,
    onOpenOpenCvCamera: () -> Unit
) {
    val context = LocalContext.current
    val bleClient: BleClient = BleClientProvider.get(context)

    val bleViewModel: BleViewModel = viewModel(
        factory = BleViewModelFactory(bleClient)
    )

    Scaffold(
        modifier = modifier
    ) { innerPadding ->
        Column(
            modifier = Modifier.padding(innerPadding)
        ) {
            HomeScreen(
                onOpenOpenCvCamera = onOpenOpenCvCamera
            )
            BleScreen(
                viewModel = bleViewModel,
                modifier = Modifier.fillMaxSize()
            )
        }
    }
}

@Composable
private fun HomeScreen(
    onOpenOpenCvCamera: () -> Unit
) {
    Column(
        modifier = Modifier
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("Parametres Bluetooth")
        Text("Configure d'abord le Bluetooth puis ouvre la camera OpenCV quand tu es pret.")

        Button(
            onClick = onOpenOpenCvCamera,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Ouvrir la camera OpenCV")
        }
    }
}
