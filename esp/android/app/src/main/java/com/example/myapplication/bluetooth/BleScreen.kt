package com.example.myapplication.bluetooth

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothManager
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BleScreen(
    viewModel: BleViewModel,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val bleClient = viewModel.bleClient

    val requiredPermissions = remember {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            arrayOf(
                Manifest.permission.BLUETOOTH_SCAN,
                Manifest.permission.BLUETOOTH_CONNECT
            )
        } else {
            arrayOf(Manifest.permission.ACCESS_FINE_LOCATION)
        }
    }

    var permissionsGranted by remember {
        mutableStateOf(
            requiredPermissions.all { permission ->
                ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
            }
        )
    }
    val bluetoothManager = remember {
        context.getSystemService(BluetoothManager::class.java)
    }
    val canReadAdapter = Build.VERSION.SDK_INT < Build.VERSION_CODES.S ||
        ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.BLUETOOTH_CONNECT
        ) == PackageManager.PERMISSION_GRANTED
    var bluetoothEnabled by remember {
        mutableStateOf(canReadAdapter && bluetoothManager.adapter?.isEnabled == true)
    }
    var showAdvancedTools by remember { mutableStateOf(false) }
    var textToSend by remember { mutableStateOf("Hello ESP32") }

    val enableBtLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) {
        bluetoothEnabled = bluetoothManager.adapter?.isEnabled == true
        bleClient.refreshSystemConnectionState()
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) {
        permissionsGranted = requiredPermissions.all { permission ->
            ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
        }
        if (permissionsGranted) {
            bluetoothEnabled = bluetoothManager.adapter?.isEnabled == true
            bleClient.refreshSystemConnectionState()
        }
    }

    LaunchedEffect(Unit) {
        permissionsGranted = requiredPermissions.all { permission ->
            ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
        }
        if (permissionsGranted) {
            bluetoothEnabled = bluetoothManager.adapter?.isEnabled == true
            bleClient.refreshSystemConnectionState()
        }
    }

    LazyColumn(
        modifier = modifier.padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item {
            DashboardCard(
                title = "Etat Bluetooth",
                body = {
                    StatusLine("Statut", bleClient.statusText)
                    StatusLine("Session app", bleClient.connectionState.name)
                    StatusLine("Permissions", if (permissionsGranted) "Accordees" else "A demander")
                    StatusLine("Bluetooth", if (bluetoothEnabled) "Active" else "Inactif")
                    StatusLine("Systeme", bleClient.systemConnectionSummary.removePrefix("System BLE: "))
                }
            )
        }

        item {
            DashboardCard(
                title = "Actions rapides",
                body = {
                    TwoButtonsRow(
                        leftLabel = "Permissions",
                        leftAction = {
                            permissionsGranted = requiredPermissions.all { permission ->
                                ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
                            }
                            if (!permissionsGranted) {
                                permissionLauncher.launch(requiredPermissions)
                            } else {
                                bluetoothEnabled = bluetoothManager.adapter?.isEnabled == true
                                bleClient.refreshSystemConnectionState()
                                bleClient.statusText = "Permissions deja accordees"
                            }
                        },
                        rightLabel = "Activer BT",
                        rightAction = {
                            val intent = Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE)
                            enableBtLauncher.launch(intent)
                        },
                        rightEnabled = permissionsGranted
                    )

                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 8.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Button(
                            onClick = { bleClient.startScan() },
                            enabled = permissionsGranted && bluetoothEnabled,
                            modifier = Modifier.weight(1f)
                        ) {
                            Text("Lancer scan")
                        }
                        FilledTonalButton(
                            onClick = {
                                bluetoothEnabled = bluetoothManager.adapter?.isEnabled == true
                                bleClient.refreshSystemConnectionState()
                            },
                            enabled = permissionsGranted,
                            modifier = Modifier.weight(1f)
                        ) {
                            Text("Rafraichir")
                        }
                    }

                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 8.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        FilledTonalButton(
                            onClick = { bleClient.stopScan() },
                            modifier = Modifier.weight(1f)
                        ) {
                            Text("Stop scan")
                        }
                        FilledTonalButton(
                            onClick = { bleClient.disconnect() },
                            modifier = Modifier.weight(1f)
                        ) {
                            Text("Deconnecter")
                        }
                    }
                }
            )
        }

        item {
            DashboardCard(
                title = "Appareils detectes",
                body = {
                    if (bleClient.scanResults.isEmpty()) {
                        Text(
                            text = "Aucun appareil BLE detecte pour l'instant. Lance un scan puis approche le module.",
                            modifier = Modifier.padding(top = 4.dp)
                        )
                    }
                }
            )
        }

        items(bleClient.scanResults, key = { it.address }) { device ->
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors()
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = device.name,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(text = device.address)
                    Text(text = "RSSI ${device.rssi}")
                    Button(
                        onClick = {
                            bleClient.stopScan()
                            bleClient.connect(device.address)
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 12.dp)
                    ) {
                        Text("Se connecter")
                    }
                }
            }
        }

        item {
            DashboardCard(
                title = "Test BLE",
                body = {
                    Text("Envoie un message simple pour verifier si l'ecriture BLE fonctionne.")

                    OutlinedTextField(
                        value = textToSend,
                        onValueChange = { textToSend = it },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 8.dp),
                        label = { Text("Texte de test") }
                    )

                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 8.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Button(
                            onClick = { viewModel.sendText(textToSend) },
                            modifier = Modifier.weight(1f)
                        ) {
                            Text("Envoyer")
                        }
                        FilledTonalButton(
                            onClick = { bleClient.readValue() },
                            modifier = Modifier.weight(1f)
                        ) {
                            Text("Lire retour")
                        }
                    }

                    StatusLine("Derniere valeur hex", bleClient.lastValueHex.ifBlank { "-" })
                }
            )
        }

        item {
            FilledTonalButton(
                onClick = { showAdvancedTools = !showAdvancedTools },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(if (showAdvancedTools) "Masquer outils avances" else "Afficher outils avances")
            }
        }

        if (showAdvancedTools) {
            item {
                DashboardCard(
                    title = "Outils avances",
                    body = {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Checkbox(
                                checked = bleClient.useServiceFilter,
                                onCheckedChange = { bleClient.useServiceFilter = it }
                            )
                            Text(
                                text = "Filtrer le scan par UUID de service",
                                modifier = Modifier.padding(top = 12.dp)
                            )
                        }

                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 8.dp),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            FilledTonalButton(
                                onClick = { bleClient.setNotify(true) },
                                modifier = Modifier.weight(1f)
                            ) {
                                Text("Notify on")
                            }
                            FilledTonalButton(
                                onClick = { bleClient.setNotify(false) },
                                modifier = Modifier.weight(1f)
                            ) {
                                Text("Notify off")
                            }
                        }

                        HorizontalDivider(modifier = Modifier.padding(vertical = 12.dp))
                        Text("Les options de debug avance servent surtout aux tests de scan et de notifications.")
                    }
                )
            }
        }
    }
}

@Composable
private fun DashboardCard(
    title: String,
    body: @Composable () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors()
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = title,
                fontWeight = FontWeight.Bold
            )
            Column(
                modifier = Modifier.padding(top = 10.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                body()
            }
        }
    }
}

@Composable
private fun StatusLine(
    label: String,
    value: String
) {
    Column {
        Text(
            text = label,
            fontWeight = FontWeight.SemiBold
        )
        Text(text = value)
    }
}

@Composable
private fun TwoButtonsRow(
    leftLabel: String,
    leftAction: () -> Unit,
    rightLabel: String,
    rightAction: () -> Unit,
    rightEnabled: Boolean = true
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        FilledTonalButton(
            onClick = leftAction,
            modifier = Modifier.weight(1f)
        ) {
            Text(leftLabel)
        }
        Button(
            onClick = rightAction,
            enabled = rightEnabled,
            modifier = Modifier.weight(1f)
        ) {
            Text(rightLabel)
        }
    }
}
