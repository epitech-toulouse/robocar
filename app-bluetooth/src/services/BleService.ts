import { BleManager, Device, Characteristic } from 'react-native-ble-plx';
import { PermissionsAndroid, Platform } from 'react-native';
import * as Location from 'expo-location';
import LogService from './LogService';
import { Buffer } from 'buffer';

// HM-10 Default UUIDs
const SERVICE_UUID = '0000ffe0-0000-1000-8000-00805f9b34fb';
const CHARACTERISTIC_UUID = '0000ffe1-0000-1000-8000-00805f9b34fb';

class BleService {
  private manager: BleManager;
  private connectedDevice: Device | null = null;
  private characteristic: Characteristic | null = null;

  constructor() {
    this.manager = new BleManager();
  }

  async requestPermissions(): Promise<boolean> {
    if (Platform.OS === 'ios') return true;

    if (Platform.OS === 'android') {
      if (Platform.Version >= 31) {
        const granted = await PermissionsAndroid.requestMultiple([
          PermissionsAndroid.PERMISSIONS.BLUETOOTH_SCAN,
          PermissionsAndroid.PERMISSIONS.BLUETOOTH_CONNECT,
          PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
        ]);
        return (
          granted['android.permission.BLUETOOTH_SCAN'] === PermissionsAndroid.RESULTS.GRANTED &&
          granted['android.permission.BLUETOOTH_CONNECT'] === PermissionsAndroid.RESULTS.GRANTED &&
          granted['android.permission.ACCESS_FINE_LOCATION'] === PermissionsAndroid.RESULTS.GRANTED
        );
      } else {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION
        );
        return granted === PermissionsAndroid.RESULTS.GRANTED;
      }
    }
    return false;
  }

  scanDevices(onDeviceFound: (device: Device) => void) {
    LogService.addLog('Démarrage du scan Bluetooth...', 'info');
    this.manager.startDeviceScan(null, null, (error, device) => {
      if (error) {
        LogService.addLog(`Erreur de scan: ${error.message}`, 'error');
        return;
      }
      if (device && device.name) {
        onDeviceFound(device);
      }
    });
  }

  stopScan() {
    this.manager.stopDeviceScan();
    LogService.addLog('Scan arrêté.', 'info');
  }

  async connectToDevice(device: Device): Promise<boolean> {
    try {
      LogService.addLog(`Connexion à ${device.name}...`, 'info');
      const connectedDevice = await device.connect();
      this.connectedDevice = connectedDevice;
      
      LogService.addLog('Découverte des services...', 'info');
      await connectedDevice.discoverAllServicesAndCharacteristics();
      
      const services = await connectedDevice.services();
      for (const service of services) {
        const characteristics = await service.characteristics();
        const found = characteristics.find(c => c.uuid.toLowerCase() === CHARACTERISTIC_UUID.toLowerCase());
        if (found) {
          this.characteristic = found;
          break;
        }
      }

      if (this.characteristic) {
        LogService.addLog('Connecté et prêt !', 'info');
        return true;
      } else {
        LogService.addLog('Caractéristique de contrôle non trouvée.', 'error');
        return false;
      }
    } catch (error: any) {
      LogService.addLog(`Erreur de connexion: ${error.message}`, 'error');
      return false;
    }
  }

  async disconnect() {
    if (this.connectedDevice) {
      await this.connectedDevice.cancelConnection();
      this.connectedDevice = null;
      this.characteristic = null;
      LogService.addLog('Déconnecté.', 'warning');
    }
  }

  async sendCommand(command: string) {
    if (!this.characteristic) {
      console.warn('Non connecté à un appareil Bluetooth');
      return;
    }

    try {
      // Buffer is better for raw bytes if needed, but for HM-10 ASCII string works
      const base64Content = Buffer.from(command).toString('base64');
      await this.characteristic.writeWithoutResponse(base64Content);
    } catch (error: any) {
      LogService.addLog(`Erreur d'envoi BLE: ${error.message}`, 'error');
    }
  }

  getConnectedDevice() {
    return this.connectedDevice;
  }
}

export default new BleService();
