import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, FlatList, TouchableOpacity, ActivityIndicator, Alert, SafeAreaView } from 'react-native';
import { Device } from 'react-native-ble-plx';
import BleService from '../services/BleService';
import LogService from '../services/LogService';

const ConnectionScreen = ({ navigation }: any) => {
  const [devices, setDevices] = useState<Device[]>([]);
  const [isScanning, setIsScanning] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);

  useEffect(() => {
    startInitialScan();
    return () => {
      BleService.stopScan();
    };
  }, []);

  const startInitialScan = async () => {
    const hasPermission = await BleService.requestPermissions();
    if (hasPermission) {
      startScan();
    } else {
      Alert.alert('Permission refusée', 'L\'application a besoin des permissions Bluetooth et Localisation pour fonctionner.');
      LogService.addLog('Permissions Bluetooth/Localisation refusées.', 'error');
    }
  };

  const startScan = () => {
    if (isScanning) return;
    
    setDevices([]);
    setIsScanning(true);
    
    BleService.scanDevices((device) => {
      setDevices((prevDevices) => {
        if (prevDevices.find((d) => d.id === device.id)) {
          return prevDevices;
        }
        return [...prevDevices, device];
      });
    });

    // Stop scan after 10 seconds
    setTimeout(() => {
      stopScan();
    }, 10000);
  };

  const stopScan = () => {
    BleService.stopScan();
    setIsScanning(false);
  };

  const handleConnect = async (device: Device) => {
    stopScan();
    setIsConnecting(true);
    
    const success = await BleService.connectToDevice(device);
    setIsConnecting(false);

    if (success) {
      // Navigate to the main tabs
      navigation.replace('Main');
    } else {
      Alert.alert('Erreur de connexion', `Impossible de se connecter à ${device.name || 'cet appareil'}.`);
    }
  };

  const renderDevice = ({ item }: { item: Device }) => (
    <TouchableOpacity 
      style={styles.deviceItem}
      onPress={() => handleConnect(item)}
      disabled={isConnecting}
    >
      <View style={styles.deviceInfo}>
        <Text style={styles.deviceName}>{item.name || 'Appareil inconnu'}</Text>
        <Text style={styles.deviceId}>{item.id}</Text>
      </View>
      <View style={styles.connectButton}>
        <Text style={styles.connectButtonText}>Connecter</Text>
      </View>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Robocar - Connexion</Text>
        <Text style={styles.subtitle}>Sélectionnez votre véhicule</Text>
      </View>

      <View style={styles.scanSection}>
        {isScanning ? (
          <ActivityIndicator size="small" color="#007AFF" />
        ) : (
          <TouchableOpacity style={styles.refreshButton} onPress={startScan} disabled={isConnecting}>
            <Text style={styles.refreshButtonText}>Actualiser la liste</Text>
          </TouchableOpacity>
        )}
      </View>

      <FlatList
        data={devices}
        keyExtractor={(item) => item.id}
        renderItem={renderDevice}
        ListEmptyComponent={
          <Text style={styles.emptyText}>
            {isScanning ? 'Recherche en cours...' : 'Aucun appareil trouvé. Assurez-vous que le Bluetooth de la voiture est activé.'}
          </Text>
        }
        contentContainerStyle={styles.listContainer}
      />

      {isConnecting && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color="#007AFF" />
          <Text style={styles.loadingText}>Connexion en cours...</Text>
        </View>
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  header: {
    padding: 30,
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1c1c1e',
  },
  subtitle: {
    fontSize: 16,
    color: '#8e8e93',
    marginTop: 5,
  },
  scanSection: {
    padding: 15,
    alignItems: 'center',
    height: 60,
    justifyContent: 'center',
  },
  refreshButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
  },
  refreshButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  listContainer: {
    padding: 15,
  },
  deviceItem: {
    flexDirection: 'row',
    backgroundColor: '#f8f8f8',
    padding: 20,
    borderRadius: 12,
    marginBottom: 10,
    alignItems: 'center',
    justifyContent: 'space-between',
    borderWidth: 1,
    borderColor: '#e5e5ea',
  },
  deviceInfo: {
    flex: 1,
  },
  deviceName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1c1c1e',
  },
  deviceId: {
    fontSize: 12,
    color: '#8e8e93',
    marginTop: 4,
  },
  connectButton: {
    backgroundColor: '#34c759',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 8,
  },
  connectButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 14,
  },
  emptyText: {
    textAlign: 'center',
    marginTop: 50,
    color: '#8e8e93',
    fontSize: 16,
    paddingHorizontal: 40,
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 10,
  },
  loadingText: {
    marginTop: 15,
    fontSize: 18,
    fontWeight: '600',
    color: '#1c1c1e',
  },
});

export default ConnectionScreen;
