/**
 * MapScreen — Tap-to-navigate map interface for Robocar.
 *
 * Features:
 *   - MapView with long-press to set target marker.
 *   - "Naviguer" button to send GPS target via BLE.
 *   - "Arrêter" button to cancel navigation.
 *   - Status display (connected/disconnected, nav state).
 */

import React, {useState, useCallback} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Platform,
} from 'react-native';
import MapView, {Marker, MapPressEvent, Region} from 'react-native-maps';
import NavigationService from '../services/NavigationService';
import LogService from '../services/LogService';

interface TargetPoint {
  latitude: number;
  longitude: number;
}

const MapScreen: React.FC = () => {
  const [target, setTarget] = useState<TargetPoint | null>(null);
  const [isNavigating, setIsNavigating] = useState(false);

  // Default region: Toulouse, France
  const [region] = useState<Region>({
    latitude: 43.6047,
    longitude: 1.4442,
    latitudeDelta: 0.01,
    longitudeDelta: 0.01,
  });

  const handleLongPress = useCallback((event: MapPressEvent) => {
    const {latitude, longitude} = event.nativeEvent.coordinate;
    setTarget({latitude, longitude});
    LogService.addLog(
      `Cible définie: ${latitude.toFixed(6)}, ${longitude.toFixed(6)}`,
      'info',
    );
  }, []);

  const handleNavigate = useCallback(async () => {
    if (!target) {
      Alert.alert('Aucune cible', 'Appuyez longuement sur la carte pour définir un point cible.');
      return;
    }

    try {
      await NavigationService.sendTarget(target.latitude, target.longitude);
      setIsNavigating(true);
      LogService.addLog(
        `Navigation lancée vers ${target.latitude.toFixed(6)}, ${target.longitude.toFixed(6)}`,
        'info',
      );
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Erreur inconnue';
      Alert.alert('Erreur', `Impossible d'envoyer la cible: ${msg}`);
      LogService.addLog(`Erreur navigation: ${msg}`, 'error');
    }
  }, [target]);

  const handleStop = useCallback(async () => {
    try {
      await NavigationService.sendStop();
      setIsNavigating(false);
      LogService.addLog('Navigation arrêtée', 'warning');
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Erreur inconnue';
      LogService.addLog(`Erreur arrêt: ${msg}`, 'error');
    }
  }, []);

  const clearTarget = useCallback(() => {
    setTarget(null);
    setIsNavigating(false);
  }, []);

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>🗺️ Navigation GPS</Text>
        <View
          style={[
            styles.statusBadge,
            isNavigating ? styles.statusActive : styles.statusIdle,
          ]}>
          <Text style={styles.statusText}>
            {isNavigating ? '🟢 En navigation' : '⚪ En attente'}
          </Text>
        </View>
      </View>

      {/* Map */}
      <MapView
        style={styles.map}
        initialRegion={region}
        onLongPress={handleLongPress}
        showsUserLocation={true}
        showsMyLocationButton={true}
        mapType="standard">
        {target && (
          <Marker
            coordinate={target}
            title="Cible"
            description={`${target.latitude.toFixed(6)}, ${target.longitude.toFixed(6)}`}
            pinColor="red"
          />
        )}
      </MapView>

      {/* Target info */}
      {target && (
        <View style={styles.targetInfo}>
          <Text style={styles.targetText}>
            📍 {target.latitude.toFixed(6)}, {target.longitude.toFixed(6)}
          </Text>
          <TouchableOpacity onPress={clearTarget} style={styles.clearButton}>
            <Text style={styles.clearButtonText}>✕</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Action buttons */}
      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={[
            styles.button,
            styles.navigateButton,
            (!target || isNavigating) && styles.buttonDisabled,
          ]}
          onPress={handleNavigate}
          disabled={!target || isNavigating}>
          <Text style={styles.buttonText}>🚀 Naviguer</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.button,
            styles.stopButton,
            !isNavigating && styles.buttonDisabled,
          ]}
          onPress={handleStop}
          disabled={!isNavigating}>
          <Text style={styles.buttonText}>🛑 Arrêter</Text>
        </TouchableOpacity>
      </View>

      {/* Instructions */}
      {!target && (
        <View style={styles.instructions}>
          <Text style={styles.instructionText}>
            Appuyez longuement sur la carte pour définir un point cible
          </Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingBottom: 12,
    backgroundColor: '#16213e',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#e8e8e8',
  },
  statusBadge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusActive: {
    backgroundColor: '#1b4332',
  },
  statusIdle: {
    backgroundColor: '#2d2d44',
  },
  statusText: {
    fontSize: 12,
    color: '#e8e8e8',
  },
  map: {
    flex: 1,
  },
  targetInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#16213e',
    paddingHorizontal: 16,
    paddingVertical: 10,
  },
  targetText: {
    color: '#a8dadc',
    fontSize: 14,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  clearButton: {
    padding: 6,
  },
  clearButtonText: {
    color: '#e76f51',
    fontSize: 18,
    fontWeight: 'bold',
  },
  buttonContainer: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
    backgroundColor: '#1a1a2e',
  },
  button: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  navigateButton: {
    backgroundColor: '#2d6a4f',
  },
  stopButton: {
    backgroundColor: '#9b2226',
  },
  buttonDisabled: {
    opacity: 0.4,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  instructions: {
    padding: 16,
    alignItems: 'center',
  },
  instructionText: {
    color: '#6c757d',
    fontSize: 13,
    textAlign: 'center',
  },
});

export default MapScreen;
