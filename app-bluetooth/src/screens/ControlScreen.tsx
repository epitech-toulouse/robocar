import React, { useState } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { GestureDetector, Gesture } from 'react-native-gesture-handler';
import CarControlService from '../services/CarControlService';
import LogService from '../services/LogService';

const ControlScreen = () => {
  const handleAction = (action: string, isPressed: boolean, command: (pressed: boolean) => void) => {
    command(isPressed);
    LogService.addLog(`Action : ${action} (${isPressed ? 'Enfoncé' : 'Relâché'})`, 'info');
  };

  const handleEmergencyStop = () => {
    CarControlService.emergencyStop();
    LogService.addLog('Arrêt d\'urgence déclenché depuis Contrôles !', 'error');
  };

  const ControlButton = ({ title, onToggle, style }: any) => {
    const [pressed, setPressed] = useState(false);

    const gesture = Gesture.Pan()
      .onBegin(() => {
        setPressed(true);
        onToggle(true);
      })
      .onFinalize(() => {
        setPressed(false);
        onToggle(false);
      })
      .runOnJS(true); // Required since we update React state

    return (
      <GestureDetector gesture={gesture}>
        <View 
          style={[
            styles.button, 
            style,
            pressed && styles.buttonPressed,
          ]} 
        >
          <Text style={styles.buttonText}>{title}</Text>
        </View>
      </GestureDetector>
    );
  };

  const EmergencyButton = () => {
    const [pressed, setPressed] = useState(false);

    const gesture = Gesture.Pan()
      .onBegin(() => {
        setPressed(true);
        handleEmergencyStop();
      })
      .onFinalize(() => {
        setPressed(false);
      })
      .runOnJS(true);

    return (
      <GestureDetector gesture={gesture}>
        <View 
          style={[
            styles.emergencyButton,
            pressed && styles.emergencyButtonPressed,
          ]} 
        >
          <Text style={styles.emergencyText}>STOP</Text>
        </View>
      </GestureDetector>
    );
  };

  return (
    <View style={styles.container}>
      {/* Left Pane - Forward / Backward */}
      <View style={styles.leftPane}>
        <View style={styles.verticalControls}>
          <ControlButton 
            title="AVANCER" 
            onToggle={(isPressed: boolean) => handleAction('Avancer', isPressed, CarControlService.handleForward)} 
            style={styles.upButton} 
          />
          <ControlButton 
            title="RECULER" 
            onToggle={(isPressed: boolean) => handleAction('Reculer', isPressed, CarControlService.handleBackward)} 
            style={styles.downButton} 
          />
        </View>
      </View>

      {/* Center - Emergency Stop */}
      <View style={styles.centerPane}>
        <EmergencyButton />
      </View>

      {/* Right Pane - Left / Right */}
      <View style={styles.rightPane}>
        <View style={styles.horizontalControls}>
          <ControlButton 
            title="GAUCHE" 
            onToggle={(isPressed: boolean) => handleAction('Tourner à Gauche', isPressed, CarControlService.handleLeft)} 
            style={styles.leftButton} 
          />
          <ControlButton 
            title="DROITE" 
            onToggle={(isPressed: boolean) => handleAction('Tourner à Droite', isPressed, CarControlService.handleRight)} 
            style={styles.rightButton} 
          />
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    flexDirection: 'row',
  },
  leftPane: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  centerPane: {
    width: 140,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderLeftWidth: 1,
    borderRightWidth: 1,
    borderColor: '#e5e5ea',
  },
  emergencyButton: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: '#ff3b30',
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    borderWidth: 4,
    borderColor: '#e02a20',
  },
  emergencyButtonPressed: {
    transform: [{ scale: 0.9 }],
    backgroundColor: '#cc2f26',
  },
  emergencyText: {
    color: '#fff',
    fontSize: 22,
    fontWeight: 'bold',
  },
  rightPane: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  verticalControls: {
    alignItems: 'center',
    justifyContent: 'center',
    gap: 30, // For react-native >= 0.71
  },
  horizontalControls: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 30,
  },
  button: {
    width: 120,
    height: 120,
    backgroundColor: '#007AFF',
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 60,
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  buttonPressed: {
    transform: [{ scale: 0.95 }],
    opacity: 0.8,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  upButton: {},
  downButton: {},
  leftButton: {},
  rightButton: {},
});

export default ControlScreen;
