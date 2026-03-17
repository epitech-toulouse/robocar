import BleService from './BleService';

class CarControlService {
  async handleForward(isPressed: boolean) {
    console.log(`[CarControl] Forward: ${isPressed ? 'PRESSED' : 'RELEASED'}`);
    await BleService.sendCommand(isPressed ? 'F' : 'f');
  }

  async handleBackward(isPressed: boolean) {
    console.log(`[CarControl] Backward: ${isPressed ? 'PRESSED' : 'RELEASED'}`);
    await BleService.sendCommand(isPressed ? 'B' : 'b');
  }

  async handleLeft(isPressed: boolean) {
    console.log(`[CarControl] Left: ${isPressed ? 'PRESSED' : 'RELEASED'}`);
    await BleService.sendCommand(isPressed ? 'L' : 'l');
  }

  async handleRight(isPressed: boolean) {
    console.log(`[CarControl] Right: ${isPressed ? 'PRESSED' : 'RELEASED'}`);
    await BleService.sendCommand(isPressed ? 'R' : 'r');
  }

  async emergencyStop() {
    console.log('[CarControl] EMERGENCY STOP TRIGGERED');
    await BleService.sendCommand('S');
  }
}

export default new CarControlService();
