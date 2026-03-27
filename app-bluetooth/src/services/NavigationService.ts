/**
 * NavigationService - Sends GPS waypoint commands to the robot via BLE.
 *
 * JSON protocol over the same BLE characteristic (0xFFE1):
 *   {"cmd":"NAV","lat":43.612290,"lon":1.428899}  → Navigate to point
 *   {"cmd":"STOP"}                                 → Stop navigation
 */

import BleService from './BleService';

class NavigationService {
  /**
   * Send a GPS target to the robot.
   * @param lat Latitude in decimal degrees (-90..90)
   * @param lon Longitude in decimal degrees (-180..180)
   */
  async sendTarget(lat: number, lon: number): Promise<void> {
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      throw new Error(`Invalid coordinates: lat=${lat}, lon=${lon}`);
    }

    const cmd = JSON.stringify({
      cmd: 'NAV',
      lat: parseFloat(lat.toFixed(6)),
      lon: parseFloat(lon.toFixed(6)),
    });

    await BleService.sendCommand(cmd);
    console.log(`[NavigationService] Target sent: ${cmd}`);
  }

  /**
   * Send a stop navigation command.
   */
  async sendStop(): Promise<void> {
    const cmd = JSON.stringify({cmd: 'STOP'});
    await BleService.sendCommand(cmd);
    console.log('[NavigationService] Stop sent');
  }
}

export default new NavigationService();
