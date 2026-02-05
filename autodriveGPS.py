#!/usr/bin/env python3
"""
AutoDrive avec support GPS optionnel
Usage:
  python3 autodriveGPS.py                              # Mode LIDAR seul
  python3 autodriveGPS.py --gps                        # Avec GPS (affichage position)
  python3 autodriveGPS.py --gps --gps-host 192.168.1.100 --gps-port 25000
"""

import argparse
import sys
from control.Manager import Manager
from autodrive.autodriveState import AutoDriveState


def main():
    parser = argparse.ArgumentParser(description='AutoDrive avec navigation GPS optionnelle')
    parser.add_argument('--gps', action='store_true', 
                        help='Activer la lecture GPS')
    parser.add_argument('--gps-host', type=str, default='localhost',
                        help='Adresse du serveur GPS (défaut: localhost)')
    parser.add_argument('--gps-port', type=int, default=25000,
                        help='Port du serveur GPS (défaut: 25000)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"🚗 Démarrage AutoDrive")
    print(f"   GPS: {'✅ activé' if args.gps else '❌ désactivé'}")
    if args.gps:
        print(f"   Serveur: {args.gps_host}:{args.gps_port}")
    print("=" * 60)
    
    # Créer l'état AutoDrive avec ou sans GPS
    autodrive_state = AutoDriveState(
        use_gps=args.gps,
        gps_host=args.gps_host,
        gps_port=args.gps_port
    )
    
    # Créer le Manager avec l'état AutoDrive
    manager = Manager(autodrive_state)
    
    try:
        manager.loop()
    except KeyboardInterrupt:
        print("\n🛑 Arrêt demandé...")
        manager.safe_stop()


if __name__ == "__main__":
    main()
