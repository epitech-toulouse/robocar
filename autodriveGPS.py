#!/usr/bin/env python3
"""
AutoDrive avec navigation GPS
Usage:
  python3 autodriveGPS.py                              # Mode LIDAR seul
  python3 autodriveGPS.py --gps                        # Mode GPS sans objectif (affichage position)
  python3 autodriveGPS.py --gps --goal-lat 43.6122222 --goal-lon 1.42975  # Avec objectif
"""

import argparse
import sys
from control.Manager import Manager
from autodrive.autodriveState import AutoDriveState


def main():
    parser = argparse.ArgumentParser(description='AutoDrive avec navigation GPS optionnelle')
    parser.add_argument('--gps', action='store_true', 
                        help='Activer la navigation GPS')
    parser.add_argument('--gps-host', type=str, default='localhost',
                        help='Adresse du serveur GPS (défaut: localhost)')
    parser.add_argument('--gps-port', type=int, default=25000,
                        help='Port du serveur GPS (défaut: 25000)')
    parser.add_argument('--goal-lat', type=float, default=None,
                        help='Latitude de l\'objectif en degrés')
    parser.add_argument('--goal-lon', type=float, default=None,
                        help='Longitude de l\'objectif en degrés')
    
    args = parser.parse_args()
    
    # Vérifier que si un objectif est défini, les deux coordonnées sont présentes
    if (args.goal_lat is not None and args.goal_lon is None) or \
       (args.goal_lat is None and args.goal_lon is not None):
        print("❌ Erreur: --goal-lat et --goal-lon doivent être tous les deux définis")
        sys.exit(1)
    
    if args.gps:
        if args.goal_lat and args.goal_lon:
            print(f"Mode: GPS avec objectif")
            print(f"Objectif: {args.goal_lat:.6f}°, {args.goal_lon:.6f}°")
        else:
            print(f"Mode: GPS sans objectif (affichage position uniquement)")
    else:
        print(f"Mode: LIDAR seul")
    

    
    # Créer l'état AutoDrive avec ou sans GPS
    autodrive_state = AutoDriveState(
        use_gps=args.gps,
        gps_host=args.gps_host,
        gps_port=args.gps_port,
        goal_lat=args.goal_lat,
        goal_lon=args.goal_lon
    )
    
    # Créer le Manager avec l'état AutoDrive
    manager = Manager(autodrive_state)
    
    try:
        manager.loop()
    except KeyboardInterrupt:
        manager.safe_stop()


if __name__ == "__main__":
    main()
