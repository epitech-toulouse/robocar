#!/bin/bash
# Lance le serveur GPS qui parse les données et les expose en JSON
# Usage: ./gps_server.sh [--goal-lat LAT --goal-lon LON]
source ../p1-host-tools/venv/bin/activate
python3 gps_server.py "$@"
