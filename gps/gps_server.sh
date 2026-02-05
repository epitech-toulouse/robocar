#!/bin/bash
# Lance le serveur GPS qui parse les données et les expose en JSON
source ../p1-host-tools/venv/bin/activate
python3 gps_server.py
