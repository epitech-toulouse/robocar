#!/bin/bash
# Test rapide du lecteur GPS simple

HOST=${1:-localhost}
PORT=${2:-25000}

echo "Test du lecteur GPS sur $HOST:$PORT"
python3 gps_simple_reader.py "$HOST" "$PORT"
