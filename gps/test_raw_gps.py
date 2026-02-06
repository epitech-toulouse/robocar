#!/usr/bin/env python3
"""
Test GPS - Affiche la latitude brute sans parsing
"""

import os
import socket
import sys
import time

# Add the Python root directory for FusionEngine imports
root_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from fusion_engine_client.messages.core import PoseMessage
from fusion_engine_client.parsers import FusionEngineDecoder


def main():
    source_host = 'localhost'
    source_port = 25000
    
    print(f"📡 Connexion GPS {source_host}:{source_port}...")
    
    try:
        transport = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        transport.connect((socket.gethostbyname(source_host), source_port))
        print(f"✅ Connecté au GPS\n")
        
        decoder = FusionEngineDecoder()
        
        while True:
            received_data = transport.recv(4096)
            if not received_data:
                break
            
            messages = decoder.on_data(received_data)
            
            for header, message in messages:
                if isinstance(message, PoseMessage):
                    lat = message.lla_deg[0]
                    lon = message.lla_deg[1]
                    alt = message.lla_deg[2]
                    
                    print(f"LAT: {lat}")
                    print(f"LON: {lon}")
                    print(f"ALT: {alt}")
                    print("-" * 40)
                    break
    
    except KeyboardInterrupt:
        print("\n🛑 Arrêt")
    except Exception as e:
        print(f"❌ Erreur: {e}")
    finally:
        transport.close()


if __name__ == "__main__":
    main()
