#!/bin/bash
#run the gps runner 1st argument should be the device port e.g /dev/ttyUSB0 the 2nd is the polaris id
#send data to tcp port 25000
source ../p1-host-tools/venv/bin/activate
cd ../p1-host-tools
sudo chmod +x $1
python3 bin/runner.py --device-port $1 --device-id gps_device --polaris $2 --tcp 25000