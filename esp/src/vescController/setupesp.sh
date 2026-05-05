source $HOME/esp/esp-idf/export.sh
idf.py set-target esp32s3
sudo chmod 777 /dev/ttyACM0
idf.py -p /dev/ttyACM0 flash monitor