source $HOME/.espressif/tools/activate_idf_v5.5.3.sh
idf.py set-target esp32s3
sudo chmod 777 /dev/ttyACM0
idf.py -p /dev/ttyACM0 flash monitor