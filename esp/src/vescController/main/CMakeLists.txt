idf_component_register(
  SRCS
  "main.cpp"
  "gps/gpsUsbHost.cpp"
  
  "sensors/cameraSensor.cpp"
  "sensors/gpsSensor.cpp"
  "sensors/wifiControlServerSensor.cpp"

  "algo/gpsGoalAlgo.cpp"
  "algo/cameraDrivingAlgo.cpp"
  "algo/userControllerAlgo.cpp"
  "algo/closeObstacleAvoidanceAlgo.cpp"

  "manager/MasterManager.cpp"
  "manager/AdvancedFusionEngine.cpp"
  "manager/CoupeCircuitManager.cpp"

  "lidarReader.cpp"
  "algo/corridorLidar/drive.cpp"
  "vesc/PhysicalVescController.cpp"
  "vescLidarUart.c"
  "rmt-uart/rmt_uart.c"

  PRIV_REQUIRES
  driver
  esp_wifi
  esp_netif
  esp_event
  nvs_flash
  lwip
  esp_http_server
  usb
  esp_driver_uart
  esp_driver_gpio

  INCLUDE_DIRS
  "."
  "manager"
  "gps"
  "rmt-uart"
  "sensors"
)
