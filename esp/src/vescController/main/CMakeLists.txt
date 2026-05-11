idf_component_register(
  SRCS
  "main.cpp"
  "gps/gpsUsbHost.cpp"
  
  "sensors/gpsSensor.cpp"
  "sensors/wifiControlServerSensor.cpp"

  "algo/gpsGoalAlgo.cpp"
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
  bt
  driver
  nvs_flash
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
