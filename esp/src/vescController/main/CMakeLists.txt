idf_component_register(
  SRCS
  "main.cpp"
  "manager/MasterManager.cpp"
  "manager/AdvancedFusionEngine.cpp"
  "manager/CoupeCircuitManager.cpp"
  "wifi_control_server.cpp"
  "lidarReader.cpp"
  "drive.cpp"
  "vesc/PhysicalVescController.cpp"
  "vescLidarUart.c"

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
)
