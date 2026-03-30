#include "usb_gps.hpp"

#include "esp_err.h"
#include "esp_log.h"
#include "sdkconfig.h"
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#include "driver/gpio.h"
#include "usb/cdc_acm_host.h"
#include "usb/cdc_host_types.h"
#include "usb/usb_host.h"
#include "usb/vcp_ch34x.h"
#include "usb/vcp_cp210x.h"
#include "usb/vcp_ftdi.h"

#define EXAMPLE_USB_HOST_PRIORITY (20)
#define MAX_CDC_DEVICES (5)
#define ESPRESSIF_VID (0x303A)
#define GPS_BAUD_RATE 460800

static const char *TAG = "USB_GPS";

static cdc_acm_dev_hdl_t cdc_devices[MAX_CDC_DEVICES] = {0};

static QueueHandle_t app_queue;
typedef struct {
  enum {
    APP_QUIT,
    APP_DEVICE_CONNECTED,
    APP_DEVICE_DISCONNECTED,
  } id;
  union {
    struct {
      uint16_t vid;
      uint16_t pid;
    } new_dev;
    int device_slot;
  } data;
} app_message_t;

// State holding the latest GPS coordinates
static struct GPSPoint latest_gps_state = {0};

struct GPSPoint get_latest_gps(void) {
    // Return a copy. Could be protected by mutex if needed, but atomic float reads are usually fine enough for this use case.
    return latest_gps_state;
}

// ----------------------------------------------------------------------------
// MATH UTILITIES
// ----------------------------------------------------------------------------

float wrap_180(float angle_deg) {
    while (angle_deg > 180.0f) angle_deg -= 360.0f;
    while (angle_deg <= -180.0f) angle_deg += 360.0f;
    return angle_deg;
}

static float deg2rad(float deg) {
    return deg * (M_PI / 180.0f);
}

static float rad2deg(float rad) {
    return rad * (180.0f / M_PI);
}

float distance_haversine_m(float lat1, float lon1, float lat2, float lon2) {
    float R = 6371000.0f; // Earth radius in meters
    float dLat = deg2rad(lat2 - lat1);
    float dLon = deg2rad(lon2 - lon1);
    
    float a = sinf(dLat/2) * sinf(dLat/2) + 
              cosf(deg2rad(lat1)) * cosf(deg2rad(lat2)) * 
              sinf(dLon/2) * sinf(dLon/2);
    float c = 2.0f * atan2f(sqrtf(a), sqrtf(1.0f - a));
    return R * c;
}

float initial_bearing_deg(float lat1, float lon1, float lat2, float lon2) {
    float lat1_rad = deg2rad(lat1);
    float lat2_rad = deg2rad(lat2);
    float dLon_rad = deg2rad(lon2 - lon1);
    
    float y = sinf(dLon_rad) * cosf(lat2_rad);
    float x = cosf(lat1_rad)*sinf(lat2_rad) - sinf(lat1_rad)*cosf(lat2_rad)*cosf(dLon_rad);
    
    float bearing = rad2deg(atan2f(y, x));
    if (bearing < 0.0f) bearing += 360.0f;
    return bearing;
}

// ----------------------------------------------------------------------------
// NMEA PARSING LOGIC
// ----------------------------------------------------------------------------
static float nmea_to_decimal(const char *nmea_coord, char dir) {
  if (!nmea_coord || strlen(nmea_coord) < 4)
    return 0.0f;

  char deg_str[4] = {0};
  const char *dot = strchr(nmea_coord, '.');
  if (!dot)
    return 0.0f;

  int min_len = 2; // Minutes have 2 digits before the dot
  int deg_len = (dot - nmea_coord) - min_len;

  if (deg_len <= 0 || deg_len > 3)
    return 0.0f;

  strncpy(deg_str, nmea_coord, deg_len);
  float degrees = atof(deg_str);
  float minutes = atof(nmea_coord + deg_len);

  float decimal = degrees + (minutes / 60.0f);
  if (dir == 'S' || dir == 'W') {
    decimal = -decimal;
  }
  return decimal;
}

static void parse_nmea(char *line) {
  char *tokens[25];
  int count = 0;

  tokens[count++] = line;
  for (int i = 0; line[i] != '\0'; i++) {
    if (line[i] == ',' || line[i] == '*') {
      line[i] = '\0';
      if (count < 25) {
        tokens[count++] = &line[i + 1];
      }
    }
  }

  if (strncmp(tokens[0], "$GNGGA", 6) == 0 || strncmp(tokens[0], "$GPGGA", 6) == 0) {
    // 2=Lat, 3=N/S, 4=Lon, 5=E/W, 6=Fix, 7=Sats, 9=Alt
    if (count >= 10 && strlen(tokens[2]) > 0 && strlen(tokens[4]) > 0) {
      float lat = nmea_to_decimal(tokens[2], tokens[3][0]);
      float lon = nmea_to_decimal(tokens[4], tokens[5][0]);
      int fix = atoi(tokens[6]);
      int sats = atoi(tokens[7]);
      float alt = atof(tokens[9]);

      latest_gps_state.sats = sats;
      latest_gps_state.alt = alt;

      if (fix > 0) {
        latest_gps_state.lat = lat;
        latest_gps_state.lon = lon;
        latest_gps_state.has_fix = true;
      } else {
        latest_gps_state.has_fix = false;
      }
    } else if (count >= 8) {
      latest_gps_state.sats = atoi(tokens[7]);
      latest_gps_state.has_fix = false;
    }
  } 
  else if (strncmp(tokens[0], "$GNRMC", 6) == 0 || strncmp(tokens[0], "$GPRMC", 6) == 0) {
    // 2=Active/Void, 3=Lat, 4=N/S, 5=Lon, 6=E/W, 7=Speed, 8=Heading
    if (count >= 9 && tokens[2][0] == 'A') {
      latest_gps_state.has_fix = true;
      latest_gps_state.lat = nmea_to_decimal(tokens[3], tokens[4][0]);
      latest_gps_state.lon = nmea_to_decimal(tokens[5], tokens[6][0]);
      if (strlen(tokens[7]) > 0) {
          latest_gps_state.speed_knots = atof(tokens[7]);
      }
      if (strlen(tokens[8]) > 0) {
          latest_gps_state.heading = atof(tokens[8]);
      }
    } else {
      latest_gps_state.has_fix = false;
    }
  }
}

// ----------------------------------------------------------------------------
// USB CALLBACKS
// ----------------------------------------------------------------------------

static inline int find_free_slot(void) {
  for (int i = 0; i < MAX_CDC_DEVICES; i++) {
    if (cdc_devices[i] == NULL) return i;
  }
  return -1;
}

static char line_buf[256];
static int line_pos = 0;

static bool handle_rx(const uint8_t *data, size_t data_len, void *arg) {
  for (size_t i = 0; i < data_len; i++) {
    char c = data[i];
    if (c == '$') {
      line_pos = 0;
      line_buf[line_pos++] = c;
    } else if (line_pos > 0) {
      if (c == '\r' || c == '\n' || line_pos >= 255) {
        line_buf[line_pos] = '\0';
        if (line_pos > 10) {
          parse_nmea(line_buf); 
        }
        line_pos = 0; 
      } else {
        line_buf[line_pos++] = c;
      }
    }
  }
  return true; 
}

static void handle_event(const cdc_acm_host_dev_event_data_t *event, void *user_ctx) {
  switch (event->type) {
  case CDC_ACM_HOST_ERROR:
    break;
  case CDC_ACM_HOST_DEVICE_DISCONNECTED:
    if (app_queue) {
      app_message_t msg = {
          .id = APP_DEVICE_DISCONNECTED,
          .data = { .device_slot = (int)(intptr_t)user_ctx }
      };
      xQueueSend(app_queue, &msg, 0);
    } else {
      cdc_acm_host_close(event->data.cdc_hdl);
    }
    break;
  default:
    break;
  }
}

static void new_dev_cb(usb_device_handle_t usb_dev) {
  const usb_device_desc_t *device_desc;
  if (usb_host_get_device_descriptor(usb_dev, &device_desc) != ESP_OK) return;

  if (app_queue) {
    app_message_t msg = {
        .id = APP_DEVICE_CONNECTED,
        .data = { .new_dev = { .vid = device_desc->idVendor, .pid = device_desc->idProduct } }
    };
    xQueueSend(app_queue, &msg, 0);
  }
}

static cdc_acm_dev_hdl_t example_cdc_open(uint16_t vid, uint16_t pid, const cdc_acm_host_device_config_t *dev_config) {
  cdc_acm_dev_hdl_t cdc_dev = NULL;
  esp_err_t err;

  switch (vid) {
  case FTDI_VID: err = ftdi_vcp_open(pid, 0, dev_config, &cdc_dev); break;
  case NANJING_QINHENG_MICROE_VID: err = ch34x_vcp_open(pid, 0, dev_config, &cdc_dev); break;
  case SILICON_LABS_VID: err = cp210x_vcp_open(pid, 0, dev_config, &cdc_dev); break;
  default: err = cdc_acm_host_open(vid, pid, 0, dev_config, &cdc_dev); break;
  }
  return (err == ESP_OK) ? cdc_dev : NULL;
}

static void free_cdc_device(int slot) {
  if (slot < 0 || slot >= MAX_CDC_DEVICES || cdc_devices[slot] == NULL) return;
  cdc_acm_host_close(cdc_devices[slot]);
  cdc_devices[slot] = NULL;
}

static void free_all_cdc_devices(void) {
  for (int i = 0; i < MAX_CDC_DEVICES; i++) {
    if (cdc_devices[i] != NULL) free_cdc_device(i);
  }
}

static void configure_gps_usb(int slot) {
  cdc_acm_dev_hdl_t cdc_dev = cdc_devices[slot];
  cdc_acm_host_set_control_line_state(cdc_dev, true, true);
  vTaskDelay(pdMS_TO_TICKS(100));

  cdc_acm_line_coding_t line_coding;
  if (cdc_acm_host_line_coding_get(cdc_dev, &line_coding) == ESP_OK) {
    line_coding.dwDTERate = GPS_BAUD_RATE;
    line_coding.bDataBits = 8;
    line_coding.bParityType = 0;
    line_coding.bCharFormat = 0;
    cdc_acm_host_line_coding_set(cdc_dev, &line_coding);
  } else {
    line_coding.dwDTERate = GPS_BAUD_RATE;
    line_coding.bDataBits = 8;
    line_coding.bParityType = 0;
    line_coding.bCharFormat = 0;
    cdc_acm_host_line_coding_set(cdc_dev, &line_coding);
  }
}

// ----------------------------------------------------------------------------
// MAIN TASK
// ----------------------------------------------------------------------------

static void usb_lib_task(void *arg) {
  const usb_host_config_t host_config = { .skip_phy_setup = false, .intr_flags = ESP_INTR_FLAG_LOWMED };
  ESP_ERROR_CHECK(usb_host_install(&host_config));

  const cdc_acm_host_driver_config_t driver_config = {
      .driver_task_stack_size = 4096,
      .driver_task_priority = EXAMPLE_USB_HOST_PRIORITY + 1,
      .xCoreID = 0,
      .new_dev_cb = new_dev_cb,
  };
  ESP_ERROR_CHECK(cdc_acm_host_install(&driver_config));
  xTaskNotifyGive(arg);

  bool has_clients = true;
  while (1) {
    uint32_t event_flags;
    usb_host_lib_handle_events(portMAX_DELAY, &event_flags);
    if (event_flags & USB_HOST_LIB_EVENT_FLAGS_NO_CLIENTS) {
      has_clients = false;
      if (ESP_OK == usb_host_device_free_all()) break;
    }
    if (event_flags & USB_HOST_LIB_EVENT_FLAGS_ALL_FREE) {
      if (!has_clients) break;
    }
  }

  vTaskDelay(pdMS_TO_TICKS(10));
  ESP_ERROR_CHECK(usb_host_uninstall());
  vTaskDelete(NULL);
}

static void gps_usb_manager_task(void *arg) {
  app_queue = xQueueCreate(10, sizeof(app_message_t));
  assert(app_queue);

  BaseType_t task_created = xTaskCreate(usb_lib_task, "usb_lib", 4096, xTaskGetCurrentTaskHandle(), EXAMPLE_USB_HOST_PRIORITY, NULL);
  assert(task_created == pdTRUE);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

  cdc_acm_host_device_config_t dev_config = {
      .connection_timeout_ms = 0,
      .out_buffer_size = 512,
      .in_buffer_size = 2048, 
      .user_arg = NULL,
      .event_cb = handle_event,
      .data_cb = handle_rx
  };

  while (true) {
    app_message_t msg;
    xQueueReceive(app_queue, &msg, portMAX_DELAY);

    switch (msg.id) {
    case APP_DEVICE_CONNECTED: {
      int slot = find_free_slot();
      if (slot < 0) continue;

      dev_config.user_arg = (void *)(intptr_t)slot;
      cdc_acm_dev_hdl_t cdc_dev = example_cdc_open(msg.data.new_dev.vid, msg.data.new_dev.pid, &dev_config);
      if (cdc_dev == NULL) continue;

      cdc_devices[slot] = cdc_dev;
      configure_gps_usb(slot);
      break;
    }
    case APP_DEVICE_DISCONNECTED: {
      free_cdc_device(msg.data.device_slot);
      break;
    }
    case APP_QUIT:
      free_all_cdc_devices();
      cdc_acm_host_uninstall();
      goto exit;
    default:
      break;
    }
  }
exit:
  vQueueDelete(app_queue);
  vTaskDelete(NULL);
}

void init_usb_gps(void) {
    xTaskCreate(gps_usb_manager_task, "gps_manager", 4096, NULL, 5, NULL);
}
