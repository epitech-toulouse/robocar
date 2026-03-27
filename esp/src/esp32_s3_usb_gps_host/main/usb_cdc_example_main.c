/*
 * Custom ESP32-S3 USB Host CDC -> NMEA GPS Parser
 * Based on ESP-IDF usb_host_cdc_acm example
 */

#include "esp_err.h"
#include "esp_log.h"
#include "sdkconfig.h"
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// La vitesse pour le Point One Navigation est 460800
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

// ----------------------------------------------------------------------------
// NMEA PARSING LOGIC
// ----------------------------------------------------------------------------
float nmea_to_decimal(const char *nmea_coord, char dir) {
  if (!nmea_coord || strlen(nmea_coord) < 4)
    return 0.0;

  char deg_str[4] = {0};
  const char *dot = strchr(nmea_coord, '.');
  if (!dot)
    return 0.0;

  int min_len = 2; // Minutes have 2 digits before the dot
  int deg_len = (dot - nmea_coord) - min_len;

  if (deg_len <= 0 || deg_len > 3)
    return 0.0;

  strncpy(deg_str, nmea_coord, deg_len);
  float degrees = atof(deg_str);
  float minutes = atof(nmea_coord + deg_len);

  float decimal = degrees + (minutes / 60.0);
  if (dir == 'S' || dir == 'W') {
    decimal = -decimal;
  }
  return decimal;
}

void parse_nmea(char *line) {
  if (strncmp(line, "$GNGGA", 6) == 0 || strncmp(line, "$GPGGA", 6) == 0) {
    char *tokens[20];
    int count = 0;

    tokens[count++] = line;
    for (int i = 0; line[i] != '\0'; i++) {
      if (line[i] == ',' || line[i] == '*') {
        line[i] = '\0';
        if (count < 20) {
          tokens[count++] = &line[i + 1];
        }
      }
    }

    // 2=Lat, 3=N/S, 4=Lon, 5=E/W, 6=Fix, 7=Sats, 9=Alt
    if (count >= 10 && strlen(tokens[2]) > 0 && strlen(tokens[4]) > 0) {
      float lat = nmea_to_decimal(tokens[2], tokens[3][0]);
      float lon = nmea_to_decimal(tokens[4], tokens[5][0]);
      int fix = atoi(tokens[6]);
      int sats = atoi(tokens[7]);
      float alt = atof(tokens[9]);

      if (fix > 0) {
        printf("\r[ ESP32-S3 USB ] ✅ Fix OK | Sats: %d | Lat: %.6f | Lon: "
               "%.6f | Alt: %.1fm      \n",
               sats, lat, lon, alt);
      } else {
        printf("\r[ ESP32-S3 USB ] ⏳ Fix en cours... | Sats vus: %d           "
               "                ",
               sats);
      }
    } else if (count >= 8) {
      int sats = atoi(tokens[7]);
      printf("\r[ ESP32-S3 USB ] ⏳ Recherche satellites... | Sats vus: %d     "
             "                ",
             sats);
    }
  }
}

// ----------------------------------------------------------------------------
// USB CALLBACKS
// ----------------------------------------------------------------------------

static inline int find_free_slot(void) {
  for (int i = 0; i < MAX_CDC_DEVICES; i++) {
    if (cdc_devices[i] == NULL) {
      return i;
    }
  }
  return -1;
}

// Buffer statique pour reconstituer les lignes NMEA reçues via USB
static char line_buf[256];
static int line_pos = 0;

static bool handle_rx(const uint8_t *data, size_t data_len, void *arg) {
  for (int i = 0; i < data_len; i++) {
    char c = data[i];

    // Début de trame NMEA
    if (c == '$') {
      line_pos = 0;
      line_buf[line_pos++] = c;
    } else if (line_pos > 0) {
      // Fin de trame
      if (c == '\r' || c == '\n' || line_pos >= 255) {
        line_buf[line_pos] = '\0';
        if (line_pos > 10) {
          parse_nmea(line_buf); // Décodage
        }
        line_pos = 0; // Reset
      } else {
        line_buf[line_pos++] = c;
      }
    }
  }
  return true; // Prêt à recevoir plus
}

static void handle_event(const cdc_acm_host_dev_event_data_t *event,
                         void *user_ctx) {
  switch (event->type) {
  case CDC_ACM_HOST_ERROR:
    ESP_LOGE(TAG, "CDC-ACM error has occurred, err_no = %i", event->data.error);
    break;
  case CDC_ACM_HOST_DEVICE_DISCONNECTED:
    if (app_queue) {
      app_message_t msg = {
          .id = APP_DEVICE_DISCONNECTED,
          .data.device_slot = (int)(intptr_t)user_ctx,
      };
      xQueueSend(app_queue, &msg, 0);
    } else {
      ESP_ERROR_CHECK(cdc_acm_host_close(event->data.cdc_hdl));
    }
    break;
  case CDC_ACM_HOST_SERIAL_STATE:
    ESP_LOGI(TAG, "Serial state notif 0x%04X", event->data.serial_state.val);
    break;
  default:
    break;
  }
}

static void new_dev_cb(usb_device_handle_t usb_dev) {
  const usb_device_desc_t *device_desc;
  esp_err_t err = usb_host_get_device_descriptor(usb_dev, &device_desc);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "usb_host_get_device_descriptor failed: %s",
             esp_err_to_name(err));
    return;
  }

  uint16_t vid = device_desc->idVendor;
  uint16_t pid = device_desc->idProduct;
  ESP_LOGI(TAG, "New USB Device connected: VID=0x%04X PID=0x%04X", vid, pid);

  const usb_config_desc_t *config_desc = NULL;
  err = usb_host_get_active_config_descriptor(usb_dev, &config_desc);
  if (err == ESP_OK && config_desc != NULL) {
    uint32_t requested_ma = (uint32_t)config_desc->bMaxPower * 2;
    bool self_powered = (config_desc->bmAttributes & USB_BM_ATTRIBUTES_SELFPOWER) != 0;
    ESP_LOGI(TAG,
             "USB power profile: %s, bus current request=%" PRIu32 " mA (bMaxPower=%u)",
             self_powered ? "self-powered" : "bus-powered", requested_ma,
             config_desc->bMaxPower);
  } else {
    ESP_LOGW(TAG, "Unable to read active USB config descriptor (power unknown): %s",
             esp_err_to_name(err));
  }

  if (app_queue) {
    app_message_t msg = {
        .id = APP_DEVICE_CONNECTED,
        .data.new_dev.vid = vid,
        .data.new_dev.pid = pid,
    };
    xQueueSend(app_queue, &msg, 0);
  }
}

static cdc_acm_dev_hdl_t
example_cdc_open(uint16_t vid, uint16_t pid,
                 const cdc_acm_host_device_config_t *dev_config) {
  cdc_acm_dev_hdl_t cdc_dev = NULL;
  esp_err_t err;

  switch (vid) {
  case FTDI_VID:
    err = ftdi_vcp_open(pid, 0, dev_config, &cdc_dev);
    break;
  case NANJING_QINHENG_MICROE_VID:
    err = ch34x_vcp_open(pid, 0, dev_config, &cdc_dev);
    break;
  case SILICON_LABS_VID:
    err = cp210x_vcp_open(pid, 0, dev_config, &cdc_dev);
    break;
  default:
    // Pour la majorité des GPS / STM32 etc
    err = cdc_acm_host_open(vid, pid, 0, dev_config, &cdc_dev);
    break;
  }

  if (err == ESP_OK)
    return cdc_dev;

  ESP_LOGE(TAG,
           "Failed to open device VID=0x%04X PID=0x%04X (Make sure it has a "
           "CDC interface)",
           vid, pid);
  return NULL;
}

static void free_cdc_device(int slot) {
  if (slot < 0 || slot >= MAX_CDC_DEVICES || cdc_devices[slot] == NULL)
    return;
  ESP_LOGI(TAG, "\t- Closing CDC device in slot %d", slot);
  cdc_acm_host_close(cdc_devices[slot]);
  cdc_devices[slot] = NULL;
}

static void free_all_cdc_devices(void) {
  for (int i = 0; i < MAX_CDC_DEVICES; i++) {
    if (cdc_devices[i] != NULL)
      free_cdc_device(i);
  }
}

static void configure_gps_usb(int slot) {
  cdc_acm_dev_hdl_t cdc_dev = cdc_devices[slot];
  esp_err_t err;

  ESP_LOGI(TAG, "Device opened (slot %d). Configuring GPS Baudrate to %d...",
           slot, GPS_BAUD_RATE);

  // Set DTR=true, RTS=true to tell the device we are ready
  cdc_acm_host_set_control_line_state(cdc_dev, true, true);
  vTaskDelay(pdMS_TO_TICKS(100));

  // Get current line coding
  cdc_acm_line_coding_t line_coding;
  err = cdc_acm_host_line_coding_get(cdc_dev, &line_coding);

  if (err == ESP_OK) {
    // Appliquer le Baudrate du Point One GPS
    line_coding.dwDTERate = GPS_BAUD_RATE;
    line_coding.bDataBits = 8;
    line_coding.bParityType = 0;
    line_coding.bCharFormat = 0;
    err = cdc_acm_host_line_coding_set(cdc_dev, &line_coding);
    if (err == ESP_OK) {
      ESP_LOGI(TAG, "✅ Baudrate applied successfully: %" PRIu32 " bps",
               line_coding.dwDTERate);
    } else {
      ESP_LOGE(TAG, "❌ Failed to set GPS Baudrate");
    }
  } else {
    // Some devices don't support get, we try to force set anyway
    line_coding.dwDTERate = GPS_BAUD_RATE;
    line_coding.bDataBits = 8;
    line_coding.bParityType = 0;
    line_coding.bCharFormat = 0;
    cdc_acm_host_line_coding_set(cdc_dev, &line_coding);
    ESP_LOGW(TAG, "Could not GET line coding, but tried to SET it anyway.");
  }

  ESP_LOGI(TAG, "GPS Ready! Waiting for NMEA data...");
}

// ----------------------------------------------------------------------------
// MAIN TASK
// ----------------------------------------------------------------------------

static void usb_lib_task(void *arg) {
  ESP_LOGI("usb_task", "Running USB task");
  const usb_host_config_t host_config = {
      .skip_phy_setup = false,
      .intr_flags = ESP_INTR_FLAG_LOWMED,
  };
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
      if (ESP_OK == usb_host_device_free_all())
        break;
    }
    if (event_flags & USB_HOST_LIB_EVENT_FLAGS_ALL_FREE) {
      if (!has_clients)
        break;
    }
  }

  vTaskDelay(pdMS_TO_TICKS(10));
  ESP_ERROR_CHECK(usb_host_uninstall());
  vTaskDelete(NULL);
}

void app_main(void) {
  ESP_LOGI(TAG, "--- DEMARRAGE ESP32-S3 USB HOST GPS ---");
  ESP_LOGI(TAG, "En attente du branchement du module GPS sur le port USB natif "
                "de l'ESP32...");

  app_queue = xQueueCreate(10, sizeof(app_message_t));
  assert(app_queue);

  BaseType_t task_created =
      xTaskCreate(usb_lib_task, "usb_lib", 4096, xTaskGetCurrentTaskHandle(),
                  EXAMPLE_USB_HOST_PRIORITY, NULL);
  assert(task_created == pdTRUE);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

  // Initialisation Callback Config pour le port CDC
  cdc_acm_host_device_config_t dev_config = {
      .connection_timeout_ms = 0,
      .out_buffer_size = 512,
      .in_buffer_size = 2048, // Agrandissement du buffer IN pour les flots de
                              // données NMEA/binaires intenses
      .user_arg = NULL,
      .event_cb = handle_event,
      .data_cb = handle_rx};

  while (true) {
    app_message_t msg;
    xQueueReceive(app_queue, &msg, portMAX_DELAY);

    switch (msg.id) {
    case APP_DEVICE_CONNECTED: {
      int slot = find_free_slot();
      if (slot < 0)
        continue;

      dev_config.user_arg = (void *)(intptr_t)slot;
      cdc_acm_dev_hdl_t cdc_dev = example_cdc_open(
          msg.data.new_dev.vid, msg.data.new_dev.pid, &dev_config);
      if (cdc_dev == NULL)
        continue;

      cdc_devices[slot] = cdc_dev;
      configure_gps_usb(slot);
      break;
    }
    case APP_DEVICE_DISCONNECTED: {
      ESP_LOGW(TAG, "GPS disconnected!");
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
}
