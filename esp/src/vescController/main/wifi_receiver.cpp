#include "wifi_receiver.hpp"

#include <atomic>
#include <cstring>
#include <cstdio>

#include "esp_err.h"
#include "esp_event.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"
#include "nvs_flash.h"

static const char *TAG = "wifi_recv";

static std::atomic<float> s_duty{0.0f};
static std::atomic<float> s_steer{0.5f};
static std::atomic<int> s_last_tick{0};
static std::atomic<bool> s_forward{false};
static std::atomic<bool> s_backward{false};
static std::atomic<bool> s_left{false};
static std::atomic<bool> s_right{false};
static std::atomic<bool> s_emergency{false};

static constexpr float STEER_CENTER = 0.5f;
static constexpr float STEER_LEFT = 0.2f;
static constexpr float STEER_RIGHT = 0.8f;
static constexpr float DUTY_FORWARD = 0.05f;
static constexpr float DUTY_BACKWARD = -0.05f;

static constexpr TickType_t MANUAL_TIMEOUT_MS = 2000;

static constexpr const char *WIFI_AP_SSID = "ROBOCAR_CTRL";
static constexpr const char *WIFI_AP_PASSWORD = "robocar123";
static constexpr uint8_t WIFI_AP_CHANNEL = 1;
static constexpr uint8_t WIFI_AP_MAX_CONN = 1;
static constexpr int CONTROL_HTTP_PORT = 3333;
static constexpr int CONTROL_TCP_PORT = 3334;
static constexpr int RX_BUFFER_SIZE = 64;

static const char *INDEX_HTML = R"HTML(
<!doctype html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>RoboCar Control</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 0; padding: 16px; background: #f3f5f8; }
        .card { max-width: 560px; margin: 0 auto; background: #fff; border-radius: 14px; box-shadow: 0 10px 24px rgba(0,0,0,.12); padding: 16px; }
        h1 { margin: 0 0 10px 0; font-size: 22px; }
        .row { display: flex; gap: 10px; margin-bottom: 10px; }
        button { border: 0; border-radius: 10px; padding: 14px; font-size: 16px; font-weight: 700; }
        .ok { background: #2e7d32; color: #fff; }
        .stop { background: #c62828; color: #fff; }
        .ctl { background: #1976d2; color: #fff; flex: 1; }
        .log { margin-top: 10px; height: 160px; overflow: auto; background: #10141a; color: #d7e2ee; border-radius: 10px; padding: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
        .muted { color: #556; font-size: 13px; margin-bottom: 10px; }
    </style>
</head>
<body>
    <div class="card">
        <h1>RoboCar Control</h1>
        <div class="muted">Open from iPhone: http://192.168.4.1:3333</div>
        <div class="row"><button id="connect" class="ok" style="flex:1">Connect</button><button id="disconnect" style="flex:1">Disconnect</button></div>
        <div class="row"><button id="f" class="ctl">Forward</button></div>
        <div class="row"><button id="l" class="ctl">Left</button><button id="r" class="ctl">Right</button></div>
        <div class="row"><button id="b" class="ctl">Backward</button></div>
        <div class="row"><button id="s" class="stop" style="flex:1">EMERGENCY STOP</button></div>
        <div id="log" class="log"></div>
    </div>

    <script>
        let connected = false;
        const logEl = document.getElementById('log');
        function log(msg) {
            const t = new Date().toLocaleTimeString();
            logEl.innerHTML += '[' + t + '] ' + msg + '<br>';
            logEl.scrollTop = logEl.scrollHeight;
        }

        async function api(path) {
            const r = await fetch(path, { method: 'GET', cache: 'no-store' });
            if (!r.ok) throw new Error('HTTP ' + r.status);
            return r.text();
        }

        async function connect() {
            try {
                await api('/status');
                connected = true;
                log('Connected');
            } catch (e) {
                connected = false;
                log('Connect failed: ' + (e.message || e));
            }
        }

        function disconnect() {
            connected = false;
            log('Disconnected');
        }

        async function send(c) {
            if (!connected) {
                log('Not connected');
                return;
            }
            try {
                await api('/cmd?c=' + encodeURIComponent(c));
                log('Sent ' + c);
            } catch (e) {
                log('Send failed: ' + (e.message || e));
            }
        }

        function bindHold(id, down, up) {
            const el = document.getElementById(id);
            let pressed = false;
            const p = (e) => { e.preventDefault(); if (pressed) return; pressed = true; send(down); };
            const r = (e) => { e.preventDefault(); if (!pressed) return; pressed = false; send(up); };
            el.addEventListener('pointerdown', p);
            el.addEventListener('pointerup', r);
            el.addEventListener('pointercancel', r);
            el.addEventListener('touchstart', p, { passive: false });
            el.addEventListener('touchend', r, { passive: false });
            el.addEventListener('touchcancel', r, { passive: false });
            el.addEventListener('mousedown', p);
            el.addEventListener('mouseup', r);
            el.addEventListener('mouseleave', r);
        }

        document.getElementById('connect').addEventListener('click', connect);
        document.getElementById('disconnect').addEventListener('click', disconnect);
        document.getElementById('s').addEventListener('click', () => send('S'));
        bindHold('f', 'F', 'f');
        bindHold('l', 'L', 'l');
        bindHold('r', 'R', 'r');
        bindHold('b', 'B', 'b');
        log('Ready');
    </script>
</body>
</html>
)HTML";

static void set_http_common_headers(httpd_req_t *req) {
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Methods", "GET, OPTIONS");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Headers", "Content-Type");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Private-Network", "true");
        httpd_resp_set_hdr(req, "Cache-Control", "no-store");
}

static void recompute_output_from_state() {
    float duty = 0.0f;
    if (s_forward.load() && !s_backward.load()) {
        duty = DUTY_FORWARD;
    } else if (s_backward.load() && !s_forward.load()) {
        duty = DUTY_BACKWARD;
    }

    float steer = STEER_CENTER;
    if (s_left.load() && !s_right.load()) {
        steer = STEER_LEFT;
    } else if (s_right.load() && !s_left.load()) {
        steer = STEER_RIGHT;
    }

    s_duty.store(duty);
    s_steer.store(steer);
    s_last_tick.store(static_cast<int>(xTaskGetTickCount()));
}

static void emergency_stop() {
    s_emergency.store(true);
    s_forward.store(false);
    s_backward.store(false);
    s_left.store(false);
    s_right.store(false);
    s_duty.store(0.0f);
    s_steer.store(STEER_CENTER);
    s_last_tick.store(static_cast<int>(xTaskGetTickCount()));
}

static bool apply_protocol_char(char c) {
    switch (c) {
        case 'F': s_forward.store(true);  recompute_output_from_state(); return true;
        case 'f': s_forward.store(false); recompute_output_from_state(); return true;
        case 'B': s_backward.store(true);  recompute_output_from_state(); return true;
        case 'b': s_backward.store(false); recompute_output_from_state(); return true;
        case 'L': s_left.store(true);  recompute_output_from_state(); return true;
        case 'l': s_left.store(false); recompute_output_from_state(); return true;
        case 'R': s_right.store(true);  recompute_output_from_state(); return true;
        case 'r': s_right.store(false); recompute_output_from_state(); return true;
        case 'S': emergency_stop(); return true;
        default: return false;
    }
}

static void parse_and_store(const uint8_t *buf, int len) {
    // Protocol: F/f/B/b/L/l/R/r/S (1 ASCII char per action)
    if (len <= 0 || buf == nullptr) {
        return;
    }

    for (int i = 0; i < len; ++i) {
        const uint8_t b = buf[i];
        if (b == 0x00 || b == '\n' || b == '\r' || b == ' ' || b == '\t') {
            continue;
        }
        if (!apply_protocol_char(static_cast<char>(b))) {
            ESP_LOGW(TAG, "Unknown protocol char: '%c' (0x%02X)", static_cast<char>(b), b);
        }
    }

    ESP_LOGI(TAG, "State F=%d B=%d L=%d R=%d -> duty=%.3f steer=%.3f",
             s_forward.load(), s_backward.load(), s_left.load(), s_right.load(),
             s_duty.load(), s_steer.load());
}

static void wifi_control_server_task(void *arg) {
    (void)arg;
    uint8_t rx_buf[RX_BUFFER_SIZE];

    while (true) {
        int listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
        if (listen_sock < 0) {
            ESP_LOGE(TAG, "Unable to create socket: errno=%d", errno);
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        int opt = 1;
        setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(CONTROL_TCP_PORT);
        addr.sin_addr.s_addr = htonl(INADDR_ANY);

        if (bind(listen_sock, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
            ESP_LOGE(TAG, "Socket bind failed: errno=%d", errno);
            close(listen_sock);
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        if (listen(listen_sock, 1) < 0) {
            ESP_LOGE(TAG, "Socket listen failed: errno=%d", errno);
            close(listen_sock);
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        ESP_LOGI(TAG, "Wi-Fi control server listening on TCP port %d", CONTROL_TCP_PORT);

        while (true) {
            sockaddr_in6 source_addr = {};
            socklen_t addr_len = sizeof(source_addr);
            int sock = accept(listen_sock, reinterpret_cast<sockaddr *>(&source_addr), &addr_len);
            if (sock < 0) {
                ESP_LOGE(TAG, "Socket accept failed: errno=%d", errno);
                break;
            }

            ESP_LOGI(TAG, "Controller connected");
            while (true) {
                const int len = recv(sock, rx_buf, sizeof(rx_buf), 0);
                if (len < 0) {
                    ESP_LOGE(TAG, "Socket recv failed: errno=%d", errno);
                    break;
                }
                if (len == 0) {
                    ESP_LOGI(TAG, "Controller disconnected");
                    break;
                }
                parse_and_store(rx_buf, len);
            }

            emergency_stop();
            shutdown(sock, 0);
            close(sock);
        }

        close(listen_sock);
        vTaskDelay(pdMS_TO_TICKS(200));
    }
}

static void init_wifi_softap() {
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    wifi_config_t wifi_config = {};
    std::strncpy(reinterpret_cast<char *>(wifi_config.ap.ssid), WIFI_AP_SSID, sizeof(wifi_config.ap.ssid));
    std::strncpy(reinterpret_cast<char *>(wifi_config.ap.password), WIFI_AP_PASSWORD, sizeof(wifi_config.ap.password));
    wifi_config.ap.channel = WIFI_AP_CHANNEL;
    wifi_config.ap.max_connection = WIFI_AP_MAX_CONN;
    wifi_config.ap.ssid_len = std::strlen(WIFI_AP_SSID);
    wifi_config.ap.authmode = WIFI_AUTH_WPA2_PSK;

    if (std::strlen(WIFI_AP_PASSWORD) == 0) {
        wifi_config.ap.authmode = WIFI_AUTH_OPEN;
    }

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Wi-Fi AP started: ssid=%s channel=%u", WIFI_AP_SSID, WIFI_AP_CHANNEL);
}

// Minimal HTTP handler for /cmd?c=X
static esp_err_t http_cmd_handler(httpd_req_t *req) {
    char buf[10] = {0};
    if (httpd_req_get_url_query_len(req) > 0 && httpd_req_get_url_query_len(req) < sizeof(buf)) {
        httpd_req_get_url_query_str(req, buf, sizeof(buf));
        const char* p = strstr(buf, "c=");
        if (p && p[2] != 0) {
            parse_and_store(reinterpret_cast<const uint8_t*>(&p[2]), 1);
            httpd_resp_set_type(req, "application/json");
            set_http_common_headers(req);
            httpd_resp_sendstr(req, "{\"ok\":true}");
            return ESP_OK;
        }
    }
    httpd_resp_set_type(req, "application/json");
    set_http_common_headers(req);
    httpd_resp_set_status(req, "400 Bad Request");
    httpd_resp_sendstr(req, "{\"ok\":false,\"error\":\"missing_or_invalid_c\"}");
    return ESP_OK;
}

static esp_err_t http_cmd_options_handler(httpd_req_t *req) {
    set_http_common_headers(req);
    httpd_resp_set_status(req, "204 No Content");
    httpd_resp_send(req, nullptr, 0);
    return ESP_OK;
}

// Minimal HTTP handler for /status
static esp_err_t http_status_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "application/json");
    set_http_common_headers(req);
    httpd_resp_sendstr(req, "{\"ok\":true,\"service\":\"robocar_ctrl\"}");
    return ESP_OK;
}

static esp_err_t http_status_options_handler(httpd_req_t *req) {
    set_http_common_headers(req);
    httpd_resp_set_status(req, "204 No Content");
    httpd_resp_send(req, nullptr, 0);
    return ESP_OK;
}

static esp_err_t http_root_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    set_http_common_headers(req);
    httpd_resp_sendstr(req, INDEX_HTML);
    return ESP_OK;
}

void init_wifi_receiver() {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ret = esp_netif_init();
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        ESP_ERROR_CHECK(ret);
    }

    ret = esp_event_loop_create_default();
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        ESP_ERROR_CHECK(ret);
    }

    init_wifi_softap();
    
    // Start minimal HTTP server on dedicated control API port.
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = CONTROL_HTTP_PORT;
    httpd_handle_t server = NULL;
    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_uri_t root_uri = {.uri = "/", .method = HTTP_GET, .handler = http_root_handler, .user_ctx = nullptr};
        httpd_uri_t cmd_uri = {.uri = "/cmd", .method = HTTP_GET, .handler = http_cmd_handler, .user_ctx = nullptr};
        httpd_uri_t status_uri = {.uri = "/status", .method = HTTP_GET, .handler = http_status_handler, .user_ctx = nullptr};
        httpd_uri_t cmd_options_uri = {.uri = "/cmd", .method = HTTP_OPTIONS, .handler = http_cmd_options_handler, .user_ctx = nullptr};
        httpd_uri_t status_options_uri = {.uri = "/status", .method = HTTP_OPTIONS, .handler = http_status_options_handler, .user_ctx = nullptr};
        httpd_register_uri_handler(server, &root_uri);
        httpd_register_uri_handler(server, &cmd_uri);
        httpd_register_uri_handler(server, &status_uri);
        httpd_register_uri_handler(server, &cmd_options_uri);
        httpd_register_uri_handler(server, &status_options_uri);
        ESP_LOGI(TAG, "HTTP control API listening on port %d", CONTROL_HTTP_PORT);
    } else {
        ESP_LOGE(TAG, "Failed to start HTTP control API on port %d", CONTROL_HTTP_PORT);
    }
    
    xTaskCreate(wifi_control_server_task, "wifi_ctrl_srv", 4096, nullptr, 5, nullptr);
    ESP_LOGI(TAG, "Wi-Fi receiver initialized");
}

bool get_manual_control(float &duty, float &steer, bool &emergency) {
    const bool hasActiveCommand =
        s_forward.load() || s_backward.load() || s_left.load() || s_right.load();

    if (hasActiveCommand) {
        duty = s_duty.load();
        steer = s_steer.load();
        emergency = s_emergency.load();
        return true;
    }

    int last = s_last_tick.load();
    if (last == 0) {
        return false;
    }

    TickType_t now = xTaskGetTickCount();
    if ((now - static_cast<TickType_t>(last)) > pdMS_TO_TICKS(MANUAL_TIMEOUT_MS)) {
        return false;
    }

    duty = s_duty.load();
    steer = s_steer.load();
    emergency = s_emergency.load();
    s_emergency.store(false);
    return true;
}