#include "wifiControlServerSensor.hpp"
#include "index.hpp"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"
#include "nvs_flash.h"
#include <atomic>

static constexpr size_t WEB_LOG_MAX_ENTRIES = 128;
static constexpr size_t WEB_LOG_LINE_MAX = 192;

struct WebLogEntry {
    uint32_t seq;
    char line[WEB_LOG_LINE_MAX];
};

static const char TAG[] = "WifiControlServerSensor";

static WebLogEntry s_webLogEntries[WEB_LOG_MAX_ENTRIES];
static std::atomic<uint32_t> s_webLogSeq;
static std::atomic<uint32_t> s_webLogCount;
static portMUX_TYPE s_webLogMux = portMUX_INITIALIZER_UNLOCKED;
using LogVprintfFn = int (*)(const char *, va_list);
static LogVprintfFn s_prevLogVprintf;
static std::atomic<bool> s_logRedirectInstalled;



static void store_web_log_line(const char *text)
{
    if (text == nullptr || text[0] == '\0') {
        return;
    }

    const uint32_t seq = s_webLogSeq.fetch_add(1) + 1;
    const size_t index = static_cast<size_t>((seq - 1) % WEB_LOG_MAX_ENTRIES);

    portENTER_CRITICAL(&s_webLogMux);
    s_webLogEntries[index].seq = seq;
    std::strncpy(s_webLogEntries[index].line, text, WEB_LOG_LINE_MAX - 1);
    s_webLogEntries[index].line[WEB_LOG_LINE_MAX - 1] = '\0';
    portEXIT_CRITICAL(&s_webLogMux);

    uint32_t count = s_webLogCount.load();
    while (count < WEB_LOG_MAX_ENTRIES &&
           !s_webLogCount.compare_exchange_weak(count, count + 1)) {
    }
}

static int web_log_vprintf(const char *fmt, va_list args)
{
    char line[WEB_LOG_LINE_MAX];
    va_list copy;
    va_copy(copy, args);
    const int rc = std::vsnprintf(line, sizeof(line), fmt, copy);
    va_end(copy);

    if (rc > 0) {
        size_t len = std::strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[len - 1] = '\0';
            --len;
        }
        store_web_log_line(line);
    }

    if (s_prevLogVprintf != nullptr) {
        return s_prevLogVprintf(fmt, args);
    }
    return std::vprintf(fmt, args);
}

static void ensure_log_redirect_installed()
{
    bool expected = false;
    if (s_logRedirectInstalled.compare_exchange_strong(expected, true)) {
        s_prevLogVprintf = esp_log_set_vprintf(web_log_vprintf);
    }
}

static std::string get_logs_since(uint32_t since)
{
    const uint32_t latest = s_webLogSeq.load();
    const uint32_t count = s_webLogCount.load();
    if (latest == 0 || count == 0) {
        return std::string();
    }

    const uint32_t oldest = (latest > count) ? (latest - count + 1) : 1;
    uint32_t start = since + 1;
    if (start < oldest) {
        start = oldest;
    }
    if (start > latest) {
        return std::string();
    }

    std::string out;
    out.reserve((latest - start + 1) * 48);

    for (uint32_t seq = start; seq <= latest; ++seq) {
        const size_t index = static_cast<size_t>((seq - 1) % WEB_LOG_MAX_ENTRIES);
        WebLogEntry entry = {};
        portENTER_CRITICAL(&s_webLogMux);
        entry = s_webLogEntries[index];
        portEXIT_CRITICAL(&s_webLogMux);

        if (entry.seq != seq) {
            continue;
        }

        out += std::to_string(entry.seq);
        out += "|";
        out += entry.line;
        out += "\n";
    }

    return out;
}






void WifiControlServerSensor::setHttpCommonHeaders(httpd_req_t *req)
{
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Methods", "GET, OPTIONS");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Headers", "Content-Type");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Private-Network", "true");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
}

WifiControlServerSensor *WifiControlServerSensor::fromRequest(httpd_req_t *req)
{
    return static_cast<WifiControlServerSensor *>(req->user_ctx);
}

void WifiControlServerSensor::recomputeOutputFromState()
{
    float duty = 0.0f;
    if (forward_.load() && !backward_.load()) {
        duty = DUTY_FORWARD;
    } else if (backward_.load() && !forward_.load()) {
        duty = DUTY_BACKWARD;
    }

    float steer = STEER_CENTER;
    if (left_.load() && !right_.load()) {
        steer = STEER_LEFT;
    } else if (right_.load() && !left_.load()) {
        steer = STEER_RIGHT;
    }

    duty_.store(duty);
    steer_.store(steer);
    lastTick_.store(static_cast<int>(xTaskGetTickCount()));
}

void WifiControlServerSensor::emergencyStop()
{
    forward_.store(false);
    backward_.store(false);
    left_.store(false);
    right_.store(false);
    duty_.store(0.0f);
    steer_.store(STEER_CENTER);
    emergency_.store(true);
    lastTick_.store(static_cast<int>(xTaskGetTickCount()));
    _vescControllerApi.stop();
    _vescControllerApi.deactivate();
}

bool WifiControlServerSensor::applyProtocolChar(char c)
{
    switch (c) {
        case 'F': forward_.store(true);  recomputeOutputFromState(); return true;
        case 'f': forward_.store(false); recomputeOutputFromState(); return true;
        case 'B': backward_.store(true);  recomputeOutputFromState(); return true;
        case 'b': backward_.store(false); recomputeOutputFromState(); return true;
        case 'L': left_.store(true);  recomputeOutputFromState(); return true;
        case 'l': left_.store(false); recomputeOutputFromState(); return true;
        case 'R': right_.store(true);  recomputeOutputFromState(); return true;
        case 'r': right_.store(false); recomputeOutputFromState(); return true;
        case 'S': emergencyStop(); return true;
        case 'A':
            emergency_.store(false);
            _vescControllerApi.activate();
            lastTick_.store(static_cast<int>(xTaskGetTickCount()));
            return true;
        default: return false;
    }
}

void WifiControlServerSensor::parseAndStore(const uint8_t *buf, int len)
{
    if (len <= 0 || buf == nullptr) {
        return;
    }

    for (int i = 0; i < len; ++i) {
        const uint8_t b = buf[i];
        if (b == 0x00 || b == '\n' || b == '\r' || b == ' ' || b == '\t') {
            continue;
        }
        if (!applyProtocolChar(static_cast<char>(b))) {
            ESP_LOGW(TAG, "Unknown protocol char: '%c' (0x%02X)", static_cast<char>(b), b);
        }
    }

    // ESP_LOGI(TAG, "State F=%d B=%d L=%d R=%d -> duty=%.3f steer=%.3f",
    //          forward_.load(), backward_.load(), left_.load(), right_.load(),
    //          duty_.load(), steer_.load());
}

void WifiControlServerSensor::runTcpServerTask()
{
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
                parseAndStore(rx_buf, len);
            }

            emergencyStop();
            shutdown(sock, 0);
            close(sock);
        }

        close(listen_sock);
        vTaskDelay(pdMS_TO_TICKS(200));
    }
}

void WifiControlServerSensor::tcpServerTask(void *arg)
{
    static_cast<WifiControlServerSensor *>(arg)->runTcpServerTask();
}

void WifiControlServerSensor::initWifiSoftAp()
{
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

void WifiControlServerSensor::startHttpServer()
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = CONTROL_HTTP_PORT;
    httpd_handle_t server = nullptr;
    if (httpd_start(&server, &config) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start HTTP control API on port %d", CONTROL_HTTP_PORT);
        return;
    }

    httpServer_ = server;

    httpd_uri_t root_uri = {.uri = "/", .method = HTTP_GET, .handler = httpRootHandler, .user_ctx = this};
    httpd_uri_t cmd_uri = {.uri = "/cmd", .method = HTTP_GET, .handler = httpCmdHandler, .user_ctx = this};
    httpd_uri_t logs_uri = {.uri = "/logs", .method = HTTP_GET, .handler = httpLogsHandler, .user_ctx = this};
    httpd_uri_t status_uri = {.uri = "/status", .method = HTTP_GET, .handler = httpStatusHandler, .user_ctx = this};
    httpd_uri_t cmd_options_uri = {.uri = "/cmd", .method = HTTP_OPTIONS, .handler = httpCmdOptionsHandler, .user_ctx = this};
    httpd_uri_t logs_options_uri = {.uri = "/logs", .method = HTTP_OPTIONS, .handler = httpLogsOptionsHandler, .user_ctx = this};
    httpd_uri_t status_options_uri = {.uri = "/status", .method = HTTP_OPTIONS, .handler = httpStatusOptionsHandler, .user_ctx = this};

    httpd_register_uri_handler(server, &root_uri);
    httpd_register_uri_handler(server, &cmd_uri);
    httpd_register_uri_handler(server, &logs_uri);
    httpd_register_uri_handler(server, &status_uri);
    httpd_register_uri_handler(server, &cmd_options_uri);
    httpd_register_uri_handler(server, &logs_options_uri);
    httpd_register_uri_handler(server, &status_options_uri);
    ESP_LOGI(TAG, "HTTP control API listening on port %d", CONTROL_HTTP_PORT);
}

void WifiControlServerSensor::start(void)
{
    if (active_.load()) {
        return;
    }

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

    ensure_log_redirect_installed();

    initWifiSoftAp();
    startHttpServer();

    BaseType_t created = xTaskCreate(tcpServerTask, "wifi_ctrl_srv", 4096, this, 5, &tcpTaskHandle_);
    if (created != pdTRUE) {
        ESP_LOGE(TAG, "Failed to create wifi control TCP task");
        if (httpServer_ != nullptr) {
            httpd_stop(httpServer_);
            httpServer_ = nullptr;
        }
        tcpTaskHandle_ = nullptr;
        return;
    }

    active_.store(true);
    ESP_LOGI(TAG, "Wi-Fi control service initialized");
}

void WifiControlServerSensor::stop(void)
{
    if (!active_.load()) {
        return;
    }

    if (tcpTaskHandle_ != nullptr) {
        vTaskDelete(tcpTaskHandle_);
        tcpTaskHandle_ = nullptr;
    }

    if (httpServer_ != nullptr) {
        httpd_stop(httpServer_);
        httpServer_ = nullptr;
    }

    active_.store(false);
    ESP_LOGI(TAG, "Wi-Fi control service stopped");
}

bool WifiControlServerSensor::isActivated(void)
{
    return active_.load();
}

bool WifiControlServerSensor::isConnected(void)
{
    if (!active_.load()) {
        return false;
    }

    const bool hasActiveCommand =
        forward_.load() || backward_.load() || left_.load() || right_.load();
    if (hasActiveCommand) {
        return true;
    }

    const int last = lastTick_.load();
    if (last == 0) {
        return false;
    }

    TickType_t now = xTaskGetTickCount();
    return (now - static_cast<TickType_t>(last)) <= pdMS_TO_TICKS(MANUAL_TIMEOUT_MS);
}

driving_mode_t WifiControlServerSensor::getDrivingMode(void)
{
    if (!isConnected()) {
        return DRIVING_MODE_DISABLED;
    }
    return DRIVING_MODE_USER;
}

float WifiControlServerSensor::getSpeed(void)
{
    if (!isConnected()) {
        return 0.0f;
    }
    return duty_.load();
}

float WifiControlServerSensor::getSteering(void)
{
    if (!isConnected()) {
        return STEER_CENTER;
    }
    return steer_.load();
}



esp_err_t WifiControlServerSensor::httpCmdHandler(httpd_req_t *req)
{
    auto *self = fromRequest(req);
    char buf[10] = {0};
    if (httpd_req_get_url_query_len(req) > 0 && httpd_req_get_url_query_len(req) < sizeof(buf)) {
        httpd_req_get_url_query_str(req, buf, sizeof(buf));
        const char *p = strstr(buf, "c=");
        if (p && p[2] != 0 && self != nullptr) {
            self->parseAndStore(reinterpret_cast<const uint8_t *>(&p[2]), 1);
            httpd_resp_set_type(req, "application/json");
            setHttpCommonHeaders(req);
            httpd_resp_sendstr(req, "{\"ok\":true}");
            return ESP_OK;
        }
    }

    httpd_resp_set_type(req, "application/json");
    setHttpCommonHeaders(req);
    httpd_resp_set_status(req, "400 Bad Request");
    httpd_resp_sendstr(req, "{\"ok\":false,\"error\":\"missing_or_invalid_c\"}");
    return ESP_OK;
}

esp_err_t WifiControlServerSensor::httpCmdOptionsHandler(httpd_req_t *req)
{
    setHttpCommonHeaders(req);
    httpd_resp_set_status(req, "204 No Content");
    httpd_resp_send(req, nullptr, 0);
    return ESP_OK;
}

esp_err_t WifiControlServerSensor::httpLogsHandler(httpd_req_t *req)
{
    char query[32] = {0};
    char since_value[16] = {0};
    uint32_t since = 0;

    if (httpd_req_get_url_query_len(req) > 0 &&
        httpd_req_get_url_query_len(req) < static_cast<int>(sizeof(query))) {
        httpd_req_get_url_query_str(req, query, sizeof(query));
        if (httpd_query_key_value(query, "since", since_value, sizeof(since_value)) == ESP_OK) {
            since = static_cast<uint32_t>(std::strtoul(since_value, nullptr, 10));
        }
    }

    const std::string payload = get_logs_since(since);
    httpd_resp_set_type(req, "text/plain; charset=utf-8");
    setHttpCommonHeaders(req);
    httpd_resp_send(req, payload.c_str(), static_cast<ssize_t>(payload.size()));
    return ESP_OK;
}

esp_err_t WifiControlServerSensor::httpLogsOptionsHandler(httpd_req_t *req)
{
    setHttpCommonHeaders(req);
    httpd_resp_set_status(req, "204 No Content");
    httpd_resp_send(req, nullptr, 0);
    return ESP_OK;
}

esp_err_t WifiControlServerSensor::httpStatusHandler(httpd_req_t *req)
{
    auto *self = fromRequest(req);
    char payload[128] = {0};
    const bool serviceActive = self != nullptr && self->active_.load();
    const bool vescActive = self != nullptr && self->_vescControllerApi.isActive();
    const bool emergency = self != nullptr && self->emergency_.load();

    std::snprintf(payload, sizeof(payload),
                  "{\"ok\":true,\"service\":\"robocar_ctrl\",\"serviceActive\":%s,\"vescActive\":%s,\"emergency\":%s}",
                  serviceActive ? "true" : "false",
                  vescActive ? "true" : "false",
                  emergency ? "true" : "false");

    httpd_resp_set_type(req, "application/json");
    setHttpCommonHeaders(req);
    httpd_resp_sendstr(req, payload);
    return ESP_OK;
}

esp_err_t WifiControlServerSensor::httpStatusOptionsHandler(httpd_req_t *req)
{
    setHttpCommonHeaders(req);
    httpd_resp_set_status(req, "204 No Content");
    httpd_resp_send(req, nullptr, 0);
    return ESP_OK;
}

esp_err_t WifiControlServerSensor::httpRootHandler(httpd_req_t *req)
{
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    setHttpCommonHeaders(req);
    httpd_resp_sendstr(req, INDEX_HTML);
    return ESP_OK;
}
