#include "wifi_board.h"

#include "display.h"
#include "application.h"
#include "system_info.h"
#include "settings.h"
#include "assets/lang_config.h"

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <esp_network.h>
#include <esp_log.h>
#include <esp_app_desc.h>
#include <esp_wifi.h>
#include <esp_netif.h>
#include <utility>

#include <font_awesome.h>
#include <wifi_manager.h>
#include <wifi_station.h>
#include <ssid_manager.h>
#include "afsk_demod.h"
#ifdef CONFIG_USE_ESP_BLUFI_WIFI_PROVISIONING
#include "blufi.h"
#include <esp_bt.h>
#endif

static const char *TAG = "WifiBoard";

// Connection timeout in seconds
static constexpr int CONNECT_TIMEOUT_SEC = 60;
// How long to wait for a DHCP lease after association before falling back to a
// static address. Must be shorter than CONNECT_TIMEOUT_SEC so a working DHCP
// never reaches the fallback, and long enough that a slow lease still wins.
static constexpr int STATIC_IP_FALLBACK_SEC = 12;

// Dev-setup address. The PC side runs a 192.168.137.0/24 ICS hotspot at
// 192.168.137.1, so this is the only address the tooling expects.
static constexpr uint32_t STATIC_IP_ADDR = ESP_IP4TOADDR(192, 168, 137, 50);
static constexpr uint32_t STATIC_IP_GW = ESP_IP4TOADDR(192, 168, 137, 1);
static constexpr uint32_t STATIC_IP_MASK = ESP_IP4TOADDR(255, 255, 255, 0);

WifiBoard::WifiBoard() {
    // Create connection timeout timer
    esp_timer_create_args_t timer_args = {
        .callback = OnWifiConnectTimeout,
        .arg = this,
        .dispatch_method = ESP_TIMER_TASK,
        .name = "wifi_connect_timer",
        .skip_unhandled_events = true
    };
    esp_timer_create(&timer_args, &connect_timer_);

    // Static-IP fallback timer, same pattern as the connect timer above.
    esp_timer_create_args_t static_args = {
        .callback = OnStaticIpFallbackTimer,
        .arg = this,
        .dispatch_method = ESP_TIMER_TASK,
        .name = "wifi_static_fallback",
        .skip_unhandled_events = true
    };
    esp_timer_create(&static_args, &static_timer_);
}

WifiBoard::~WifiBoard() {
    if (connect_timer_) {
        esp_timer_stop(connect_timer_);
        esp_timer_delete(connect_timer_);
    }
    if (static_timer_) {
        esp_timer_stop(static_timer_);
        esp_timer_delete(static_timer_);
    }
}

std::string WifiBoard::GetBoardType() {
    return "wifi";
}

void WifiBoard::StartNetwork() {
    auto& wifi_manager = WifiManager::GetInstance();

    // Initialize WiFi manager
    WifiManagerConfig config;
    config.ssid_prefix = "RIG-Omni";
    config.language = Lang::CodeStr();
#if CONFIG_BOARD_TYPE_HOVER
    // The PC mobile hotspot can start after the robot's first scan. Keep the
    // reconnect scan bounded so a missed boot-time beacon does not back off to
    // the shared component's five-minute maximum.
    config.station_scan_min_interval_seconds = 5;
    config.station_scan_max_interval_seconds = 15;
#endif
    wifi_manager.Initialize(config);

    // Set unified event callback - forward to NetworkEvent with SSID data
    wifi_manager.SetEventCallback([this](WifiEvent event, const std::string& data) {
        switch (event) {
            case WifiEvent::Scanning:
                OnNetworkEvent(NetworkEvent::Scanning);
                break;
            case WifiEvent::Connecting:
#if CONFIG_BOARD_TYPE_HOVER
                // Start Hover's DHCP/static-IP fallback only after the saved AP
                // is visible; otherwise its timer expires during hotspot startup.
                if (static_timer_) {
                    esp_timer_stop(static_timer_);
                    esp_timer_start_once(static_timer_, STATIC_IP_FALLBACK_SEC * 1000000ULL);
                }
#endif
                OnNetworkEvent(NetworkEvent::Connecting, data);
                break;
            case WifiEvent::Connected:
                OnNetworkEvent(NetworkEvent::Connected, data);
                break;
            case WifiEvent::Disconnected:
                OnNetworkEvent(NetworkEvent::Disconnected);
                break;
            case WifiEvent::ConfigModeEnter:
                OnNetworkEvent(NetworkEvent::WifiConfigModeEnter);
                break;
            case WifiEvent::ConfigModeExit:
                OnNetworkEvent(NetworkEvent::WifiConfigModeExit);
                break;
        }
    });

    // Try to connect or enter config mode
    TryWifiConnect();
}

void WifiBoard::TryWifiConnect() {
    auto& ssid_manager = SsidManager::GetInstance();
    bool have_ssid = !ssid_manager.GetSsidList().empty();

    if (have_ssid) {
        // Start connection attempt with timeout. Hover keeps scanning for a
        // late PC hotspot instead of entering config mode after 60 seconds.
        ESP_LOGI(TAG, "Starting WiFi connection attempt");
#if !CONFIG_BOARD_TYPE_HOVER
        esp_timer_start_once(connect_timer_, CONNECT_TIMEOUT_SEC * 1000000ULL);
        if (static_timer_) {
            esp_timer_stop(static_timer_);
            esp_timer_start_once(static_timer_, STATIC_IP_FALLBACK_SEC * 1000000ULL);
        }
#endif
        WifiManager::GetInstance().StartStation();
    } else {
        // No SSID configured, enter config mode
        // Wait for the board version to be shown
        vTaskDelay(pdMS_TO_TICKS(1500));
        StartWifiConfigMode();
    }
}

void WifiBoard::OnStaticIpFallbackTimer(void* arg) {
    auto* board = static_cast<WifiBoard*>(arg);
    if (board != nullptr) {
        board->ApplyStaticIpFallback();
    }
}

void WifiBoard::ApplyStaticIpFallback() {
    esp_netif_t* netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
    if (netif == nullptr) {
        ESP_LOGW(TAG, "static-IP fallback: no STA netif");
        return;
    }

    esp_netif_ip_info_t cur{};
    if (esp_netif_get_ip_info(netif, &cur) == ESP_OK && cur.ip.addr != 0) {
        ESP_LOGI(TAG, "static-IP fallback: DHCP already gave " IPSTR,
                 IP2STR(&cur.ip));
        return;
    }

    // Only worth doing when the station actually associated; otherwise the
    // connect timeout path owns the flow and will retry or enter config mode.
    wifi_ap_record_t ap{};
    if (esp_wifi_sta_get_ap_info(&ap) != ESP_OK) {
        ESP_LOGW(TAG, "static-IP fallback: station not associated, skipping");
        return;
    }

    esp_err_t err = esp_netif_dhcpc_stop(netif);
    if (err != ESP_OK && err != ESP_ERR_ESP_NETIF_DHCP_ALREADY_STOPPED) {
        ESP_LOGW(TAG, "static-IP fallback: dhcpc_stop failed: %s",
                 esp_err_to_name(err));
        return;
    }

    esp_netif_ip_info_t ip{};
    ip.ip.addr = STATIC_IP_ADDR;
    ip.gw.addr = STATIC_IP_GW;
    ip.netmask.addr = STATIC_IP_MASK;
    err = esp_netif_set_ip_info(netif, &ip);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "static-IP fallback: set_ip_info failed: %s",
                 esp_err_to_name(err));
        return;
    }

    esp_netif_dns_info_t dns{};
    dns.ip.type = ESP_IPADDR_TYPE_V4;
    dns.ip.u_addr.ip4.addr = STATIC_IP_GW;
    esp_netif_set_dns_info(netif, ESP_NETIF_DNS_MAIN, &dns);

    ESP_LOGW(TAG, "DHCP gave no address after %d s; static fallback " IPSTR
             " gw " IPSTR, STATIC_IP_FALLBACK_SEC,
             IP2STR(&ip.ip), IP2STR(&ip.gw));

    // Nothing else to do: esp_netif_set_ip_info() posts IP_EVENT_STA_GOT_IP
    // itself, which drives WifiStation's Connected event and therefore the
    // normal OnNetworkEvent path. Calling OnNetworkEvent here as well makes the
    // whole connect handler run twice -- that starts the debug server twice and
    // the double init panics with LoadProhibited.
}

void WifiBoard::OnNetworkEvent(NetworkEvent event, const std::string& data) {
    switch (event) {
        case NetworkEvent::Connected: {
            // Stop timeout timer
            esp_timer_stop(connect_timer_);
            if (static_timer_) {
                esp_timer_stop(static_timer_);
            }
            in_config_mode_ = false;
            std::string ip = WifiManager::GetInstance().GetIpAddress();
            ESP_LOGI(TAG, "Connected to WiFi: %s, IP: %s", data.c_str(), ip.c_str());
            // Local-control build: keep the radio awake. The stock firmware only disables
            // power save after cloud activation, which never happens on a LAN-only setup.
            esp_wifi_set_ps(WIFI_PS_NONE);
            ESP_LOGI(TAG, "WiFi power save disabled (WIFI_PS_NONE)");
            Board::GetInstance().OnWifiConnected();
            break;
        }
        case NetworkEvent::Scanning:
            ESP_LOGI(TAG, "WiFi scanning");
            {
                // 显示扫描网络的表情和状态提示
                auto display = Board::GetInstance().GetDisplay();
                if (display) {
                    display->SetStatus(Lang::Strings::SCANNING_WIFI());
                    display->SetEmotion("scanning");
                }
                // TODO: 准备好语音文件后取消注释
                // Application::GetInstance().PlaySound(Lang::Sounds::OGG_SCANNING_WIFI());
            }
            break;
        case NetworkEvent::Connecting:
            ESP_LOGI(TAG, "WiFi connecting to %s", data.c_str());
            break;
        case NetworkEvent::Disconnected:
            ESP_LOGW(TAG, "WiFi disconnected");
            break;
        case NetworkEvent::WifiConfigModeEnter:
            ESP_LOGI(TAG, "WiFi config mode entered");
            in_config_mode_ = true;
            break;
        case NetworkEvent::WifiConfigModeExit:
            ESP_LOGI(TAG, "WiFi config mode exited");
            in_config_mode_ = false;
            // 调用板级配网结束回调
            Board::GetInstance().OnWifiConfigEnd();
            // Try to connect with the new credentials
            TryWifiConnect();
            break;
        default:
            break;
    }

    // Notify external callback if set
    if (network_event_callback_) {
        network_event_callback_(event, data);
    }
}

void WifiBoard::SetNetworkEventCallback(NetworkEventCallback callback) {
    network_event_callback_ = std::move(callback);
}

void WifiBoard::OnWifiConnectTimeout(void* arg) {
#if CONFIG_BOARD_TYPE_HOVER
    // Hover uses stored credentials for the PC hotspot. Do not stop the station
    // after one timeout: it must keep scanning for a hotspot started later.
    ESP_LOGW(TAG, "WiFi still scanning for saved hotspot; retry remains active");
#else
    auto* board = static_cast<WifiBoard*>(arg);
    ESP_LOGW(TAG, "WiFi connection timeout, entering config mode");
    WifiManager::GetInstance().StopStation();
    board->StartWifiConfigMode();
#endif
}

void WifiBoard::StartWifiConfigMode() {
    in_config_mode_ = true;
    // Transition to wifi configuring state
    Application::GetInstance().SetDeviceState(kDeviceStateWifiConfiguring);
#ifdef CONFIG_USE_HOTSPOT_WIFI_PROVISIONING
    auto& wifi_manager = WifiManager::GetInstance();

    wifi_manager.StartConfigAp();

    // Show config prompt after a short delay
    Application::GetInstance().Schedule([&wifi_manager]() {
        std::string hint = Lang::Strings::CONNECT_TO_HOTSPOT();
        hint += wifi_manager.GetApSsid();
        hint += Lang::Strings::ACCESS_VIA_BROWSER();
        hint += wifi_manager.GetApWebUrl();

#if CONFIG_BOARD_TYPE_TARS
        Application::GetInstance().Alert(Lang::Strings::WIFI_CONFIG_MODE(), hint.c_str(), "wificonfig", Lang::Sounds::OGG_TARS_WIFICONFIG());
#else
        Application::GetInstance().Alert(Lang::Strings::WIFI_CONFIG_MODE(), hint.c_str(), "wificonfig", Lang::Sounds::OGG_WIFICONFIG());
#endif
    });
#elif CONFIG_USE_ESP_BLUFI_WIFI_PROVISIONING
    auto &blufi = Blufi::GetInstance();
    // initialize esp-blufi protocol
    blufi.init();
    // 显示 WiFi 配网表情并播放语音提示
    Application::GetInstance().Schedule([]() {
        auto display = Board::GetInstance().GetDisplay();
        if (display) {
            display->SetEmotion("wificonfig");
        }
#if CONFIG_BOARD_TYPE_TARS
        Application::GetInstance().PlaySound(Lang::Sounds::OGG_TARS_WIFICONFIG());
#else
        Application::GetInstance().PlaySound(Lang::Sounds::OGG_WIFICONFIG());
#endif
        // 调用板级配网开始回调（如让机器狗坐下）
        Board::GetInstance().OnWifiConfigStart();
    });
#endif
#if CONFIG_USE_ACOUSTIC_WIFI_PROVISIONING
    // Start acoustic provisioning task
    auto codec = Board::GetInstance().GetAudioCodec();
    int channel = codec ? codec->input_channels() : 1;
    ESP_LOGI(TAG, "Starting acoustic WiFi provisioning, channels: %d", channel);

    xTaskCreate([](void* arg) {
        auto ch = reinterpret_cast<intptr_t>(arg);
        auto& app = Application::GetInstance();
        auto& wifi = WifiManager::GetInstance();
        auto disp = Board::GetInstance().GetDisplay();
        audio_wifi_config::ReceiveWifiCredentialsFromAudio(&app, &wifi, disp, ch);
        vTaskDelete(NULL);
    }, "acoustic_wifi", 4096, reinterpret_cast<void*>(channel), 2, NULL);
#endif
}

void WifiBoard::EnterWifiConfigMode() {
    ESP_LOGI(TAG, "EnterWifiConfigMode called");
    GetDisplay()->ShowNotification(Lang::Strings::ENTERING_WIFI_CONFIG_MODE());

    auto& app = Application::GetInstance();
    auto state = app.GetDeviceState();

    if (state == kDeviceStateSpeaking || state == kDeviceStateListening || state == kDeviceStateIdle) {
        // Reset protocol (close audio channel, reset protocol)
        Application::GetInstance().ResetProtocol();

        xTaskCreate([](void* arg) {
            auto* board = static_cast<WifiBoard*>(arg);

            // Wait for 1 second to allow speaking to finish gracefully
            vTaskDelay(pdMS_TO_TICKS(1000));

            // Stop any ongoing connection attempt
            esp_timer_stop(board->connect_timer_);
            WifiManager::GetInstance().StopStation();

            // Enter config mode
            board->StartWifiConfigMode();

            vTaskDelete(NULL);
        }, "wifi_cfg_delay", 4096, this, 2, NULL);
        return;
    }

    if (state != kDeviceStateStarting) {
        ESP_LOGE(TAG, "EnterWifiConfigMode called but device state is not starting or speaking, device state: %d", state);
        return;
    }

    // Stop any ongoing connection attempt
    esp_timer_stop(connect_timer_);
    WifiManager::GetInstance().StopStation();

    StartWifiConfigMode();
}

bool WifiBoard::IsInWifiConfigMode() const {
    return WifiManager::GetInstance().IsConfigMode();
}

void WifiBoard::NotifyWifiConnected(const std::string& ssid) {
    ESP_LOGI(TAG, "NotifyWifiConnected: %s", ssid.c_str());
    OnNetworkEvent(NetworkEvent::Connected, ssid);
}

NetworkInterface* WifiBoard::GetNetwork() {
    static EspNetwork network;
    return &network;
}

const char* WifiBoard::GetNetworkStateIcon() {
    auto& wifi = WifiManager::GetInstance();

    if (wifi.IsConfigMode()) {
        return FONT_AWESOME_WIFI;
    }
    if (!wifi.IsConnected()) {
        return FONT_AWESOME_WIFI_SLASH;
    }

    int rssi = wifi.GetRssi();
    if (rssi >= -65) {
        return FONT_AWESOME_WIFI;
    } else if (rssi >= -75) {
        return FONT_AWESOME_WIFI_FAIR;
    }
    return FONT_AWESOME_WIFI_WEAK;
}

std::string WifiBoard::GetBoardJson() {
    auto& wifi = WifiManager::GetInstance();
    std::string json = R"({"type":")" + std::string(BOARD_TYPE) + R"(",)";
    json += R"("name":")" + std::string(BOARD_NAME) + R"(",)";

    if (!wifi.IsConfigMode()) {
        json += R"("ssid":")" + wifi.GetSsid() + R"(",)";
        json += R"("rssi":)" + std::to_string(wifi.GetRssi()) + R"(,)";
        json += R"("channel":)" + std::to_string(wifi.GetChannel()) + R"(,)";
        json += R"("ip":")" + wifi.GetIpAddress() + R"(",)";
    }

    json += R"("mac":")" + SystemInfo::GetMacAddress() + R"("})";
    return json;
}

void WifiBoard::SetPowerSaveLevel(PowerSaveLevel level) {
    WifiPowerSaveLevel wifi_level;
    switch (level) {
        case PowerSaveLevel::LOW_POWER:
            wifi_level = WifiPowerSaveLevel::LOW_POWER;
            break;
        case PowerSaveLevel::BALANCED:
            wifi_level = WifiPowerSaveLevel::BALANCED;
            break;
        case PowerSaveLevel::PERFORMANCE:
        default:
            wifi_level = WifiPowerSaveLevel::PERFORMANCE;
            break;
    }
    WifiManager::GetInstance().SetPowerSaveLevel(wifi_level);
}

std::string WifiBoard::GetDeviceStatusJson() {
    auto& board = Board::GetInstance();
    auto root = cJSON_CreateObject();

    // Audio speaker
    auto audio_speaker = cJSON_CreateObject();
    if (auto codec = board.GetAudioCodec()) {
        cJSON_AddNumberToObject(audio_speaker, "volume", codec->output_volume());
    }
    cJSON_AddItemToObject(root, "audio_speaker", audio_speaker);

    // Screen
    auto screen = cJSON_CreateObject();
    if (auto backlight = board.GetBacklight()) {
        cJSON_AddNumberToObject(screen, "brightness", backlight->brightness());
    }
    if (auto display = board.GetDisplay(); display && display->height() > 64) {
        if (auto theme = display->GetTheme()) {
            cJSON_AddStringToObject(screen, "theme", theme->name().c_str());
        }
    }
    cJSON_AddItemToObject(root, "screen", screen);

    // Battery
    int level = 0;
    bool charging = false, discharging = false;
    if (board.GetBatteryLevel(level, charging, discharging)) {
        auto battery = cJSON_CreateObject();
        cJSON_AddNumberToObject(battery, "level", level);
        cJSON_AddBoolToObject(battery, "charging", charging);
        cJSON_AddItemToObject(root, "battery", battery);
    }

    // Network
    auto& wifi = WifiManager::GetInstance();
    auto network = cJSON_CreateObject();
    cJSON_AddStringToObject(network, "type", "wifi");
    cJSON_AddStringToObject(network, "ssid", wifi.GetSsid().c_str());
    int rssi = wifi.GetRssi();
    const char* signal = rssi >= -60 ? "strong" : (rssi >= -70 ? "medium" : "weak");
    cJSON_AddStringToObject(network, "signal", signal);
    cJSON_AddItemToObject(root, "network", network);

    // Firmware version
    auto app_desc = esp_app_get_description();
    cJSON_AddStringToObject(root, "firmware_version", app_desc->version);

    // Language
    cJSON_AddStringToObject(root, "language", Lang::CodeStr());

    // Board hardware description
    auto desc = board.GetBoardDescription();
    if (!desc.empty()) {
        cJSON_AddStringToObject(root, "board_type", desc.c_str());
    }

    // Interrupt mode (AEC)
    auto& app = Application::GetInstance();
    const char* aec_label = nullptr;
    switch (app.GetAecMode()) {
        case kAecOff:         aec_label = "off";     break;
        case kAecOnDeviceSide: aec_label = "device";  break;
        case kAecOnServerSide: aec_label = "server";  break;
    }
    cJSON_AddStringToObject(root, "interrupt_mode", aec_label);

    // Chip temperature
    float temp = 0.0f;
    if (board.GetTemperature(temp)) {
        auto chip = cJSON_CreateObject();
        cJSON_AddNumberToObject(chip, "temperature", temp);
        cJSON_AddItemToObject(root, "chip", chip);
    }

    auto str = cJSON_PrintUnformatted(root);
    std::string result(str);
    cJSON_free(str);
    cJSON_Delete(root);
    return result;
}
