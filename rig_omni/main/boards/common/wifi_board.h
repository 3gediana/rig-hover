#ifndef WIFI_BOARD_H
#define WIFI_BOARD_H

#include "board.h"
#include <freertos/FreeRTOS.h>
#include <freertos/event_groups.h>
#include <esp_timer.h>

class WifiBoard : public Board {
protected:
    esp_timer_handle_t connect_timer_ = nullptr;
    esp_timer_handle_t static_timer_ = nullptr;
    bool in_config_mode_ = false;
    NetworkEventCallback network_event_callback_ = nullptr;

    /**
     * Static-IP fallback, fired once shortly after the station associates.
     *
     * The PC-side ICS hotspot on this dev setup does not reliably answer DHCP:
     * the robot associates (WPA2 handshake completes, ``wifi:connected with`` is
     * logged) but never receives a lease, so it has no IP and no HTTP endpoint is
     * reachable. That used to be patched with an external Python DHCP daemon in
     * %TEMP%, which is exactly the class of failure TECHNICAL-REPORT.md records
     * as hardest to find (an infrastructure timeout nobody was watching).
     *
     * If DHCP has produced no address a few seconds after StartStation(), assign
     * the dev address below and report Connected through the normal path, so
     * everything downstream (power save off, httpd, telemetry) behaves as if the
     * lease had arrived.
     */
    void ApplyStaticIpFallback();

    /**
     * esp_timer trampoline for :meth:`ApplyStaticIpFallback`; ``arg`` is the board.
     */
    static void OnStaticIpFallbackTimer(void* arg);

    virtual std::string GetBoardJson() override;

    /**
     * Handle network event (called from WiFi manager callbacks)
     * @param event The network event type
     * @param data Additional data (e.g., SSID for Connecting/Connected events)
     */
    void OnNetworkEvent(NetworkEvent event, const std::string& data = "");

    /**
     * Start WiFi connection attempt
     */
    void TryWifiConnect();

    /**
     * Enter WiFi configuration mode
     */
    void StartWifiConfigMode();

    /**
     * WiFi connection timeout callback
     */
    static void OnWifiConnectTimeout(void* arg);

public:
    WifiBoard();
    virtual ~WifiBoard();
    
    virtual std::string GetBoardType() override;
    
    /**
     * Start network connection asynchronously
     * This function returns immediately. Network events are notified through the callback set by SetNetworkEventCallback().
     */
    virtual void StartNetwork() override;
    
    virtual NetworkInterface* GetNetwork() override;
    virtual void SetNetworkEventCallback(NetworkEventCallback callback) override;
    virtual const char* GetNetworkStateIcon() override;
    virtual void SetPowerSaveLevel(PowerSaveLevel level) override;
    virtual AudioCodec* GetAudioCodec() override { return nullptr; }
    virtual std::string GetDeviceStatusJson() override;
    
    /**
     * Enter WiFi configuration mode (thread-safe, can be called from any task)
     */
    void EnterWifiConfigMode();
    
    /**
     * Check if in WiFi config mode
     */
    bool IsInWifiConfigMode() const;

    /**
     * Notify WiFi connected from external source (e.g., BluFi)
     * This triggers the network connected event to Application
     */
    void NotifyWifiConnected(const std::string& ssid);
};

#endif // WIFI_BOARD_H
