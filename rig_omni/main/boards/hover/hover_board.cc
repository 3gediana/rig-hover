#include "wifi_board.h"
#include "codecs/no_audio_codec.h"
#include "application.h"
#include "button.h"
#include "config.h"
#include "mcp_server.h"
#include "esp32_camera.h"
// #include "camera_web_server.h"  // TODO: Migrate camera web server if needed
// #include "lulu_ble.h"  // 鏆傛椂灞忚斀锛屽惎鐢?BluFi 閰嶇綉
#include "assets/lang_config.h"
#include "board_config.h"

#include "display/emote_display.h"

#include <wifi_manager.h>
#include <esp_log.h>
#include <cJSON.h>
#include <esp_lcd_panel_vendor.h>
#include <esp_lcd_panel_io.h>
#include <esp_lcd_panel_ops.h>
#include <driver/spi_common.h>
#include <driver/uart.h>
#include <driver/gpio.h>
#include <nvs_flash.h>
#include <esp_heap_caps.h>

#include "esp_lcd_gc9a01.h"
#include "robot.h"
#include "robot_action.h"
#include "imu.h"
#include "hover_debug_server.h"

#define TAG "HOVER"

// Hover 涓撳睘闊虫晥
namespace {
    // 寮€鏈洪煶鏁?(0~3s)
    extern const char engine_startup_ogg_start[] asm("_binary_engine_startup_ogg_start");
    extern const char engine_startup_ogg_end[] asm("_binary_engine_startup_ogg_end");
    static const std::string_view STARTUP_SOUND {
        static_cast<const char*>(engine_startup_ogg_start),
        static_cast<size_t>(engine_startup_ogg_end - engine_startup_ogg_start)
    };
    // 娌归棬闊虫晥 (8s~缁撳熬)
    extern const char engine_throttle_ogg_start[] asm("_binary_engine_throttle_ogg_start");
    extern const char engine_throttle_ogg_end[] asm("_binary_engine_throttle_ogg_end");
    static const std::string_view THROTTLE_SOUND {
        static_cast<const char*>(engine_throttle_ogg_start),
        static_cast<size_t>(engine_throttle_ogg_end - engine_throttle_ogg_start)
    };
}

// 闀挎寜閲嶇疆 NVS 鐨勬椂闂撮槇鍊硷紙姣锛?
static constexpr int kLongPressResetMs = 3000;
static constexpr int kLongPressShowEmotionMs = 1000;  // 闀挎寜1绉掑悗鏄剧ず琛ㄦ儏
static constexpr int kDoubleTapWindowMs = 400;  // 鍙屽嚮妫€娴嬬獥鍙ｏ紙姣锛?

class HoverBoard : public WifiBoard {
private:
    Button boot_button_;
    Button touch_button_;
    emote::EmoteDisplay* display_ = nullptr;  // AAF鍔ㄧ敾鏄剧ず
    Esp32Camera* camera_ = nullptr;  // 鍒濆鍖栦负nullptr
    TaskHandle_t robot_task_handle_ = nullptr;
    TaskHandle_t motor_rx_task_handle_ = nullptr;
    TaskHandle_t imu_task_handle_ = nullptr;
    int64_t button_press_start_time_ = 0;  // 鎸夐敭鎸変笅鏃堕棿鎴?
    esp_timer_handle_t long_press_timer_ = nullptr;  // 闀挎寜妫€娴嬪畾鏃跺櫒
    bool nvs_reset_emotion_shown_ = false;  // 鏄惁宸叉樉绀?nvs_reset 琛ㄦ儏

    // 瑙︽懜鍙屽嚮妫€娴?
    int touch_tap_count_ = 0;                    // 瑙︽懜鐐瑰嚮璁℃暟
    esp_timer_handle_t tap_timer_ = nullptr;    // 鍙屽嚮妫€娴嬪畾鏃跺櫒

    void InitializeUart() {
        uart_config_t uart_cfg = {
            .baud_rate = 1000000,
            .data_bits = UART_DATA_8_BITS,
            .parity = UART_PARITY_DISABLE,
            .stop_bits = UART_STOP_BITS_1,
            .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        };
        ESP_LOGI(TAG, "Initialize UART");
        uart_driver_install(UART_NUM_2, 1024, 1024, 0, NULL, 0);
        uart_param_config(UART_NUM_2, &uart_cfg);
        uart_set_pin(UART_NUM_2, MOTOR_UART_TX_PIN, MOTOR_UART_RX_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
        WriteByte_P_V(3, 1500, 1300);
    }

    void InitializeBootButton() {
        esp_rom_gpio_pad_select_gpio(GPIO_NUM_0);
        gpio_reset_pin(GPIO_NUM_0);
        gpio_config_t io_conf = {
            .pin_bit_mask = (1ULL << GPIO_NUM_0),
            .mode = GPIO_MODE_INPUT,
            .pull_up_en = GPIO_PULLUP_ENABLE,
            .pull_down_en = GPIO_PULLDOWN_DISABLE,
            .intr_type = GPIO_INTR_DISABLE,
        };
        ESP_ERROR_CHECK(gpio_config(&io_conf));
    }

    void InitializeSpi() {
        spi_bus_config_t buscfg = {};
        buscfg.mosi_io_num = DISPLAY_MOSI_PIN;
        buscfg.miso_io_num = GPIO_NUM_NC;
        buscfg.sclk_io_num = DISPLAY_CLK_PIN;
        buscfg.quadwp_io_num = GPIO_NUM_NC;
        buscfg.quadhd_io_num = GPIO_NUM_NC;
        buscfg.max_transfer_sz = DISPLAY_WIDTH * DISPLAY_HEIGHT * sizeof(uint16_t);
        ESP_ERROR_CHECK(spi_bus_initialize(SPI3_HOST, &buscfg, SPI_DMA_CH_AUTO));
    }

    void InitializeLcdDisplay() {
        esp_lcd_panel_io_handle_t panel_io = nullptr;
        esp_lcd_panel_handle_t panel = nullptr;

        ESP_LOGI(TAG, "Install panel IO");
        esp_lcd_panel_io_spi_config_t io_config = {};
        io_config.cs_gpio_num = DISPLAY_CS_PIN;
        io_config.dc_gpio_num = DISPLAY_DC_PIN;
        io_config.spi_mode = DISPLAY_SPI_MODE;
        io_config.pclk_hz = 80 * 1000 * 1000;  // 80MHz SPI
        io_config.trans_queue_depth = 10;
        io_config.lcd_cmd_bits = 8;
        io_config.lcd_param_bits = 8;
        ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi(SPI3_HOST, &io_config, &panel_io));

        ESP_LOGI(TAG, "Install GC9A01 LCD driver");
        esp_lcd_panel_dev_config_t panel_config = {};
        panel_config.reset_gpio_num = DISPLAY_RST_PIN;
        panel_config.rgb_ele_order = DISPLAY_RGB_ORDER;
        panel_config.bits_per_pixel = 16;
        // 浣跨敤榛樿GC9A01鍒濆鍖栵紝涓嶄娇鐢ㄨ嚜瀹氫箟gc9107鍛戒护
        ESP_ERROR_CHECK(esp_lcd_new_panel_gc9a01(panel_io, &panel_config, &panel));
        esp_lcd_panel_reset(panel);
        esp_lcd_panel_init(panel);
        esp_lcd_panel_invert_color(panel, DISPLAY_INVERT_COLOR);
        esp_lcd_panel_swap_xy(panel, DISPLAY_SWAP_XY);
        esp_lcd_panel_mirror(panel, DISPLAY_MIRROR_X, DISPLAY_MIRROR_Y);
        esp_lcd_panel_disp_on_off(panel, true);  // 鎵撳紑鏄剧ず

        // 浣跨敤 EmoteDisplay 鎾斁 AAF 鍔ㄧ敾
        display_ = new emote::EmoteDisplay(panel, panel_io, DISPLAY_WIDTH, DISPLAY_HEIGHT);
        ESP_LOGI(TAG, "Using EmoteDisplay for AAF animations");
    }

    void InitializeCamera() {
        camera_config_t camera_config = {
            .pin_pwdn = CAMERA_PIN_PWDN,
            .pin_reset = CAMERA_PIN_RESET,
            .pin_xclk = CAMERA_PIN_XCLK,
            .pin_sccb_sda = CAMERA_PIN_SIOD,
            .pin_sccb_scl = CAMERA_PIN_SIOC,
            .pin_d7 = CAMERA_PIN_D7,
            .pin_d6 = CAMERA_PIN_D6,
            .pin_d5 = CAMERA_PIN_D5,
            .pin_d4 = CAMERA_PIN_D4,
            .pin_d3 = CAMERA_PIN_D3,
            .pin_d2 = CAMERA_PIN_D2,
            .pin_d1 = CAMERA_PIN_D1,
            .pin_d0 = CAMERA_PIN_D0,
            .pin_vsync = CAMERA_PIN_VSYNC,
            .pin_href = CAMERA_PIN_HREF,
            .pin_pclk = CAMERA_PIN_PCLK,
            .xclk_freq_hz = XCLK_FREQ_HZ,
            .ledc_timer = LEDC_TIMER_0,
            .ledc_channel = LEDC_CHANNEL_0,
            // Native grayscale: GC0308 has a Y-only output mode (OUTPUT_FMT=0xb1),
            // so the sensor hands us 1 byte/pixel and the DMA moves half the
            // bytes. Doing the luma conversion in software instead measured
            // 11 ms per frame, which is a quarter of the /capture budget.
            // Consequence: the MCP camera preview path (Capture()) shows no
            // colour image; the perception wire format is unaffected because it
            // only ever used the luma plane.
            .pixel_format = PIXFORMAT_GRAYSCALE,
            .frame_size = FRAMESIZE_240X240,
            .jpeg_quality = 12,
            .fb_count = 2,
            .fb_location = CAMERA_FB_IN_PSRAM,
            .grab_mode = CAMERA_GRAB_LATEST,  // 濮嬬粓鑾峰彇鏈€鏂板抚
            .sccb_i2c_port = 1,  // 鎽勫儚澶寸敤GPIO 4/5锛孖MU鐢℅PIO 48/14锛屽繀椤荤敤涓嶅悓绔彛
        };

        // 鍏堟鏌ユ憚鍍忓ご鏄惁鍙敤
        sensor_t* sensor = esp_camera_sensor_get();
        if (sensor != nullptr) {
            // 浼犳劅鍣ㄥ凡瀛樺湪锛屼笉闇€瑕侀噸澶嶅垵濮嬪寲
            ESP_LOGW(TAG, "Camera sensor already initialized");
            camera_ = nullptr;
            return;
        }

        camera_ = new Esp32Camera(camera_config);
        
        // 妫€鏌ユ憚鍍忓ご浼犳劅鍣ㄦ槸鍚︽垚鍔熷垵濮嬪寲
        sensor = esp_camera_sensor_get();
        if (sensor != nullptr) {
            ESP_LOGI(TAG, "Camera initialized successfully, sensor PID: 0x%x", sensor->id.PID);
        } else {
            ESP_LOGW(TAG, "Camera initialization failed, camera tools will not be available");
            // 娉ㄦ剰: 杩欓噷涓嶅垹闄?camera_ 浠ラ伩鍏嶅鎬佺被鍨嬫瀽鏋勮鍛?
            // 鍐呭瓨娉勬紡閲忓緢灏忥紙瀵硅薄鏈韩寰堝皬锛夛紝涓斿彧浼氬彂鐢熶竴娆?
            camera_ = nullptr;
        }
    }



    void InitializeButtons() {
        // 鍒涘缓鍙屽嚮妫€娴嬪畾鏃跺櫒
        esp_timer_create_args_t tap_timer_args = {
            .callback = [](void* arg) {
                auto board = static_cast<HoverBoard*>(arg);
                if (board->touch_tap_count_ == 1) {
                    // 鍗曡Е锛氶殢鏈烘憞澶?+ 寮曟搸鍚姩澹?
                    int angle = (esp_random() % 150) - 75;  // -75 ~ 75 搴?
                    target_head_pos = angle;
                    auto& app = Application::GetInstance();
                    app.PlaySound(STARTUP_SOUND);
                    ESP_LOGI(TAG, "Single tap: random head shake to %d deg + startup sound", angle);
                } else if (board->touch_tap_count_ >= 2) {
                    // 鍙屽嚮锛氭挱鏀炬补闂ㄩ煶鏁?+ 鍘熷湴杞湀
                    auto& app = Application::GetInstance();
                    app.PlaySound(THROTTLE_SOUND);
                    stable_yaw += 360.0f;
                    yaw_ctrl_time = esp_timer_get_time() / 1000.0;
                    ESP_LOGI(TAG, "Double tap: spin 360掳 + throttle sound");
                }
                board->touch_tap_count_ = 0;
            },
            .arg = this,
            .dispatch_method = ESP_TIMER_TASK,
            .name = "tap_timer",
        };
        esp_timer_create(&tap_timer_args, &tap_timer_);

        // 鍒涘缓闀挎寜妫€娴嬪畾鏃跺櫒
        esp_timer_create_args_t timer_args = {
            .callback = [](void* arg) {
                auto board = static_cast<HoverBoard*>(arg);
                // 妫€鏌ユ寜閿槸鍚︿粛琚寜浣?
                if (board->button_press_start_time_ > 0) {
                    ESP_LOGI(TAG, "Long press detected (>1s), showing nvs_reset emotion");
                    if (board->display_) {
                        board->display_->SetEmotion("nvs_reset");
                    }
                    board->nvs_reset_emotion_shown_ = true;
                }
            },
            .arg = this,
            .dispatch_method = ESP_TIMER_TASK,
            .name = "long_press_timer",
        };
        esp_timer_create(&timer_args, &long_press_timer_);

        // 璁板綍鎸夐敭鎸変笅鏃堕棿锛屽苟鍚姩瀹氭椂鍣?
        boot_button_.OnPressDown([this]() {
            button_press_start_time_ = esp_timer_get_time() / 1000;  // 杞崲涓烘绉?
            nvs_reset_emotion_shown_ = false;
            ESP_LOGI(TAG, "Button pressed down");
            // 鍚姩1绉掑畾鏃跺櫒锛屾娴嬮暱鎸?
            esp_timer_start_once(long_press_timer_, kLongPressShowEmotionMs * 1000);
        });

        // 妫€鏌ラ暱鎸夋椂闂?
        boot_button_.OnPressUp([this]() {
            // 鍋滄瀹氭椂鍣?
            esp_timer_stop(long_press_timer_);
            
            if (button_press_start_time_ > 0) {
                int64_t press_duration = (esp_timer_get_time() / 1000) - button_press_start_time_;
                ESP_LOGI(TAG, "Button released, press duration: %d ms", press_duration);
                
                if (press_duration >= kLongPressResetMs) {
                    ESP_LOGW(TAG, "Long press detected (>3s), resetting NVS...");
                    // 鎾斁鎻愮ず闊筹紙濡傛灉鍙敤锛?
                    auto& app = Application::GetInstance();
                    app.PlaySound(THROTTLE_SOUND);
                    vTaskDelay(pdMS_TO_TICKS(500));
                    
                    // 娓呴櫎 NVS
                    esp_err_t ret = nvs_flash_erase();
                    if (ret == ESP_OK) {
                        ESP_LOGI(TAG, "NVS erased successfully");
                    } else {
                        ESP_LOGE(TAG, "Failed to erase NVS: %s", esp_err_to_name(ret));
                    }
                    nvs_flash_init();
                    
                    // 閲嶅惎璁惧
                    ESP_LOGI(TAG, "Restarting device in 1 second...");
                    vTaskDelay(pdMS_TO_TICKS(1000));
                    esp_restart();
                } else if (nvs_reset_emotion_shown_) {
                    // 鏈揪鍒伴噸缃椂闂达紝浣嗗凡鏄剧ず浜嗚〃鎯咃紝鎭㈠姝ｅ父琛ㄦ儏
                    ESP_LOGI(TAG, "Long press cancelled, restoring emotion");
                    if (display_) {
                        display_->SetEmotion("neutral");
                    }
                }
                button_press_start_time_ = 0;
                nvs_reset_emotion_shown_ = false;
            }
        });

        // 淇濇寔鍘熸湁鐨勫崟鍑诲姛鑳?
        boot_button_.OnClick([this]() {
            auto& app = Application::GetInstance();
            if (app.GetDeviceState() == kDeviceStateStarting && !WifiManager::GetInstance().IsConnected()) {
                EnterWifiConfigMode();
            }
            app.ToggleChatState();
        });

        // Touch 浼犳劅鍣?鈥斺€?鍗曡Е闅忔満鎽囧ご锛屽弻鍑昏浆鍦?+ 闊虫晥
        touch_button_.OnPressDown([this]() {
            button_press_start_time_ = esp_timer_get_time() / 1000;
            nvs_reset_emotion_shown_ = false;
            ESP_LOGI(TAG, "Touch pressed down");
            esp_timer_start_once(long_press_timer_, kLongPressShowEmotionMs * 1000);
        });

        touch_button_.OnPressUp([this]() {
            esp_timer_stop(long_press_timer_);
            if (button_press_start_time_ > 0) {
                int64_t press_duration = (esp_timer_get_time() / 1000) - button_press_start_time_;
                ESP_LOGI(TAG, "Touch released, press duration: %d ms", press_duration);
                if (press_duration >= kLongPressResetMs) {
                    // 闀挎寜 >3s锛氶噸缃?NVS
                    ESP_LOGW(TAG, "Touch long press detected (>3s), resetting NVS...");
                    auto& app = Application::GetInstance();
                    app.PlaySound(THROTTLE_SOUND);
                    vTaskDelay(pdMS_TO_TICKS(500));
                    esp_err_t ret = nvs_flash_erase();
                    if (ret == ESP_OK) {
                        ESP_LOGI(TAG, "NVS erased successfully");
                    } else {
                        ESP_LOGE(TAG, "Failed to erase NVS: %s", esp_err_to_name(ret));
                    }
                    nvs_flash_init();
                    ESP_LOGI(TAG, "Restarting device in 1 second...");
                    vTaskDelay(pdMS_TO_TICKS(1000));
                    esp_restart();
                } else if (nvs_reset_emotion_shown_) {
                    // 闀挎寜 >1s 浣?<3s锛氬彇娑堬紝鎭㈠琛ㄦ儏
                    ESP_LOGI(TAG, "Touch long press cancelled, restoring emotion");
                    if (display_) {
                        display_->SetEmotion("neutral");
                    }
                } else {
                    // 鐭寜锛氭娴嬪崟/鍙屽嚮
                    touch_tap_count_++;
                    if (touch_tap_count_ == 1) {
                        // 鍚姩鍙屽嚮妫€娴嬪畾鏃跺櫒
                        esp_timer_start_once(tap_timer_, kDoubleTapWindowMs * 1000);
                    }
                }
                button_press_start_time_ = 0;
                nvs_reset_emotion_shown_ = false;
            }
        });
    }

    void SetDogSpeed(int dog_vx, int dog_vyaw, int time) {
        ESP_LOGI(TAG, "SetDogSpeed: vx=%d, vyaw=%d, time=%d", dog_vx, dog_vyaw, time);

        vx = int(2.2 * dog_vx);
        vyaw = int(2.8 * dog_vyaw);
        if (time > 0) {
            vTaskDelay(pdMS_TO_TICKS(time));
        }
        vx = 0.0;
        vyaw = 0.0;
        ESP_LOGI(TAG, "SetDogSpeed: done");
    }


    void InitializeTools() {
        auto& mcp_server = McpServer::GetInstance();
        mcp_server.AddTool("self.robot.head_angle",
            "璁剧疆鏈哄櫒澶撮儴瑙掑害,瑙掑害姝ｈ礋45,0涓轰腑闂?姝ｄ负宸︽壄澶?璐熶负鍙虫壄澶",
            PropertyList({
                Property("angle", kPropertyTypeInteger, -75, 75),
            }), [this](const PropertyList& properties) -> ReturnValue {
                int angle = properties["angle"].value<int>();
                ESP_LOGI(TAG, "SetHeadAngle called with angle=%d", angle);
                target_head_pos = angle;
                return true;
            });

        mcp_server.AddTool("self.robot.move",
            "璁剧疆鍓嶈繘鍚庨€€璺濈,鍗曚綅鍘樼背,姝ｄ负鍓嶈繘,璐熶负鍚庨€€,涓嶈鍏蜂綋鍊奸粯璁?0鍘樼背",
            PropertyList({
                Property("distance", kPropertyTypeInteger, -20, 20),
            }), [this](const PropertyList& properties) -> ReturnValue {
                int distance = properties["distance"].value<int>();
                ESP_LOGI(TAG, "SetHeadAngle called with move distance=%d", distance);
                stable_pos = stable_pos + distance/100.0;
                return true;
            });

        mcp_server.AddTool("self.robot.rotate",
            "璁剧疆杩愬姩鏃嬭浆瑙掑害,鏈鸿韩鏃嬭浆娉ㄦ剰鍜屽ご閮ㄦ棆杞尯鍒?鍗曚綅搴?姝ｄ负宸?璐熶负鍙?涓嶈鍏蜂綋鍊奸粯璁?0搴",
            PropertyList({
                Property("angle", kPropertyTypeInteger, -180, 180),
            }), [this](const PropertyList& properties) -> ReturnValue {
                int angle = properties["angle"].value<int>();
                ESP_LOGI(TAG, "SetHeadAngle called with rotate angle=%d", angle);
                stable_yaw = stable_yaw + angle;
                yaw_ctrl_time = esp_timer_get_time()/1000.0;  // 涓嬭揪鍛戒护鎵峺aw闂幆杩愬姩锛屽钩鏃秠aw涓嶉棴鐜?
                return true;
            });

        mcp_server.AddTool("self.robot.set_mode",
            "鍒囨崲鍒板皬杞︽ā寮忔垨浜戝彴妯″紡銆傜敤鎴疯銆屽彉鎴愪簯鍙般€嶃€屼簯鍙版ā寮忋€嶃€屾斁鍒板彴搴с€嶆椂 mode=gimbal锛"
            "鐢ㄦ埛璇淬€屽彉鎴愬皬杞︺€嶃€屽皬杞︽ā寮忋€嶃€屾仮澶嶅钩琛°€嶆椂 mode=car銆",
            PropertyList({
                Property("mode", kPropertyTypeString)
            }), [](const PropertyList& properties) -> ReturnValue {
                auto mode = properties["mode"].value<std::string>();
                if (mode == "gimbal" || mode == "yun_tai" || mode == "浜戝彴") {
                    SetControlMode(1);
                    return std::string("宸插垏鎹㈠埌浜戝彴妯″紡");
                }
                if (mode == "car" || mode == "xiao_che" || mode == "灏忚溅") {
                    SetControlMode(0);
                    return std::string("宸插垏鎹㈠埌灏忚溅妯″紡");
                }
                return std::string("鏈煡妯″紡锛岃浣跨敤 car 鎴?gimbal");
            });

        mcp_server.AddTool("self.robot.pitch",
            "浜戝彴妯″紡涓嬭皟鏁存満韬刊浠帮紝浠呬簯鍙版ā寮忕敓鏁堬紱灏忚溅妯″紡璋冪敤鍒欑洿鎺ヨ繑鍥炰笉鍔ㄤ綔銆"
            "鐢ㄦ埛璇存姮澶淬€佸ご鎶珮锛歳elative=1锛宎ngle 涓烘锛堜笉璇村害鏁伴粯璁?15锛夛紱"
            "璇翠綆澶淬€佷綆涓嬶細relative=1锛宎ngle 涓鸿礋锛堜笉璇村害鏁伴粯璁?-15锛夛紱"
            "璇磋瀹氬埌鏌愯搴︺€佷刊浠板埌鏌愬害锛歳elative=0锛宎ngle 涓虹粷瀵硅搴︺€"
            "0 涓烘按骞筹紝姝ｄ负鎶ご锛堝唴閮ㄤ細鍙栧弽鍐欏叆鐩爣锛夛紝璐熶负浣庡ご锛岃寖鍥?卤45銆",
            PropertyList({
                Property("angle", kPropertyTypeInteger, 15, -45, 45),
                Property("relative", kPropertyTypeInteger, 0, 0, 1),
            }), [](const PropertyList& properties) -> ReturnValue {
                if (control_mode != 1) {
                    return std::string("褰撳墠鏄皬杞︽ā寮忥紝淇话璋冩暣浠呬簯鍙版ā寮忔湁鏁");
                }
                int angle = properties["angle"].value<int>();
                bool relative = properties["relative"].value<int>() != 0;
                SetGimbalPitch((float)angle, relative);
                return std::string("浜戝彴淇话宸茶涓?") + std::to_string((int)gimbal_pitch_target) + "搴";
            });

        mcp_server.AddTool("self.status.battery",
            "鏌ヨ褰撳墠鐢垫睜鐢甸噺,杩斿洖鍓╀綑鐢甸噺鐧惧垎姣",
            PropertyList(std::vector<Property>{}), [this](const PropertyList& properties) -> ReturnValue {
                int level = 0;
                bool charging = false, discharging = false;
                if (GetBatteryLevel(level, charging, discharging)) {
                    return std::string("褰撳墠鐢垫睜鐢甸噺 ") + std::to_string(level) + "%"
                        + (charging ? ", 姝ｅ湪鍏呯數" : "");
                }
                return std::string("鏆傛湭璇诲彇鍒扮數姹犵數鍘");
            });

    }

public:
    HoverBoard() : boot_button_(BOOT_BUTTON_GPIO), touch_button_(TOUCH_BUTTON_GPIO) {
        InitializeSpi();
        InitializeLcdDisplay();
        InitializeButtons();
        InitializeTools();
        
        InitializeCamera();
        
        InitializeUart();
        InitializeController();

        // 绔嬪嵆璇诲彇涓€娆＄數姹犵數鍘嬶紝閬垮厤绛夊緟 60 绉掓墠鏈夌數閲忔暟鎹?
        ReadServoVoltage(3);

        InitializeBootButton();
        
        // IMU 鍒濆鍖?
        imu_init();

        // 鏈哄櫒浜烘帶鍒朵换鍔?
        xTaskCreatePinnedToCore([](void* arg) {
            (void)arg;
            while (true) {
                robot_control();
            }
            vTaskDelete(NULL);
        }, "robot_task", 4096, this, 12, &robot_task_handle_, 0);

        xTaskCreatePinnedToCore([](void* arg) {
            (void)arg;
            while (true) {
                motor_rx();
                vTaskDelay(pdMS_TO_TICKS(3));
            }
            vTaskDelete(NULL);
        }, "motor_rx_task", 4096, this, 12, &motor_rx_task_handle_, 1);
        ESP_LOGI(TAG, "Robot control tasks created");

        xTaskCreatePinnedToCore([](void* arg) {
            (void)arg;
            while (true) {
                imu_read_once();
                vTaskDelay(pdMS_TO_TICKS(2));
            }
            vTaskDelete(NULL);
        }, "imu_task", 4096, this, 12, &imu_task_handle_, 1);
        ESP_LOGI(TAG, "IMU control tasks created");

    }

    virtual AudioCodec* GetAudioCodec() override {
#ifdef AUDIO_I2S_METHOD_SIMPLEX
        static NoAudioCodecSimplex audio_codec(AUDIO_INPUT_SAMPLE_RATE, AUDIO_OUTPUT_SAMPLE_RATE,
            AUDIO_I2S_SPK_GPIO_BCLK, AUDIO_I2S_SPK_GPIO_LRCK, AUDIO_I2S_SPK_GPIO_DOUT,
            AUDIO_I2S_MIC_GPIO_SCK, AUDIO_I2S_MIC_GPIO_WS, AUDIO_I2S_MIC_GPIO_DIN);
#else
        static NoAudioCodecDuplex audio_codec(AUDIO_INPUT_SAMPLE_RATE, AUDIO_OUTPUT_SAMPLE_RATE,
            AUDIO_I2S_GPIO_BCLK, AUDIO_I2S_GPIO_WS, AUDIO_I2S_GPIO_DOUT, AUDIO_I2S_GPIO_DIN);
#endif
        return &audio_codec;
    }

    virtual Display* GetDisplay() override {
        return display_;
    }

    virtual Backlight* GetBacklight() override {
        if (DISPLAY_BACKLIGHT_PIN != GPIO_NUM_NC) {
            static PwmBacklight backlight(DISPLAY_BACKLIGHT_PIN, DISPLAY_BACKLIGHT_OUTPUT_INVERT);
            return &backlight;
        }
        return nullptr;
    }

    virtual Camera* GetCamera() override {
        return camera_;
    }

    virtual std::string GetDeviceStatusJson() override {
        std::string base_json = WifiBoard::GetDeviceStatusJson();
        cJSON* root = cJSON_Parse(base_json.c_str());
        if (!root) {
            return base_json;
        }

        imu_read_once();

        cJSON* imu = cJSON_CreateObject();
        cJSON_AddBoolToObject(imu, "initialized", imu_is_initialized());
        cJSON_AddNumberToObject(imu, "roll", roll);
        cJSON_AddNumberToObject(imu, "pitch", pitch);
        cJSON_AddNumberToObject(imu, "yaw", yaw);
        cJSON_AddItemToObject(root, "imu", imu);

        char* json_str = cJSON_PrintUnformatted(root);
        std::string result(json_str);
        cJSON_free(json_str);
        cJSON_Delete(root);

        return result;
    }

    virtual void OnStartup() override {

        display_->SetEmotion("launch");
        Application::GetInstance().PlaySound(STARTUP_SOUND);
        ESP_LOGI(TAG, "Boot animation: launch + engine startup sound");
    }

    virtual void OnInitializationComplete() override {
        // gpio_set_level(LASER_GPIO, 0);  // 鍒濆鍖栧畬鎴愶紝鍏抽棴婵€鍏?
        ESP_LOGI(TAG, "Initialization complete, laser off");

        // 鎾斁寮€鏈洪煶鏁?
        auto& app = Application::GetInstance();
        app.PlaySound(STARTUP_SOUND);
    }

    virtual void OnWifiConfigStart() override {
        ESP_LOGI(TAG, "WiFi config start, reset to stand then keep sit");
    }

    virtual void OnWifiConfigEnd() override {
        // 閰嶇綉缁撴潫锛屼粠鍧愬Э璧疯韩骞舵挱鏀炬垚鍔熻闊?

        Application::GetInstance().PlaySound(Lang::Sounds::OGG_WIFI_SUCCESS());
        ESP_LOGI(TAG, "WiFi config end, sit reset (stand up) and play success audio");
    }

    virtual void OnWifiConnected() override {
        std::string ip = WifiManager::GetInstance().GetIpAddress();
        // sdkconfig 榛樿鏃ュ織绾у埆涓?WARN锛屽繀椤荤敤 LOGW 鎵嶈兘鍦ㄤ覆鍙ｇ湅鍒?
        ESP_LOGW(TAG, "WiFi IP: %s", ip.c_str());

        if (!hover_debug_server_is_running()) {
            // 杩欎釜鍥炶皟璺戝湪 WiFi 浜嬩欢浠诲姟 sys_evt 閲岋紝鏍堝彧鏈?
            // CONFIG_ESP_SYSTEM_EVENT_TASK_STACK_SIZE = 4096 B銆傚惎鍔ㄨ皟璇曢潰鏉胯鎶?
            // 妯″瀷鎼繘 RAM銆佸缓涓変釜浠诲姟銆佸啀璧?HTTP 鏈嶅姟锛屽疄娴嬪湪杩欓噷鏍堟孩鍑猴紱婧㈠嚭
            // 韪╁潖鐩搁偦浠诲姟鐨?TCB 涔嬪悗宕╁湪璋冨害鍣ㄩ噷锛坴TaskSwitchContext /
            // xTaskIncrementTick锛夛紝琛ㄧ幇涓哄紑鏈哄弽澶嶉噸鍚€傛墍浠ョ粰瀹冭嚜宸辩殑鏍堬紝
            // 浜嬩欢浠诲姟鍙礋璐ｆ妸杩欎竴涓嬫姏鍑哄幓銆?
            //
            // 鍏抽敭锛?*缁濅笉鍦ㄨ繖閲屽唴鑱斿厹搴?*銆傚唴鑱旇皟鐢ㄦ鏄穿婧冭矾寰勬湰韬€斺€旀棭鍏堢増鏈?
            // 鍦ㄥ缓浠诲姟澶辫触鏃朵細鍥為€€鍒板唴鑱旓紝缁撴灉灏辨槸寮€鏈洪噸鍚惊鐜€傞檷绾ц矾寰勫繀椤绘瘮
            // 鍘熻矾寰勬洿瀹夊叏锛氬缓涓嶅嚭浠诲姟灏辨斁寮?HTTP 绔偣骞跺瀹炴墦鏃ュ織锛屾満鍣ㄤ汉浠嶇劧
            // 鍙互骞宠　銆佸彲浠ヨ闊筹紝鍙槸娌℃湁璋冭瘯鎺ュ彛銆?
            ESP_LOGW(TAG, "dbg_boot: internal_free=%u before create",
                     (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
            const BaseType_t ok = xTaskCreatePinnedToCore(
                [](void* arg) {
                    (void)arg;
                    ESP_LOGW(TAG, "dbg_boot: internal_free=%u at entry",
                             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
                    hover_debug_server_start();
                    ESP_LOGW(TAG, "dbg_boot: internal_free=%u after start",
                             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
                    vTaskDelete(NULL);
                }, "dbg_boot", 12288, nullptr, 3, nullptr, 0);
            if (ok != pdPASS) {
                ESP_LOGE(TAG,
                         "dbg_boot task not created (rc=%d) -- skipping HTTP "
                         "endpoints instead of calling start() inline, which "
                         "would overflow the 4 KB sys_evt stack",
                         (int)ok);
            }
        }
        ESP_LOGW(TAG, "Debug panel: http://%s/", ip.c_str());
    }

    // 浠庤埖鏈?ID=3 鐨?PRESENT_VOLTAGE 瀵勫瓨鍣ㄨ鍙栫數姹犵數鍘?
    virtual bool GetBatteryLevel(int &level, bool& charging, bool& discharging) override {
        float voltage = servo_voltage;
        if (voltage < 0.1f) return false;  // 灏氭湭璇诲彇鍒版湁鏁堢數鍘?
        
        // 2S 閿傜數姹? 6.6V(~0%) ~ 8.4V(100%)
        if (voltage >= 8.4f)
            level = 100;
        else if (voltage <= 6.6f)
            level = 0;
        else
            level = (int)((voltage - 6.6f) / (8.4f - 6.6f) * 100.0f);
        
        charging = false;
        discharging = true;
        return true;
    }

    // Hover 浣跨敤寮曟搸澹颁綔涓烘垚鍔熸彁绀洪煶
    virtual std::string_view GetSuccessSound() override {
        return THROTTLE_SOUND;
    }

    virtual std::string GetBoardDescription() override {
        return "涓€涓弻杞钩琛″皬杞﹀舰鎬佹満鍣ㄤ汉锛屾惌杞藉渾褰?240x240 LCD 灞忓箷銆丒SP32-S3 MCU銆?MB PSRAM銆"
               "2璺櫤鑳借埖鏈恒€両MU 濮挎€佷紶鎰熷櫒銆丟C0308 鎽勫儚澶达紝鏀寔鍏夊墤鎺у埗銆"
               "2S 閿傜數姹犱緵鐢?6.6V-8.4V)銆";
    }
};

DECLARE_BOARD(HoverBoard);
