#include "hover_debug_server.h"
#include "robot.h"
#include "imu.h"
#include "board.h"
#include "display.h"
#include "esp32_camera.h"
#include "cfc_model.h"
#include "cfc_task.h"
#include "control_arbiter.h"
#include "gaze_guard.h"
#include "telemetry_stream.h"
#include <esp_http_server.h>
#include <lwip/sockets.h>
#include <esp_heap_caps.h>
#include <esp_log.h>
#include <esp_system.h>
#include <esp_timer.h>
#include <cJSON.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

static const char* TAG = "HoverDebug";

#define DEBUG_VAR_COUNT 20

/* Software JPEG quality for GET /capture. Overridable per request with ?q=NN.
 * High enough that YOLO pose keypoints do not move, low enough that the
 * encoder does not own the frame budget -- the exact point is measured. */
#define CAPTURE_QUALITY_DEFAULT 80

float debug_var[DEBUG_VAR_COUNT] = {0};

static httpd_handle_t server = NULL;

// HTML页面
static const char* INDEX_HTML = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hover Debug</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }
        h1 { color: #00d9ff; text-align: center; }
        .section { background: #16213e; border-radius: 10px; padding: 15px; margin: 15px 0; }
        .section h2 { color: #00d9ff; margin-top: 0; font-size: 18px; }
        .imu-data { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
        .imu-item { background: #0f3460; padding: 15px; border-radius: 8px; text-align: center; }
        .imu-item .label { color: #888; font-size: 12px; }
        .imu-item .value { font-size: 24px; font-weight: bold; color: #00d9ff; }
        .var-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
        .var-item { display: flex; align-items: center; gap: 10px; }
        .var-item label { min-width: 100px; color: #888; }
        .var-item input { flex: 1; padding: 8px; border: 1px solid #0f3460; border-radius: 5px; background: #0f3460; color: #fff; font-size: 16px; }
        .var-item button { padding: 8px 15px; background: #00d9ff; border: none; border-radius: 5px; color: #000; cursor: pointer; font-weight: bold; }
        .var-item button:hover { background: #00b8d4; }
        .var-item.reserved input { opacity: 0.5; }
        .status { text-align: center; padding: 10px; color: #0f0; font-size: 14px; }
        .refresh-btn { display: block; width: 100%; padding: 12px; background: #e94560; border: none; border-radius: 5px; color: #fff; font-size: 16px; cursor: pointer; margin-top: 10px; }
        .refresh-btn:hover { background: #c73e54; }
    </style>
</head>
<body>
    <h1>Hover Debug Panel</h1>
    
    <div class="section">
        <h2>IMU Data</h2>
        <div class="imu-data">
            <div class="imu-item"><div class="label">Roll</div><div class="value" id="roll">--</div></div>
            <div class="imu-item"><div class="label">Pitch</div><div class="value" id="pitch">--</div></div>
            <div class="imu-item"><div class="label">Yaw</div><div class="value" id="yaw">--</div></div>
        </div>
    </div>
    
    <div class="section">
        <h2>Debug Variables</h2>
        <div class="var-grid" id="var-grid"></div>
    </div>
    
    <div class="status" id="status">Ready</div>
    <button class="refresh-btn" onclick="fetchData()">Refresh</button>
    
    <script>
        const VAR_COUNT = 20;
        const VAR_LABELS = [
            'head', 'delta_pos', 'POS_kp', 'POS_kd', 'VEL_kp', 'VEL_ki', 'PIT_kp', 'PIT_kd', 'YAW_kp', 'delta_yaw',
            'LQR_k0', 'LQR_k1', 'LQR_k2', 'LQR_k3',
            'GIM_kp', 'GIM_kd', 'GIM_ff', 'var17', 'var18', 'var19'
        ];
        function createVarInputs() {
            const grid = document.getElementById('var-grid');
            for (let i = 0; i < VAR_COUNT; i++) {
                const div = document.createElement('div');
                div.className = 'var-item' + (i >= 17 ? ' reserved' : '');
                div.innerHTML = `
                    <label>${VAR_LABELS[i]}:</label>
                    <input type="number" step="any" id="var${i}" value="0" ${i >= 17 ? 'disabled' : ''}>
                    <button onclick="setVar(${i})" ${i >= 17 ? 'disabled' : ''}>Set</button>
                `;
                grid.appendChild(div);
            }
        }
        
        function fetchData() {
            fetch('/api/data')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('roll').textContent = data.imu.roll.toFixed(2);
                    document.getElementById('pitch').textContent = data.imu.pitch.toFixed(2);
                    document.getElementById('yaw').textContent = data.imu.yaw.toFixed(2);
                    if (data.vars) {
                        for (let i = 0; i < data.vars.length; i++) {
                            const el = document.getElementById('var' + i);
                            if (el && document.activeElement !== el) el.value = data.vars[i];
                        }
                    }
                    document.getElementById('status').textContent = 'Updated: ' + new Date().toLocaleTimeString();
                })
                .catch(e => {
                    document.getElementById('status').textContent = 'Error: ' + e.message;
                });
        }
        
        function setVar(index) {
            if (index >= 17) return;
            const value = parseFloat(document.getElementById('var' + index).value);
            fetch('/api/set?i=' + index + '&v=' + value)
                .then(r => r.json())
                .then(data => {
                    document.getElementById('status').textContent = VAR_LABELS[index] + ' set to ' + value;
                })
                .catch(e => {
                    document.getElementById('status').textContent = 'Error: ' + e.message;
                });
        }
        
        createVarInputs();
        fetchData();
        setInterval(fetchData, 500);
    </script>
</body>
</html>
)rawliteral";

// Disable Nagle on the connection serving this request.
//
// Measured on the real link: the first byte arrives in ~5 ms but the rest of the
// body lands 50-75 ms later. esp_http_server writes the status line/headers and
// the body as separate small segments, and lwIP only turns TCP_NODELAY on for
// *error* responses (CONFIG_HTTPD_ERR_RESP_NO_DELAY). A sub-MSS segment is then
// held by Nagle until the peer's delayed ACK, which is the whole 50 ms budget
// that dominates the end-to-end loop. There is no global Kconfig switch for it
// in IDF 5.5, so the hot handlers set it per connection.
static void force_nodelay(httpd_req_t *req) {
    int fd = httpd_req_to_sockfd(req);
    if (fd < 0) {
        return;
    }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

// GET / - 返回HTML页面
static esp_err_t index_handler(httpd_req_t *req) {
    force_nodelay(req);
    httpd_resp_set_type(req, "text/html");
    httpd_resp_send(req, INDEX_HTML, strlen(INDEX_HTML));
    return ESP_OK;
}

// GET /api/data - 返回IMU和变量数据
static esp_err_t data_handler(httpd_req_t *req) {
    force_nodelay(req);
    cJSON *root = cJSON_CreateObject();

    cJSON *imu = cJSON_CreateObject();
    cJSON_AddNumberToObject(imu, "roll", roll);
    cJSON_AddNumberToObject(imu, "pitch", pitch);
    cJSON_AddNumberToObject(imu, "yaw", yaw);
    cJSON_AddItemToObject(root, "imu", imu);

    cJSON *vars = cJSON_CreateArray();
    cJSON_AddItemToArray(vars, cJSON_CreateNumber(target_head_pos));
    cJSON_AddItemToArray(vars, cJSON_CreateNumber(0));
    cJSON_AddItemToArray(vars, cJSON_CreateNumber(pid_pos.fpKp));
    cJSON_AddItemToArray(vars, cJSON_CreateNumber(pid_pos.fpKd));
    cJSON_AddItemToArray(vars, cJSON_CreateNumber(pid_vel.fpKp));
    cJSON_AddItemToArray(vars, cJSON_CreateNumber(pid_vel.fpKi));
    cJSON_AddItemToArray(vars, cJSON_CreateNumber(pid_pit.fpKp));
    cJSON_AddItemToArray(vars, cJSON_CreateNumber(kd_pit));
    cJSON_AddItemToArray(vars, cJSON_CreateNumber(imu_zero));
    cJSON_AddItemToArray(vars, cJSON_CreateNumber(0));
    cJSON_AddItemToArray(vars, cJSON_CreateNumber(lqr_k[0]));
    cJSON_AddItemToArray(vars, cJSON_CreateNumber(lqr_k[1]));
    cJSON_AddItemToArray(vars, cJSON_CreateNumber(lqr_k[2]));
    cJSON_AddItemToArray(vars, cJSON_CreateNumber(lqr_k[3]));
    cJSON_AddItemToArray(vars, cJSON_CreateNumber(gimbal_kp));
    cJSON_AddItemToArray(vars, cJSON_CreateNumber(gimbal_kd));
    cJSON_AddItemToArray(vars, cJSON_CreateNumber(gimbal_ff));
    cJSON_AddItemToArray(vars, cJSON_CreateNumber(0));
    cJSON_AddItemToArray(vars, cJSON_CreateNumber(0));
    cJSON_AddItemToArray(vars, cJSON_CreateNumber(0));
    cJSON_AddItemToObject(root, "vars", vars);

    char *json_str = cJSON_PrintUnformatted(root);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, json_str, strlen(json_str));

    cJSON_free(json_str);
    cJSON_Delete(root);
    return ESP_OK;
}

// GET /api/set?i=0&v=1.5 - 设置变量
static esp_err_t set_handler(httpd_req_t *req) {
    force_nodelay(req);
    char buf[100];
    int buf_len = httpd_req_get_url_query_len(req) + 1;
    if (buf_len > 1 && buf_len < sizeof(buf)) {
        httpd_req_get_url_query_str(req, buf, buf_len);

        char param[32];
        int index = -1;
        float value = 0.0f;

        if (httpd_query_key_value(buf, "i", param, sizeof(param)) == ESP_OK) {
            index = atoi(param);
        }
        if (httpd_query_key_value(buf, "v", param, sizeof(param)) == ESP_OK) {
            value = atof(param);
        }

        switch (index) {
            case 0:
                target_head_pos = value;
                break;
            case 1:
                stable_pos = stable_pos + value;
                break;
            case 2:
                pid_pos.fpKp = value;
                break;
            case 3:
                pid_pos.fpKd = value;
                break;
            case 4:
                pid_vel.fpKp = value;
                break;
            case 5:
                pid_vel.fpKi = value;
                break;
            case 6:
                pid_pit.fpKp = value;
                break;
            case 7:
                kd_pit = value;
                break;
            case 8:
                imu_zero = value;
                break;
            case 9:
                stable_yaw = stable_yaw + value;
                break;
            case 10:
                lqr_k[0] = value;
                break;
            case 11:
                lqr_k[1] = value;
                break;
            case 12:
                lqr_k[2] = value;
                break;
            case 13:
                lqr_k[3] = value;
                break;
            case 14:
                gimbal_kp = value;
                break;
            case 15:
                gimbal_kd = value;
                break;
            case 16:
                gimbal_ff = value;
                break;
            default:
                break;
        }
    }

    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, "{\"ok\":true}");
    return ESP_OK;
}

// ---------------------------------------------------------------------------
// Local control API. Added so the robot can be driven from any device on the
// same LAN without the cloud: /status, /cmd, /capture.
// ---------------------------------------------------------------------------

static esp_timer_handle_t cmd_watchdog = NULL;

// Stop the wheels if the controller goes quiet (Wi-Fi drop, app crash, ...).
static void cmd_watchdog_cb(void*) {
    vx = 0.0f;
    vyaw = 0.0f;
}

static void cmd_watchdog_arm(void) {
    if (cmd_watchdog == NULL) {
        esp_timer_create_args_t args = {
            .callback = cmd_watchdog_cb,
            .arg = NULL,
            .dispatch_method = ESP_TIMER_TASK,
            .name = "cmd_wd",
        };
        if (esp_timer_create(&args, &cmd_watchdog) != ESP_OK) {
            return;
        }
    }
    esp_timer_stop(cmd_watchdog);
    esp_timer_start_once(cmd_watchdog, 800 * 1000);
}

// GET /status -> full proprioceptive telemetry for the off-board policy
static esp_err_t status_handler(httpd_req_t *req) {
    force_nodelay(req);
    cJSON *root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "t", (double)(esp_timer_get_time() / 1000));

    cJSON *imu = cJSON_CreateObject();
    cJSON_AddNumberToObject(imu, "roll", roll);
    cJSON_AddNumberToObject(imu, "pitch", pitch);
    cJSON_AddNumberToObject(imu, "yaw", yaw);
    cJSON_AddNumberToObject(imu, "ax", accel_x);
    cJSON_AddNumberToObject(imu, "ay", accel_y);
    cJSON_AddNumberToObject(imu, "az", accel_z);
    cJSON_AddNumberToObject(imu, "pit_gyro", pit_gyro);
    cJSON_AddNumberToObject(imu, "dq", dq);
    /* d(yaw)/dt：航向环 D 项的输入。台架上把它和 /cfg 的 k_dyaw 一起看，就能
     * 在不跑平衡环的情况下确认 D 通道的极性与量级。 */
    cJSON_AddNumberToObject(imu, "yaw_rate", yaw_rate);
    cJSON_AddItemToObject(root, "imu", imu);

    cJSON *wheel = cJSON_CreateObject();
    cJSON_AddNumberToObject(wheel, "v1", wheel1_vel);
    cJSON_AddNumberToObject(wheel, "v2", wheel2_vel);
    cJSON_AddNumberToObject(wheel, "x1", wheel1_x);
    cJSON_AddNumberToObject(wheel, "x2", wheel2_x);
    cJSON_AddItemToObject(root, "wheel", wheel);

    cJSON *head = cJSON_CreateObject();
    cJSON_AddNumberToObject(head, "q", q_head);
    cJSON_AddNumberToObject(head, "target", target_head_pos);
    cJSON_AddNumberToObject(head, "gimbal_target", gimbal_pitch_target);
    cJSON_AddItemToObject(root, "head", head);

    cJSON *cmd = cJSON_CreateObject();
    cJSON_AddNumberToObject(cmd, "vx", vx);
    cJSON_AddNumberToObject(cmd, "vyaw", vyaw);
    cJSON_AddNumberToObject(cmd, "mode", control_mode);
    cJSON_AddNumberToObject(cmd, "pos", stable_pos);
    cJSON_AddNumberToObject(cmd, "yaw_target", stable_yaw);
    cJSON_AddItemToObject(root, "cmd", cmd);

    cJSON *torque = cJSON_CreateObject();
    cJSON_AddNumberToObject(torque, "t1", tor1);
    cJSON_AddNumberToObject(torque, "t2", tor2);
    cJSON_AddItemToObject(root, "torque", torque);

    cJSON *pid = cJSON_CreateObject();
    cJSON_AddNumberToObject(pid, "pos_e", pid_pos.fpE);
    cJSON_AddNumberToObject(pid, "pos_u", pid_pos.fpU);
    cJSON_AddNumberToObject(pid, "vel_e", pid_vel.fpE);
    cJSON_AddNumberToObject(pid, "vel_u", pid_vel.fpU);
    cJSON_AddNumberToObject(pid, "pit_e", pid_pit.fpE);
    cJSON_AddNumberToObject(pid, "pit_u", pid_pit.fpU);
    cJSON_AddItemToObject(root, "pid", pid);

    // Policy runtime state, so a tuning session never has to guess.
    {
        const ctrl_out_t *a = arbiter_get();
        cJSON *cfc = cJSON_CreateObject();
        cJSON_AddNumberToObject(cfc, "src", arbiter_source());
        cJSON_AddNumberToObject(cfc, "active", arbiter_active() ? 1 : 0);
        cJSON_AddNumberToObject(cfc, "slot", cfc_model_active_slot());
        // bit0 = int8 权重, bit1 = int8 SIMD 内核（见 cfc_model_runtime_flags）
        cJSON_AddNumberToObject(cfc, "q", cfc_model_runtime_flags());
        // 权重驻留：1 = 内部 SRAM，0 = PSRAM。单步耗时的大头是权重访存，
        // 不带上这个比特就没法解释同一个 blob 的两次测量为什么差一倍。
        cJSON_AddNumberToObject(cfc, "mem", cfc_model_blob_in_sram());
        {
            // 上一帧 forward 的分段耗时（norm / proj / cell / heads，微秒）
            int64_t sp[4];
            cfc_model_last_split(sp);
            cJSON_AddNumberToObject(cfc, "n_us", (double)sp[0]);
            cJSON_AddNumberToObject(cfc, "p_us", (double)sp[1]);
            cJSON_AddNumberToObject(cfc, "s_us", (double)sp[2]);
            cJSON_AddNumberToObject(cfc, "h_us", (double)sp[3]);
        }
        cJSON_AddNumberToObject(cfc, "us", (double)cfc_task_last_us());
        cJSON_AddNumberToObject(cfc, "hz", cfc_task_last_hz());
        cJSON_AddNumberToObject(cfc, "frames", cfc_task_frames());
        cJSON_AddNumberToObject(cfc, "dropped", cfc_task_dropped());
        cJSON_AddNumberToObject(cfc, "feat_age_ms", (double)arbiter_feature_age_ms());
        cJSON_AddNumberToObject(cfc, "danger", gaze_guard_danger() ? 1 : 0);
        cJSON_AddNumberToObject(cfc, "block", arbiter_block_reason());
        cJSON_AddNumberToObject(cfc, "head_des", a->head_des);
        cJSON_AddNumberToObject(cfc, "vx_des", a->vx_des);
        cJSON_AddNumberToObject(cfc, "vyaw_des", a->vyaw_des);
        cJSON_AddNumberToObject(cfc, "cmode", a->mode);
        cJSON_AddNumberToObject(cfc, "emote", a->emote);
        cJSON_AddItemToObject(root, "cfc", cfc);

        cJSON *tel = cJSON_CreateObject();
        cJSON_AddNumberToObject(tel, "frames", (double)telemetry_stream_frames());
        cJSON_AddNumberToObject(tel, "client", (double)telemetry_stream_client_ip());
        cJSON_AddNumberToObject(tel, "enabled", telemetry_stream_enabled() ? 1 : 0);
        cJSON_AddItemToObject(root, "telemetry", tel);
    }

    cJSON_AddNumberToObject(root, "heap", (double)esp_get_free_heap_size());

    char *json_str = cJSON_PrintUnformatted(root);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    httpd_resp_sendstr(req, json_str);
    cJSON_free(json_str);
    cJSON_Delete(root);
    return ESP_OK;
}

// GET /cmd?vx=&vyaw=&head=&mode=&pitch=&stop=1
static esp_err_t cmd_handler(httpd_req_t *req) {
    force_nodelay(req);
    char buf[160];
    char param[24];

    int qlen = httpd_req_get_url_query_len(req) + 1;
    if (qlen <= 1 || qlen > (int)sizeof(buf)) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "bad query");
        return ESP_FAIL;
    }
    if (httpd_req_get_url_query_str(req, buf, sizeof(buf)) != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "bad query");
        return ESP_FAIL;
    }

    bool drive = false;
    if (httpd_query_key_value(buf, "vx", param, sizeof(param)) == ESP_OK) {
        vx = (float)atof(param);
        drive = true;
    }
    if (httpd_query_key_value(buf, "vyaw", param, sizeof(param)) == ESP_OK) {
        vyaw = (float)atof(param);
        drive = true;
    }
    if (httpd_query_key_value(buf, "head", param, sizeof(param)) == ESP_OK) {
        target_head_pos = (float)atof(param);
    }
    if (httpd_query_key_value(buf, "mode", param, sizeof(param)) == ESP_OK) {
        SetControlMode(atoi(param));
    }
    if (httpd_query_key_value(buf, "pitch", param, sizeof(param)) == ESP_OK) {
        SetGimbalPitch((float)atof(param), false);
    }
    if (httpd_query_key_value(buf, "stop", param, sizeof(param)) == ESP_OK) {
        vx = 0.0f;
        vyaw = 0.0f;
        drive = false;
    }

    if (drive) {
        cmd_watchdog_arm();
    }

    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, "{\"ok\":true}");
    return ESP_OK;
}

// GET /capture?q=NN -> single JPEG frame.
//
// `q` (10..95) tunes the software JPEG quality live. /capture is the whole
// latency budget now, and the encoder cost scales with quality while the
// perception chain only ever sees 240x240 grayscale, so the usable operating
// point has to be found by measurement rather than guessed.
static esp_err_t capture_handler(httpd_req_t *req) {
    force_nodelay(req);
    auto *cam = dynamic_cast<Esp32Camera *>(Board::GetInstance().GetCamera());
    if (cam == nullptr) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "no camera");
        return ESP_FAIL;
    }

    int quality = CAPTURE_QUALITY_DEFAULT;
    char qbuf[32];
    char qval[8];
    if (httpd_req_get_url_query_str(req, qbuf, sizeof(qbuf)) == ESP_OK &&
        httpd_query_key_value(qbuf, "q", qval, sizeof(qval)) == ESP_OK) {
        const long q = strtol(qval, nullptr, 10);
        if (q >= 10 && q <= 95) {
            quality = (int)q;
        }
    }

    uint8_t *jpg = nullptr;
    size_t jpg_len = 0;
    if (!cam->CaptureJpeg(&jpg, &jpg_len, quality)) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "capture failed");
        return ESP_FAIL;
    }

    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=frame.jpg");
    esp_err_t err = httpd_resp_send(req, (const char *)jpg, jpg_len);
    // No per-frame log here: at the 115200 console a one-line ESP_LOGI costs
    // ~3 ms and runs after the response, which lands squarely in the next
    // request of a keep-alive loop. Capture timing is sampled 1-in-8 inside
    // Esp32Camera::CaptureJpeg instead.
    free(jpg);
    return err;
}

// ---------------------------------------------------------------------------
// Vision downlink: the PC runs YOLO, the robot receives the full feature set
// and (later) runs the policy locally. Binary in, timing out.
// Frame layout (little endian):
//   u32 t_ms | u16 n_obj | n_obj * { i16 cx,cy,w,h (x1000) | u8 conf | u8 cls }
// ---------------------------------------------------------------------------

#define VISION_BUF_SIZE 8192
static uint8_t vision_buf[VISION_BUF_SIZE];
static size_t vision_last_len = 0;
static uint16_t vision_last_nobj = 0;
static uint32_t vision_last_t = 0;

// POST /vision
static esp_err_t vision_post_handler(httpd_req_t *req) {
    force_nodelay(req);
    int64_t t0 = esp_timer_get_time();
    int total = 0;

    while (total < VISION_BUF_SIZE) {
        int r = httpd_req_recv(req, (char *)vision_buf + total, VISION_BUF_SIZE - total);
        if (r == HTTPD_SOCK_ERR_TIMEOUT) {
            continue;
        }
        if (r <= 0) {
            break;
        }
        total += r;
    }
    int64_t t1 = esp_timer_get_time();

    uint16_t nobj = 0;
    uint32_t ts = 0;
    if (total >= 6) {
        memcpy(&ts, vision_buf, 4);
        memcpy(&nobj, vision_buf + 4, 2);
        // Sanity-check the object count against the payload we actually got.
        if (6 + (size_t)nobj * 10 > (size_t)total) {
            nobj = (uint16_t)((total - 6) / 10);
        }
        vision_last_len = (size_t)total;
        vision_last_nobj = nobj;
        vision_last_t = ts;
    }
    int64_t t2 = esp_timer_get_time();

    char resp[160];
    snprintf(resp, sizeof(resp),
             "{\"rx\":%d,\"nobj\":%u,\"rx_us\":%lld,\"parse_us\":%lld,\"heap\":%u}",
             total, (unsigned)nobj, (long long)(t1 - t0), (long long)(t2 - t1),
             (unsigned)esp_get_free_heap_size());
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, resp);
    ESP_LOGI(TAG, "vision: %d bytes, %u objs, rx=%lldus parse=%lldus",
             total, (unsigned)nobj, (long long)(t1 - t0), (long long)(t2 - t1));
    return ESP_OK;
}

// GET /vision -> what the robot currently holds
static esp_err_t vision_get_handler(httpd_req_t *req) {
    force_nodelay(req);
    char resp[128];
    snprintf(resp, sizeof(resp), "{\"len\":%u,\"nobj\":%u,\"t\":%u}",
             (unsigned)vision_last_len, (unsigned)vision_last_nobj, (unsigned)vision_last_t);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, resp);
    return ESP_OK;
}

// ---------------------------------------------------------------------------
// Policy runtime API (TECH-PLAN 11.4)
//
//   POST /feat    binary vision frame  -> cfc_task snapshot
//   POST /model   weight blob upload (atomic, per slot)
//   GET  /model   slot table
//   POST /mode    manual | teacher | cfc | slot0 | slot1
//   POST /cfg     runtime parameters (JSON body or query string)
//   GET  /cfg     every runtime parameter back
//   POST /emote   LCD emotion (index or name)
//   GET  /emote
//   POST /gaze    table-edge danger flag (highest priority stop)
//   GET  /gaze
//
// None of these handlers touches the 500 Hz loop; they only publish through
// the seqlocks owned by control_arbiter / cfc_task.
// ---------------------------------------------------------------------------

#define FEAT_RX_MAX 2048

static uint8_t feat_rx_buf[FEAT_RX_MAX];

// Copy the URL query string into buf. Returns false when there is none.
static bool query_load(httpd_req_t *req, char *buf, size_t n) {
    int qlen = httpd_req_get_url_query_len(req) + 1;
    if (qlen <= 1 || (size_t)qlen > n) {
        return false;
    }
    return httpd_req_get_url_query_str(req, buf, n) == ESP_OK;
}

static bool query_param(httpd_req_t *req, const char *key, char *out, size_t n) {
    char q[192];
    if (!query_load(req, q, sizeof(q))) {
        return false;
    }
    return httpd_query_key_value(q, key, out, n) == ESP_OK;
}

// Read exactly req->content_len bytes into `dst`. Returns the byte count or -1.
static int recv_body(httpd_req_t *req, uint8_t *dst, size_t cap) {
    size_t want = req->content_len;
    size_t got = 0;
    if (want == 0 || want > cap) {
        return -1;
    }
    while (got < want) {
        int r = httpd_req_recv(req, (char *)dst + got, want - got);
        if (r == HTTPD_SOCK_ERR_TIMEOUT) {
            continue;
        }
        if (r <= 0) {
            break;
        }
        got += (size_t)r;
    }
    return (got == want) ? (int)got : -1;
}

// POST /feat -- binary frame, see TECH-PLAN 5.2.
static esp_err_t feat_post_handler(httpd_req_t *req) {
    force_nodelay(req);
    cfc_feat_reply_t rep;
    char resp[320];
    int total;

    if (req->content_len == 0 || req->content_len > FEAT_RX_MAX) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "bad length");
        return ESP_FAIL;
    }
    total = recv_body(req, feat_rx_buf, sizeof(feat_rx_buf));
    if (total < 0) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "short body");
        return ESP_FAIL;
    }

    cfc_feat_ingest(feat_rx_buf, (size_t)total, &rep);

    snprintf(resp, sizeof(resp),
             "{\"ok\":%d,\"err\":\"%s\",\"rx\":%d,\"n_obj\":%d,\"n_feat\":%d,"
             "\"age_ms\":%d,\"dt_ms\":%d,\"cfc_us\":%d,\"cfc_hz\":%d,\"heap\":%u}",
             rep.ok, rep.err ? rep.err : "", rep.rx_bytes, rep.n_obj, rep.n_feat,
             rep.age_ms, rep.dt_ms, (int)rep.cfc_us, rep.cfc_hz,
             (unsigned)esp_get_free_heap_size());
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    httpd_resp_sendstr(req, resp);
    return ESP_OK;
}

// POST /step -- the whole control loop in one round trip.
//
// The PC used to do GET /capture then POST /feat: two round trips per decision,
// and the second carried no state the robot needed before capturing. /step takes
// exactly the /feat frame and answers with the next JPEG, so the observation and
// the reply share one request. The CFC acknowledgement rides in an X-Cfc header
// because the body is the image.
static esp_err_t step_handler(httpd_req_t *req) {
    force_nodelay(req);
    cfc_feat_reply_t rep;
    int total;

    if (req->content_len == 0 || req->content_len > FEAT_RX_MAX) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "bad length");
        return ESP_FAIL;
    }
    total = recv_body(req, feat_rx_buf, sizeof(feat_rx_buf));
    if (total < 0) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "short body");
        return ESP_FAIL;
    }
    cfc_feat_ingest(feat_rx_buf, (size_t)total, &rep);

    auto *cam = dynamic_cast<Esp32Camera *>(Board::GetInstance().GetCamera());
    if (cam == nullptr) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "no camera");
        return ESP_FAIL;
    }

    uint8_t *jpg = nullptr;
    size_t jpg_len = 0;
    if (!cam->CaptureJpeg(&jpg, &jpg_len, CAPTURE_QUALITY_DEFAULT)) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "capture failed");
        return ESP_FAIL;
    }

    char hdr[192];
    snprintf(hdr, sizeof(hdr),
             "{\"ok\":%d,\"age_ms\":%d,\"cfc_us\":%d,\"hz\":%d,\"heap\":%u}",
             rep.ok, rep.age_ms, (int)rep.cfc_us, rep.cfc_hz,
             (unsigned)esp_get_free_heap_size());
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    httpd_resp_set_hdr(req, "X-Cfc", hdr);
    esp_err_t err = httpd_resp_send(req, (const char *)jpg, jpg_len);
    free(jpg);
    return err;
}

// POST /model?slot=0|1 -- streamed into PSRAM, validated, then committed.
static esp_err_t model_post_handler(httpd_req_t *req) {
    force_nodelay(req);
    char param[16];
    char err[96];
    char resp[320];
    int slot = -1;
    size_t len = req->content_len;
    uint8_t *buf;
    int rc;

    if (!query_param(req, "slot", param, sizeof(param))) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "slot=0 or slot=1 required");
        return ESP_FAIL;
    }
    slot = atoi(param);
    if (slot != 0 && slot != 1) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "slot must be 0 or 1");
        return ESP_FAIL;
    }
    if (!cfc_model_storage_ok()) {
        snprintf(resp, sizeof(resp), "{\"ok\":0,\"err\":\"%s\"}",
                 cfc_model_last_error());
        httpd_resp_set_type(req, "application/json");
        httpd_resp_sendstr(req, resp);
        return ESP_OK;
    }
    if (len < CFC_HDR_BYTES || len > CFC_SLOT_MAX_BYTES) {
        snprintf(resp, sizeof(resp),
                 "{\"ok\":0,\"err\":\"content-length %u outside [%u,%u]\"}",
                 (unsigned)len, (unsigned)CFC_HDR_BYTES, (unsigned)CFC_SLOT_MAX_BYTES);
        httpd_resp_set_type(req, "application/json");
        httpd_resp_sendstr(req, resp);
        return ESP_OK;
    }

    buf = (uint8_t *)heap_caps_malloc(len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (buf == nullptr) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "no PSRAM");
        return ESP_FAIL;
    }
    if (recv_body(req, buf, len) != (int)len) {
        heap_caps_free(buf);
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "short body");
        return ESP_FAIL;
    }

    err[0] = 0;
    // Flash erase/program disables the instruction cache for short bursts.
    // Hold the wheel channels at zero for the duration so a missed loop tick
    // cannot turn into motion.
    arbiter_set_upload(true);
    rc = cfc_model_upload(slot, buf, len, err, sizeof(err));
    arbiter_set_upload(false);
    heap_caps_free(buf);

    if (rc == 0) {
        int in_dim = 0, hidden = 0, heads = 0;
        cfc_model_slot_dims(slot, &in_dim, &hidden, &heads);
        snprintf(resp, sizeof(resp),
                 "{\"ok\":1,\"slot\":%d,\"bytes\":%u,\"crc\":%u,\"in_dim\":%d,"
                 "\"hidden\":%d,\"n_heads\":%d,\"active\":%d}",
                 slot, (unsigned)cfc_model_slot_bytes(slot),
                 (unsigned)cfc_model_slot_crc(slot), in_dim, hidden, heads,
                 cfc_model_active_slot());
    } else {
        snprintf(resp, sizeof(resp), "{\"ok\":0,\"slot\":%d,\"err\":\"%s\"}",
                 slot, err);
    }
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    httpd_resp_sendstr(req, resp);
    return ESP_OK;
}

// GET /model
static esp_err_t model_get_handler(httpd_req_t *req) {
    force_nodelay(req);
    cJSON *root = cJSON_CreateObject();
    cJSON *slots = cJSON_CreateArray();
    int i;

    cJSON_AddNumberToObject(root, "storage_ok", cfc_model_storage_ok() ? 1 : 0);
    cJSON_AddStringToObject(root, "err", cfc_model_last_error());
    cJSON_AddNumberToObject(root, "active", cfc_model_active_slot());
    cJSON_AddNumberToObject(root, "src", arbiter_source());

    for (i = 0; i < 2; i++) {
        cJSON *o = cJSON_CreateObject();
        int in_dim = 0, hidden = 0, heads = 0;
        const int valid = cfc_model_slot_valid(i);
        cJSON_AddNumberToObject(o, "slot", i);
        cJSON_AddNumberToObject(o, "valid", valid > 0 ? 1 : 0);
        cJSON_AddNumberToObject(o, "bytes", (double)cfc_model_slot_bytes(i));
        cJSON_AddNumberToObject(o, "crc", (double)cfc_model_slot_crc(i));
        if (valid > 0 && cfc_model_slot_dims(i, &in_dim, &hidden, &heads) == 0) {
            cJSON_AddNumberToObject(o, "in_dim", in_dim);
            cJSON_AddNumberToObject(o, "hidden", hidden);
            cJSON_AddNumberToObject(o, "n_heads", heads);
        }
        cJSON_AddItemToArray(slots, o);
    }
    cJSON_AddItemToObject(root, "slots", slots);
    cJSON_AddNumberToObject(root, "cfc_us", (double)cfc_task_last_us());
    cJSON_AddNumberToObject(root, "cfc_hz", cfc_task_last_hz());
    cJSON_AddNumberToObject(root, "frames", cfc_task_frames());
    cJSON_AddNumberToObject(root, "dropped", cfc_task_dropped());

    char *json_str = cJSON_PrintUnformatted(root);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    httpd_resp_sendstr(req, json_str);
    cJSON_free(json_str);
    cJSON_Delete(root);
    return ESP_OK;
}

// POST /mode?src=manual|teacher|cfc|slot0|slot1
// Teacher push (repeat calls at 10-30 Hz): &head=&vx=&vyaw=&md=&em=
static esp_err_t mode_post_handler(httpd_req_t *req) {
    force_nodelay(req);
    char param[24];
    char resp[256];
    const char *src = "manual";
    int src_id = ARB_SRC_MANUAL;

    if (query_param(req, "src", param, sizeof(param))) {
        src = param;
    }

    if (strcmp(src, "manual") == 0) {
        src_id = ARB_SRC_MANUAL;
    } else if (strcmp(src, "teacher") == 0) {
        src_id = ARB_SRC_TEACHER;
    } else if (strcmp(src, "cfc") == 0 || strcmp(src, "slot0") == 0 ||
               strcmp(src, "slot1") == 0) {
        const int slot = (strcmp(src, "slot1") == 0) ? 1 : 0;
        src_id = ARB_SRC_CFC;
        if (cfc_model_select(slot) != 0) {
            snprintf(resp, sizeof(resp),
                     "{\"ok\":0,\"err\":\"%s\",\"src\":%d,\"active\":%d}",
                     cfc_model_last_error(), arbiter_source(),
                     cfc_model_active_slot());
            httpd_resp_set_type(req, "application/json");
            httpd_resp_sendstr(req, resp);
            return ESP_OK;
        }
    } else {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "unknown src");
        return ESP_FAIL;
    }

    arbiter_set_source(src_id);

    if (src_id == ARB_SRC_TEACHER) {
        const ctrl_out_t *prev = arbiter_get();
        float head = prev->head_des;
        float vx = prev->vx_des;
        float vyaw = prev->vyaw_des;
        int md = prev->mode;
        int em = prev->emote;
        if (query_param(req, "head", param, sizeof(param))) head = (float)atof(param);
        if (query_param(req, "vx", param, sizeof(param))) vx = (float)atof(param);
        if (query_param(req, "vyaw", param, sizeof(param))) vyaw = (float)atof(param);
        if (query_param(req, "md", param, sizeof(param))) md = atoi(param);
        if (query_param(req, "em", param, sizeof(param))) em = atoi(param);
        arbiter_set_teacher(head, vx, vyaw, md, em);
    }

    snprintf(resp, sizeof(resp),
             "{\"ok\":1,\"src\":%d,\"active\":%d,\"act\":%d}",
             arbiter_source(), cfc_model_active_slot(), arbiter_active() ? 1 : 0);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    httpd_resp_sendstr(req, resp);
    return ESP_OK;
}

// GET /cfg -- every runtime parameter, no hidden state.
static esp_err_t cfg_get_handler(httpd_req_t *req) {
    force_nodelay(req);
    cJSON *root = cJSON_CreateObject();
    const int n = arbiter_cfg_count();
    int i;

    for (i = 0; i < n; i++) {
        const char *name = arbiter_cfg_name(i);
        if (name == nullptr) {
            continue;
        }
        if (arbiter_cfg_is_int(i)) {
            cJSON_AddNumberToObject(root, name, arbiter_cfg_get_int(i));
        } else {
            cJSON_AddNumberToObject(root, name, arbiter_cfg_get_float(i));
        }
    }
    char *json_str = cJSON_PrintUnformatted(root);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    httpd_resp_sendstr(req, json_str);
    cJSON_free(json_str);
    cJSON_Delete(root);
    return ESP_OK;
}

// POST /cfg -- JSON body {"vx_max":0.2,...} and/or ?vx_max=0.2
static esp_err_t cfg_post_handler(httpd_req_t *req) {
    force_nodelay(req);
    char body[1024];
    char resp[192];
    int applied = 0;
    int rejected = 0;

    if (req->content_len > 0) {
        if (req->content_len >= sizeof(body)) {
            httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "cfg too large");
            return ESP_FAIL;
        }
        if (recv_body(req, (uint8_t *)body, sizeof(body) - 1) < 0) {
            httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "short body");
            return ESP_FAIL;
        }
        body[req->content_len] = 0;

        cJSON *root = cJSON_Parse(body);
        if (root != nullptr) {
            for (cJSON *it = root->child; it != nullptr; it = it->next) {
                if (it->string == nullptr || !cJSON_IsNumber(it)) {
                    rejected++;
                    continue;
                }
                if (arbiter_cfg_set(it->string, (float)it->valuedouble) == 0) {
                    applied++;
                } else {
                    rejected++;
                    ESP_LOGW(TAG, "cfg: unknown key '%s'", it->string);
                }
            }
            cJSON_Delete(root);
        } else {
            ESP_LOGW(TAG, "cfg: JSON parse failed");
            rejected++;
        }
    }

    {
        char q[256];
        if (query_load(req, q, sizeof(q))) {
            char *p = q;
            while (p != nullptr && *p != 0) {
                char *amp = strchr(p, '&');
                if (amp != nullptr) {
                    *amp = 0;
                }
                char *eq = strchr(p, '=');
                if (eq != nullptr) {
                    *eq = 0;
                    if (arbiter_cfg_set(p, (float)atof(eq + 1)) == 0) {
                        applied++;
                    } else {
                        rejected++;
                        ESP_LOGW(TAG, "cfg: unknown key '%s'", p);
                    }
                }
                p = (amp != nullptr) ? (amp + 1) : nullptr;
            }
        }
    }

    snprintf(resp, sizeof(resp), "{\"ok\":1,\"applied\":%d,\"rejected\":%d}", applied,
             rejected);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    httpd_resp_sendstr(req, resp);
    return ESP_OK;
}

// LCD emotions, in the order of main/boards/hover/240_240/emote.json.
static const char *const kEmoteNames[] = {
    "neutral", "happy", "sad", "angry", "surprised", "thinking", "winking", "cool",
    "laughing", "loving", "crying", "confused", "embarrassed", "sleepy",
    "confident", "delicious", "kiss", "relaxed", "shocked", "silly",
    "icon_speaker_zzz", "listen", "nvs_reset", "wificonfig", "calibration",
    "scanning", "launch", "remote_mode",
};
#define EMOTE_COUNT ((int)(sizeof(kEmoteNames) / sizeof(kEmoteNames[0])))

// CfC emote channel 0..7 -> LCD emotion name.
static const char *const kCfcEmotes[8] = {
    "neutral", "happy", "surprised", "thinking", "winking", "laughing", "loving", "cool",
};

static bool display_set_emotion(const char *name) {
    if (name == nullptr) {
        return false;
    }
    auto *display = Board::GetInstance().GetDisplay();
    if (display == nullptr) {
        return false;
    }
    display->SetEmotion(name);   // EmoteDisplay locks internally
    return true;
}

// POST /emote?i=0..27 or ?name=happy
static esp_err_t emote_post_handler(httpd_req_t *req) {
    force_nodelay(req);
    char param[40];
    char resp[160];
    const char *name = nullptr;

    if (query_param(req, "name", param, sizeof(param))) {
        name = param;
    } else if (query_param(req, "i", param, sizeof(param))) {
        const int idx = atoi(param);
        if (idx < 0 || idx >= EMOTE_COUNT) {
            httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "i out of range");
            return ESP_FAIL;
        }
        name = kEmoteNames[idx];
    }
    if (name == nullptr) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "i= or name= required");
        return ESP_FAIL;
    }
    const bool ok = display_set_emotion(name);
    snprintf(resp, sizeof(resp), "{\"ok\":%d,\"emote\":\"%s\"}", ok ? 1 : 0, name);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, resp);
    return ESP_OK;
}

static esp_err_t emote_get_handler(httpd_req_t *req) {
    force_nodelay(req);
    cJSON *root = cJSON_CreateObject();
    cJSON *arr = cJSON_CreateArray();
    int i;

    for (i = 0; i < EMOTE_COUNT; i++) {
        cJSON_AddItemToArray(arr, cJSON_CreateString(kEmoteNames[i]));
    }
    cJSON_AddItemToObject(root, "names", arr);
    cJSON_AddNumberToObject(root, "arbiter_emote", arbiter_emote());
    char *json_str = cJSON_PrintUnformatted(root);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    httpd_resp_sendstr(req, json_str);
    cJSON_free(json_str);
    cJSON_Delete(root);
    return ESP_OK;
}

// POST /gaze?danger=1|0 (also accepts d=). Highest priority stop.
static esp_err_t gaze_post_handler(httpd_req_t *req) {
    force_nodelay(req);
    char param[16];
    char resp[128];
    int danger;

    if (query_param(req, "danger", param, sizeof(param))) {
        danger = atoi(param);
    } else if (query_param(req, "d", param, sizeof(param))) {
        danger = atoi(param);
    } else {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "danger=0|1 required");
        return ESP_FAIL;
    }

    gaze_guard_set(danger != 0);
    snprintf(resp, sizeof(resp), "{\"ok\":1,\"danger\":%d,\"latched\":%d,\"events\":%u}",
             gaze_guard_danger() ? 1 : 0, gaze_guard_latched() ? 1 : 0,
             (unsigned)gaze_guard_events());
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    httpd_resp_sendstr(req, resp);
    return ESP_OK;
}

static esp_err_t gaze_get_handler(httpd_req_t *req) {
    force_nodelay(req);
    char resp[160];
    snprintf(resp, sizeof(resp),
             "{\"danger\":%d,\"latched\":%d,\"events\":%u,\"age_ms\":%d}",
             gaze_guard_danger() ? 1 : 0, gaze_guard_latched() ? 1 : 0,
             (unsigned)gaze_guard_events(), (int)gaze_guard_age_ms());
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    httpd_resp_sendstr(req, resp);
    return ESP_OK;
}

// Mirror the CfC emote channel onto the LCD. Change-driven only, so the
// display is touched at most once per new emotion.
static void emote_follow_task(void *arg) {
    int last = -1;
    (void)arg;
    for (;;) {
        if (arbiter_source() == ARB_SRC_CFC) {
            const int e = arbiter_emote();
            if (e != last && e >= 0 && e < 8) {
                last = e;
                display_set_emotion(kCfcEmotes[e]);
            }
        } else {
            last = -1;
        }
        vTaskDelay(pdMS_TO_TICKS(250));
    }
}

// NOTE: there used to be a GET /bench here that timed cfc_model_forward() in a
// synchronous loop. It is removed: it ran a full forward pass on the httpd
// worker's 4 KB stack and crashed the board the first time it was called.
// Per-step cost is measured with the saturation method instead -- raise
// `cfc_hz` via POST /cfg until cfc_task can no longer keep up, then read the
// achieved rate from /status. That needs no firmware endpoint and cannot crash
// the server.

void hover_debug_server_start(void) {
    if (server != NULL) {
        ESP_LOGW(TAG, "Server already running");
        return;
    }

    /* HTTP server FIRST, policy runtime second.
     *
     * cfc_model_init() keeps the int8 weight blob (<= CFC_SRAM_BLOB_MAX) in
     * internal SRAM on purpose: the SIMD kernel then reads it without PSRAM
     * cache misses. That claim is ~43 KB, and it left 4859 bytes of internal
     * heap before httpd_start() -- less than the 4096-byte httpd task stack --
     * so httpd_start() returned ESP_ERR_HTTPD_TASK and the board came up with
     * no endpoints at all while the balance loop ran happily. Starting the
     * server first makes endpoint availability independent of model size.
     *
     * URI handlers are registered only after the runtime is up, so a request
     * arriving inside the ~100 ms init window gets 404 instead of reaching a
     * half-initialised arbiter. */
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;
    // 19 handlers are registered in s_uris below; keep two slots of headroom so
    // adding an endpoint cannot silently overflow the table (registration
    // failures are logged, but a full table would drop them at runtime).
    config.max_uri_handlers = 22;

    // Socket pool. The default (7) is exhausted by rapid polling: every short
    // connection parks in TIME_WAIT, accept() then fails with ENFILE and the
    // whole HTTP server looks dead while the balance loop keeps running. Seen
    // in practice after ~40 back-to-back requests. lru_purge_enable reclaims
    // the least recently used socket instead of refusing the connection.
    config.max_open_sockets = 13;
    config.lru_purge_enable = true;
    config.recv_wait_timeout = 15;
    config.send_wait_timeout = 15;
    config.keep_alive_enable = true;
    config.keep_alive_idle = 10;
    config.keep_alive_interval = 3;
    config.keep_alive_count = 3;

    ESP_LOGI(TAG, "httpd start: max_uri_handlers=%d internal_free=%u",
             config.max_uri_handlers,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    esp_err_t ret = httpd_start(&server, &config);
    if (ret != ESP_OK) {
        server = NULL;
        ESP_LOGE(TAG, "Failed to start server: %s", esp_err_to_name(ret));
        return;
    }

    // Bisect switch: HOVER_DBG_SKIP_INIT=1 starts ONLY the HTTP server, so a
    // crash can be attributed either to httpd itself or to the policy runtime
    // it shares the task with.
#ifndef HOVER_DBG_SKIP_INIT
#define HOVER_DBG_SKIP_INIT 0
#endif
#if HOVER_DBG_SKIP_INIT >= 3
    arbiter_init();
    gaze_guard_init();
    ESP_LOGW(TAG, "BISECT: arbiter+gaze only (no model init)");
#elif HOVER_DBG_SKIP_INIT >= 2
    arbiter_init();
    gaze_guard_init();
    if (!cfc_model_init()) {
        ESP_LOGE(TAG, "model storage unavailable: %s", cfc_model_last_error());
    }
    ESP_LOGW(TAG, "BISECT: init + model only (no cfc task, no telemetry)");
#elif HOVER_DBG_SKIP_INIT == 1
    ESP_LOGW(TAG, "BISECT: policy runtime init SKIPPED, httpd only");
#else
    // Policy runtime: safe to call repeatedly, and every failure degrades to
    // "that feature is off" instead of touching the balance loop.
    //
    // Allocation order is deliberately "mandatory first, optimisation last".
    // Internal SRAM here is ~48 KB after the httpd is up; cfc_task needs a
    // 12 KB contiguous stack and the int8 weight blob wants ~33 KB more, so
    // the two together sit exactly on the edge and whether both survive used
    // to be luck -- observed failing with "failed to create cfc_task
    // (inference disabled)" while the board otherwise looked healthy.
    // cfc_model_init() already falls back to PSRAM when the internal
    // allocation does not fit, so the model is the one that may move;
    // the task is the one that must not fail.
    arbiter_init();
    gaze_guard_init();
    cfc_task_start();
    if (!cfc_model_init()) {
        ESP_LOGE(TAG, "model storage unavailable: %s", cfc_model_last_error());
    }
    telemetry_stream_start();
    // Cosmetic only: makes the face follow the gaze. Last, so it can never
    // take memory away from the inference path.
    if (xTaskCreatePinnedToCore(emote_follow_task, "emote_follow", 3072, NULL, 2,
                                NULL, 0) != pdPASS) {
        ESP_LOGW(TAG, "emote follow task not started (internal SRAM too low)");
    }
#endif

    /* The descriptor table is static so it does not depend on this frame living
     * long enough, and every registration is checked: a silently ignored error
     * here shows up much later as "404 on every endpoint", which is a day of
     * debugging for a one-line log. */
    static const httpd_uri_t s_uris[] = {
        { .uri = "/",          .method = HTTP_GET,  .handler = index_handler },
        { .uri = "/api/data",  .method = HTTP_GET,  .handler = data_handler },
        { .uri = "/api/set",   .method = HTTP_GET,  .handler = set_handler },
        { .uri = "/status",    .method = HTTP_GET,  .handler = status_handler },
        { .uri = "/cmd",       .method = HTTP_GET,  .handler = cmd_handler },
        { .uri = "/capture",   .method = HTTP_GET,  .handler = capture_handler },
        { .uri = "/vision",    .method = HTTP_POST, .handler = vision_post_handler },
        { .uri = "/vision",    .method = HTTP_GET,  .handler = vision_get_handler },
        { .uri = "/feat",      .method = HTTP_POST, .handler = feat_post_handler },
        { .uri = "/step",      .method = HTTP_POST, .handler = step_handler },
        { .uri = "/model",     .method = HTTP_POST, .handler = model_post_handler },
        { .uri = "/model",     .method = HTTP_GET,  .handler = model_get_handler },
        { .uri = "/mode",      .method = HTTP_POST, .handler = mode_post_handler },
        { .uri = "/cfg",       .method = HTTP_POST, .handler = cfg_post_handler },
        { .uri = "/cfg",       .method = HTTP_GET,  .handler = cfg_get_handler },
        { .uri = "/emote",     .method = HTTP_POST, .handler = emote_post_handler },
        { .uri = "/emote",     .method = HTTP_GET,  .handler = emote_get_handler },
        { .uri = "/gaze",      .method = HTTP_POST, .handler = gaze_post_handler },
        { .uri = "/gaze",      .method = HTTP_GET,  .handler = gaze_get_handler },
    };
    for (size_t i = 0; i < sizeof(s_uris) / sizeof(s_uris[0]); ++i) {
        esp_err_t err = httpd_register_uri_handler(server, &s_uris[i]);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "register %s failed: %s (internal_free=%u)",
                     s_uris[i].uri, esp_err_to_name(err),
                     (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
        }
    }

    ESP_LOGI(TAG, "Debug server started on port 80 (internal_free=%u)",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
}

void hover_debug_server_stop(void) {
    if (server != NULL) {
        httpd_stop(server);
        server = NULL;
        ESP_LOGI(TAG, "Debug server stopped");
    }
}

bool hover_debug_server_is_running(void) {
    return server != NULL;
}
