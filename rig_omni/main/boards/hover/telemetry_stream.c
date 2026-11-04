/*
 * telemetry_stream.c -- 500 Hz UDP telemetry.
 *
 * One socket, one low priority task on core 0. The task is the only user of
 * the socket, so no locking is needed. Frames are assembled in a local buffer
 * from plain loads of the robot globals; a value that updates mid-frame is
 * still a valid recent value, which is exactly what a data collection host
 * wants (125 us of skew between two fields is irrelevant here).
 */

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "lwip/sockets.h"
#include "lwip/inet.h"

#include "robot.h"
#include "imu.h"

#include "gaze_guard.h"
#include "control_arbiter.h"
#include "telemetry_stream.h"

static const char *TAG = "Telemetry";

/* Defined in robot.cc, not declared in robot.h. */
extern float wheel_x;
extern float wheek_vx;
extern int   robot_state;

#define TLM_MAGIC      0x314D4C54u /* 'TLM1' */
#define TLM_VER        1
#define TLM_NFLOAT     34
#define TLM_HDR        16
#define TLM_BYTES      (TLM_HDR + TLM_NFLOAT * 4)
#define TLM_PRIO       3
#define TLM_CORE       0
#define TLM_STACK      4096
#define TLM_CLIENT_TTL_MS 5000

static TaskHandle_t s_task;
static volatile bool s_enabled = true;
static volatile uint32_t s_frames;
static volatile uint32_t s_client_ip;
static int64_t s_client_seen_us;

static void wr_u32(uint8_t *p, uint32_t v)
{
    memcpy(p, &v, 4);
}

static void wr_u16(uint8_t *p, uint16_t v)
{
    memcpy(p, &v, 2);
}

static void wr_f32(uint8_t *p, float v)
{
    memcpy(p, &v, 4);
}

static void put(const ctrl_out_t *arb, uint8_t *buf)
{
    float f[TLM_NFLOAT];
    int i;

    /* imu */
    f[0] = roll;
    f[1] = pitch;
    f[2] = yaw;
    f[3] = dq;
    f[4] = pit_gyro;
    f[5] = accel_x;
    f[6] = accel_y;
    f[7] = accel_z;
    /* wheel */
    f[8] = wheel1_vel;
    f[9] = wheel2_vel;
    f[10] = wheel1_x;
    f[11] = wheel2_x;
    f[12] = wheel_x;
    f[13] = wheek_vx;
    /* head */
    f[14] = q_head;
    f[15] = target_head_pos;
    /* ctrl */
    f[16] = stable_pos;
    f[17] = stable_yaw;
    f[18] = vx;
    f[19] = vyaw;
    f[20] = tor1;
    f[21] = tor2;
    /* state */
    f[22] = (float)robot_state;
    f[23] = (float)control_mode;
    /* arbiter */
    f[24] = arb->head_des;
    f[25] = arb->vx_des;
    f[26] = arb->vyaw_des;
    f[27] = (float)arb->mode;
    f[28] = (float)arb->emote;
    f[29] = (float)arbiter_source();
    /* vision / guard */
    {
        const int64_t age = arbiter_feature_age_ms();
        f[30] = (float)(age < 0 ? -1.0 : (double)age);
    }
    f[31] = 0.0f; /* n_obj is not exported by the snapshot; kept for layout */
    f[32] = 0.0f; /* theta_img likewise */
    f[33] = gaze_guard_danger() ? 1.0f : 0.0f;

    wr_u32(buf + 0, TLM_MAGIC);
    wr_u16(buf + 4, TLM_VER);
    wr_u16(buf + 6, (uint16_t)TLM_BYTES);
    wr_u32(buf + 8, s_frames);
    wr_u32(buf + 12, (uint32_t)(esp_timer_get_time() / 1000));
    for (i = 0; i < TLM_NFLOAT; i++) {
        wr_f32(buf + TLM_HDR + 4 * i, f[i]);
    }
}

static void tlm_task_main(void *arg)
{
    int sock = -1;
    struct sockaddr_in peer;
    socklen_t peer_len = sizeof(peer);
    uint8_t rx[64];
    uint8_t buf[TLM_BYTES];
    TickType_t last = xTaskGetTickCount();
    bool have_peer = false;

    (void)arg;
    memset(&peer, 0, sizeof(peer));

    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        ESP_LOGE(TAG, "socket() failed, telemetry disabled");
        s_task = NULL;
        vTaskDelete(NULL);
        return;
    }
    {
        int nb = 1;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &nb, sizeof(nb));
    }
    {
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons((uint16_t)arbiter_cfg()->udp_port);
        if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            ESP_LOGE(TAG, "bind(:%d) failed, telemetry disabled",
                     (int)arbiter_cfg()->udp_port);
            close(sock);
            s_task = NULL;
            vTaskDelete(NULL);
            return;
        }
    }
    ESP_LOGW(TAG, "UDP telemetry on :%d", (int)arbiter_cfg()->udp_port);

    for (;;) {
        const int div = (arbiter_cfg()->udp_div > 0) ? arbiter_cfg()->udp_div : 1;
        TickType_t ticks = pdMS_TO_TICKS(2 * div);
        int r;

        if (ticks == 0) {
            ticks = 1;
        }

        /* Any datagram from a client subscribes it (and re-subscribes). */
        r = recvfrom(sock, rx, sizeof(rx), MSG_DONTWAIT,
                     (struct sockaddr *)&peer, &peer_len);
        if (r >= 0) {
            have_peer = true;
            s_client_ip = (uint32_t)peer.sin_addr.s_addr;
            s_client_seen_us = esp_timer_get_time();
        }

        if (have_peer && (!s_enabled ||
                          (esp_timer_get_time() - s_client_seen_us) >
                              (int64_t)TLM_CLIENT_TTL_MS * 1000)) {
            have_peer = false;
        }

        if (have_peer && s_enabled) {
            put(arbiter_get(), buf);
            if (sendto(sock, buf, sizeof(buf), 0,
                       (struct sockaddr *)&peer, peer_len) > 0) {
                s_frames = s_frames + 1;
            }
        }

        vTaskDelayUntil(&last, ticks);
    }
}

void telemetry_stream_start(void)
{
    if (s_task != NULL) {
        return;
    }
    if (xTaskCreatePinnedToCore(tlm_task_main, "tlm_task", TLM_STACK, NULL,
                                TLM_PRIO, &s_task, TLM_CORE) != pdPASS) {
        s_task = NULL;
        ESP_LOGE(TAG, "failed to create telemetry task (stream disabled)");
        return;
    }
}

void telemetry_stream_stop(void)
{
    if (s_task != NULL) {
        vTaskDelete(s_task);
        s_task = NULL;
    }
}

bool telemetry_stream_running(void)
{
    return s_task != NULL;
}

bool telemetry_stream_enabled(void)
{
    return s_enabled;
}

void telemetry_stream_set_enabled(bool on)
{
    s_enabled = on;
}

uint32_t telemetry_stream_client_ip(void)
{
    return s_client_ip;
}

uint32_t telemetry_stream_frames(void)
{
    return s_frames;
}
