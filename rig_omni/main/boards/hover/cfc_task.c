/*
 * cfc_task.c -- /feat ingestion, observation assembly, on-board CfC inference.
 *
 * Task placement (must not change):
 *   robot_control : core 0, priority 12, ~6 ms period  <- never delayed
 *   cfc_task      : core 1, priority  5, 50 Hz default
 * The inference task only reads snapshots and writes through a seqlock, so the
 * 500 Hz loop never waits on it.
 */

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "robot.h"
#include "imu.h"

#include "cfc_model.h"
#include "cfc_task.h"
#include "control_arbiter.h"
#include "gaze_guard.h"

static const char *TAG = "CfcTask";

/* Defined in robot.cc, not declared in robot.h. */
extern float wheel_x;
extern int   robot_state;

/* ------------------------------------------------------------------ */
/* constants                                                           */
/* ------------------------------------------------------------------ */

#define FEAT_MAGIC     0xFE01u
#define FEAT_MAX_OBJ   8
#define FEAT_MAX_DIM   64
#define OBS_DIM        106    /* TECH-PLAN 4.3: 64+6+8+22+4+2 */
#define OBJ_STRIDE_BIG   16   /* i16*4 + u8*4 + i16*2, the struct as written */
#define OBJ_STRIDE_SMALL 12   /* i16*4 + u8*4, when the sender omits the wave */

#define CFC_TASK_PRIO  5      /* strictly below robot_control (12) */
#define CFC_TASK_CORE  1
/* 12 KB: the int8 step keeps its two int8 activation vectors plus the float z
 * buffer on the stack, and cfc_model_forward keeps the normalised observation
 * and the projection output. */
#define CFC_TASK_STACK 12288

/* ------------------------------------------------------------------ */
/* vision snapshot (writer: HTTP task, reader: cfc_task)               */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t pc_t_ms;
    uint32_t rx_ms;      /* robot monotonic clock at reception, 0 = none */
    uint16_t n_obj;
    uint16_t n_feat;
    float    feat[FEAT_MAX_DIM];

    /* best target, already normalised */
    int      seen;
    float    theta_img;  /* deg, from cx and cfg.fov_h */
    float    person_w;
    float    person_h;
    float    person_conf;
    int      hand_up_l;
    int      hand_up_r;
    float    wave_sin;
    float    wave_cos;
    float    wave_amp;
    float    gesture_conf;
} feat_snap_t;

static portMUX_TYPE s_feat_mux = portMUX_INITIALIZER_UNLOCKED;

static struct {
    volatile uint32_t seq;
    feat_snap_t       v;
} s_feat_src;

static volatile uint32_t s_feat_rx_ms;   /* last good frame, robot clock */
static volatile uint32_t s_feat_dt_ms;

static void feat_write(const feat_snap_t *in)
{
    taskENTER_CRITICAL(&s_feat_mux);
    s_feat_src.seq++;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    s_feat_src.v = *in;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    s_feat_src.seq++;
    taskEXIT_CRITICAL(&s_feat_mux);
}

/* Copy the newest snapshot. Returns false when the writer was mid-update. */
static bool feat_read(feat_snap_t *out)
{
    uint32_t s1 = s_feat_src.seq;
    uint32_t s2;

    if (s1 & 1u) {
        return false;
    }
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    *out = s_feat_src.v;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    s2 = s_feat_src.seq;
    return (s1 == s2);
}

/* ------------------------------------------------------------------ */
/* little endian readers                                               */
/* ------------------------------------------------------------------ */

static uint16_t rd_u16(const uint8_t *p)
{
    uint16_t v;
    memcpy(&v, p, 2);
    return v;
}

static int16_t rd_i16(const uint8_t *p)
{
    int16_t v;
    memcpy(&v, p, 2);
    return v;
}

static uint32_t rd_u32(const uint8_t *p)
{
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static float rd_f32(const uint8_t *p)
{
    float v;
    memcpy(&v, p, 4);
    return v;
}

static float clampf(float v, float lo, float hi)
{
    if (!(v == v)) {
        return 0.0f;
    }
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

static uint32_t now_ms32(void)
{
    return (uint32_t)(esp_timer_get_time() / 1000);
}

/* ------------------------------------------------------------------ */
/* /feat parsing                                                       */
/* ------------------------------------------------------------------ */

static void reply_fail(cfc_feat_reply_t *r, const char *why)
{
    if (r != NULL) {
        r->ok = 0;
        r->err = why;
    }
}

void cfc_feat_ingest(const uint8_t *buf, size_t len, cfc_feat_reply_t *reply)
{
    feat_snap_t snap;
    uint16_t magic, n_obj, n_feat;
    uint32_t pc_t;
    size_t need;
    size_t stride;
    int best;
    float best_score;
    int i;

    if (reply != NULL) {
        memset(reply, 0, sizeof(*reply));
        reply->ok = 0;
        reply->err = "no data";
        reply->rx_bytes = (int)len;
        reply->cfc_us = -1;
    }
    if (buf == NULL || len < 12) {
        reply_fail(reply, "too short");
        return;
    }

    magic = rd_u16(buf);
    if (magic != FEAT_MAGIC) {
        reply_fail(reply, "bad magic");
        return;
    }
    n_obj = rd_u16(buf + 2);
    pc_t = rd_u32(buf + 4);
    n_feat = rd_u16(buf + 8);
    /* buf[10..11] reserved */

    if (n_obj > FEAT_MAX_OBJ) {
        reply_fail(reply, "n_obj>8");
        return;
    }
    if (n_feat > FEAT_MAX_DIM) {
        reply_fail(reply, "n_feat>64");
        return;
    }

    /* TECH-PLAN 5.2 writes "obj 12 bytes" in the comment but the struct it
     * prints is 16 bytes (wave_sin/wave_cos included). Accept both strides so
     * neither a Python implementation of the comment nor of the struct is
     * rejected; the 16-byte form wins when both fit. */
    stride = OBJ_STRIDE_BIG;
    need = 12 + (size_t)OBJ_STRIDE_BIG * n_obj + 4 + 2u * n_feat;
    if (len < need) {
        stride = OBJ_STRIDE_SMALL;
        need = 12 + (size_t)OBJ_STRIDE_SMALL * n_obj + 4 + 2u * n_feat;
        if (len < need) {
            reply_fail(reply, "length mismatch");
            return;
        }
    }

    memset(&snap, 0, sizeof(snap));
    snap.pc_t_ms = pc_t;
    snap.rx_ms = now_ms32();
    snap.n_obj = n_obj;
    snap.n_feat = n_feat;

    /* Quantised features: int16/32767 with an explicit dequant scale. */
    {
        const size_t foff = 12 + stride * n_obj + 4;
        float k = rd_f32(buf + 12 + stride * n_obj);
        if (!(k > 0.0f) || !isfinite(k)) {
            k = 1.0f / 32767.0f;
        }
        for (i = 0; i < n_feat; i++) {
            snap.feat[i] = clampf((float)rd_i16(buf + foff + 2u * i) * k, -1.0f, 1.0f);
        }
    }

    /* Pick the nearest / most confident target: largest confidence-weighted
     * box area. Cheap, good enough for the single-person v1 scenario. */
    best = -1;
    best_score = 0.0f;
    for (i = 0; i < n_obj; i++) {
        const uint8_t *o = buf + 12 + stride * i;
        const float conf = (float)o[8] / 255.0f;
        const float w = (float)rd_i16(o + 4) / 1000.0f;
        const float h = (float)rd_i16(o + 6) / 1000.0f;
        const float score = conf * (w * h);
        if (conf > 0.05f && score > best_score) {
            best_score = score;
            best = i;
        }
    }

    if (best >= 0) {
        const uint8_t *o = buf + 12 + stride * best;
        const float cx = (float)rd_i16(o + 0) / 1000.0f;
        const float fov = arbiter_cfg()->fov_h;
        snap.seen = 1;
        snap.theta_img = (cx - 0.5f) * fov;
        snap.person_w = clampf((float)rd_i16(o + 4) / 1000.0f, 0.0f, 2.0f);
        snap.person_h = clampf((float)rd_i16(o + 6) / 1000.0f, 0.0f, 2.0f);
        snap.person_conf = (float)o[8] / 255.0f;
        snap.hand_up_l = (o[10] & 0x01u) ? 1 : 0;
        snap.hand_up_r = (o[10] & 0x02u) ? 1 : 0;
        if (stride == OBJ_STRIDE_BIG) {
            snap.wave_sin = clampf((float)rd_i16(o + 12) / 1000.0f, -1.0f, 1.0f);
            snap.wave_cos = clampf((float)rd_i16(o + 14) / 1000.0f, -1.0f, 1.0f);
        }
        snap.wave_amp = clampf(sqrtf(snap.wave_sin * snap.wave_sin +
                                     snap.wave_cos * snap.wave_cos), 0.0f, 1.0f);
        snap.gesture_conf = snap.person_conf;
    }

    {
        const uint32_t now = snap.rx_ms;
        const uint32_t prev = s_feat_rx_ms;
        s_feat_dt_ms = (prev != 0) ? (uint32_t)(now - prev) : 0;
        s_feat_rx_ms = now;
    }
    feat_write(&snap);
    arbiter_note_feat();

    if (reply != NULL) {
        reply->ok = 1;
        reply->err = "ok";
        reply->n_obj = n_obj;
        reply->n_feat = n_feat;
        reply->dt_ms = (int)s_feat_dt_ms;
        reply->cfc_us = cfc_task_last_us();
        reply->cfc_hz = cfc_task_last_hz();
        /* age_ms compares two unsynchronised clocks (PC millis vs robot
         * uptime). It is only meaningful once the PC syncs to /status "t";
         * anything implausible is reported as 0 instead of a huge number. */
        {
            const int64_t d = (int64_t)now_ms32() - (int64_t)pc_t;
            reply->age_ms = (d >= 0 && d < 60000) ? (int)d : 0;
        }
    }
}

/* ------------------------------------------------------------------ */
/* observation assembly                                                */
/* ------------------------------------------------------------------ */

static float s_prev_head;
static float s_prev_vyaw;
static float s_prev_vx;
static int   s_prev_mode;

static void build_obs(const cfc_model_t *m, const feat_snap_t *f, float *obs)
{
    const arb_cfg_t *cfg = arbiter_cfg();
    static float s_dist;                 /* low-passed distance estimate, m */
    const float t_s = (float)(esp_timer_get_time() / 1000000.0);
    float dist;
    int k = 0;

    /* Always clear the full buffer: the model may declare an input_dim larger
     * than OBS_DIM, and forward() reads exactly input_dim floats. */
    memset(obs, 0, sizeof(float) * CFC_MAX_INPUT_DIM);

    /* 0..63 vision */
    for (k = 0; k < FEAT_MAX_DIM; k++) {
        obs[k] = f->feat[k];
    }

    /* 64..69 target */
    obs[64] = f->theta_img;
    obs[65] = f->person_w;
    obs[66] = f->person_h;
    /* Monocular distance (TECH-PLAN 7.2b): must be low passed, otherwise a
     * person crouching reads as "walked away" and the robot lunges forward. */
    {
        const float tau = clampf(cfg->dist_lpf, 0.02f, 10.0f);
        const float dt = 1.0f / (float)((cfg->cfc_hz > 0) ? cfg->cfc_hz : 50);
        const float alpha = clampf(dt / (tau + dt), 0.0f, 1.0f);
        const float raw = (f->seen && f->person_h > 0.02f)
                              ? (cfg->dist_k / f->person_h)
                              : s_dist;
        if (s_dist <= 0.0f) {
            s_dist = cfg->dist_target;
        }
        s_dist += alpha * (raw - s_dist);
        s_dist = clampf(s_dist, 0.05f, 20.0f);
        dist = s_dist;
    }
    obs[67] = dist;
    obs[68] = f->person_conf;
    obs[69] = (float)f->seen;

    /* 70..77 gesture primitives */
    obs[70] = (float)f->hand_up_l;
    obs[71] = (float)f->hand_up_r;
    obs[72] = f->wave_sin;
    obs[73] = f->wave_cos;
    obs[74] = f->wave_amp;
    obs[75] = f->gesture_conf;
    obs[76] = 0.0f;
    obs[77] = 0.0f;

    /* 78..99 proprioception (22) */
    obs[78] = roll;
    obs[79] = pitch;
    obs[80] = pit_gyro;
    obs[81] = dq;
    obs[82] = yaw;
    obs[83] = q_head;
    obs[84] = target_head_pos;
    obs[85] = q_head - target_head_pos;
    obs[86] = wheel1_vel;
    obs[87] = wheel2_vel;
    obs[88] = wheel1_vel - wheel2_vel;
    obs[89] = wheel_x;
    obs[90] = stable_pos;
    obs[91] = wheel_x - stable_pos;
    obs[92] = yaw - q_head;
    obs[93] = (yaw - q_head) - stable_yaw;
    obs[94] = (float)robot_state;
    obs[95] = vx;
    obs[96] = vyaw;
    obs[97] = (float)control_mode;
    {
        const int64_t age = arbiter_feature_age_ms();
        obs[98] = (float)((age < 0) ? -1.0 : (double)age / 1000.0);
    }
    obs[99] = gaze_guard_danger() ? 1.0f : 0.0f;

    /* 100..103 feedback */
    obs[100] = s_prev_head / 45.0f;
    obs[101] = s_prev_vyaw / 90.0f;
    obs[102] = (float)s_prev_mode / 3.0f;
    obs[103] = s_prev_vx / 0.30f;

    /* 104..105 low frequency clock (10 s period) */
    {
        const float ph = 6.2831853f * t_s / 10.0f;
        obs[104] = sinf(ph);
        obs[105] = cosf(ph);
    }

    /* Adapt to the model: truncate or zero-pad. */
    if (m != NULL && m->input_dim < OBS_DIM) {
        for (k = m->input_dim; k < OBS_DIM; k++) {
            obs[k] = 0.0f;
        }
    }
}

int cfc_task_obs_dim(void)
{
    return OBS_DIM;
}

/* ------------------------------------------------------------------ */
/* inference task                                                      */
/* ------------------------------------------------------------------ */

static TaskHandle_t s_task;
static volatile int64_t s_last_us = -1;
static volatile int  s_last_hz;
static volatile int  s_frames;
static volatile int  s_dropped;

static int argmax(const float *p, int n)
{
    int i, best = 0;
    for (i = 1; i < n; i++) {
        if (p[i] > p[best]) {
            best = i;
        }
    }
    return best;
}

static void publish(const cfc_model_t *m, const float *out)
{
    const arb_cfg_t *cfg = arbiter_cfg();
    ctrl_out_t o;

    memset(&o, 0, sizeof(o));
    o.head_des = clampf(out[0 * CFC_MAX_HEAD_DIM], -1.0f, 1.0f) * cfg->head_max;
    o.vyaw_des = clampf(out[1 * CFC_MAX_HEAD_DIM], -1.0f, 1.0f) * cfg->vyaw_max;
    if (m->n_heads > 2) {
        o.mode = argmax(&out[2 * CFC_MAX_HEAD_DIM], m->head_dim[2]);
    }
    if (m->n_heads > 3) {
        int e = argmax(&out[3 * CFC_MAX_HEAD_DIM], m->head_dim[3]);
        if (e < 0 || e > 7) {
            e = 0;
        }
        o.emote = e;
    }
    /* vx is the optional 5th head; with a 4-head model the translation
     * channel stays at zero on purpose (TECH-PLAN 11.4: n_heads 4 or 5). */
    if (m->n_heads > 4) {
        o.vx_des = clampf(out[4 * CFC_MAX_HEAD_DIM], -1.0f, 1.0f) * cfg->vx_max;
    }

    s_prev_head = o.head_des;
    s_prev_vyaw = o.vyaw_des;
    s_prev_vx = o.vx_des;
    s_prev_mode = o.mode;

    arbiter_post_raw(&o);
}

/* Fraction of observation entries that had to be clipped by the domain guard.
 * More than 50 % means the model is being fed data it was never trained on. */
static float clip_ratio(const cfc_model_t *m, const float *obs)
{
    int i;
    int clipped = 0;
    if (m->input_dim <= 0) {
        return 0.0f;
    }
    for (i = 0; i < m->input_dim; i++) {
        const float mu = m->input_mean[i];
        const float sd = m->input_std[i];
        const float v = obs[i];
        if (!(v == v) || !isfinite(v)) {
            clipped++;
            continue;
        }
        if (v < mu - m->clip_sigma * sd || v > mu + m->clip_sigma * sd) {
            clipped++;
        }
    }
    return (float)clipped / (float)m->input_dim;
}

static void cfc_task_main(void *arg)
{
    float h[CFC_MAX_HIDDEN_DIM];
    float out[CFC_MAX_HEADS * CFC_MAX_HEAD_DIM];
    float obs[CFC_MAX_INPUT_DIM];
    feat_snap_t feat;
    int h_dim = -1;
    TickType_t last = xTaskGetTickCount();
    int64_t win_start = esp_timer_get_time();
    int win_frames = 0;

    (void)arg;
    memset(h, 0, sizeof(h));
    memset(&feat, 0, sizeof(feat));

    for (;;) {
        const cfc_model_t *m = NULL;
        const int hz = (arbiter_cfg()->cfc_hz > 0) ? arbiter_cfg()->cfc_hz : 50;
        const TickType_t period = pdMS_TO_TICKS(1000 / (hz > 200 ? 200 : hz));

        if (arbiter_source() == ARB_SRC_CFC) {
            m = cfc_model_acquire();
        }
        if (m != NULL) {
            if (h_dim != m->hidden_dim) {
                cfc_model_reset_state(m, h);
                h_dim = m->hidden_dim;
            }
            {
                /* Keep the last complete snapshot: a torn read must never
                 * feed half-updated vision data into the model. */
                feat_snap_t tmp;
                if (feat_read(&tmp)) {
                    feat = tmp;
                }
            }
            build_obs(m, &feat, obs);

            if (clip_ratio(m, obs) > 0.5f) {
                s_dropped++;
                /* Keep the last output: feeding garbage would teach nothing
                 * and moving on garbage is worse than freezing the head. */
            } else {
                const int64_t t0 = esp_timer_get_time();
                const int n = cfc_model_forward(m, obs, h, out,
                                                CFC_MAX_HEADS * CFC_MAX_HEAD_DIM);
                const int64_t t1 = esp_timer_get_time();
                s_last_us = (t1 - t0);
                if (n > 0) {
                    publish(m, out);
                    s_frames++;
                    win_frames++;
                }
            }
            cfc_model_release();

            {
                const int64_t now = esp_timer_get_time();
                if (now - win_start >= 1000000) {
                    s_last_hz = win_frames;
                    win_frames = 0;
                    win_start = now;
                }
            }
        } else {
            h_dim = -1;
            win_frames = 0;
            win_start = esp_timer_get_time();
            s_last_hz = 0;
        }

        vTaskDelayUntil(&last, period);
    }
}

void cfc_task_start(void)
{
    if (s_task != NULL) {
        return;
    }
    if (xTaskCreatePinnedToCore(cfc_task_main, "cfc_task", CFC_TASK_STACK, NULL,
                                CFC_TASK_PRIO, &s_task, CFC_TASK_CORE) != pdPASS) {
        s_task = NULL;
        ESP_LOGE(TAG, "failed to create cfc_task (inference disabled)");
        return;
    }
    ESP_LOGW(TAG, "cfc_task started on core %d prio %d", CFC_TASK_CORE, CFC_TASK_PRIO);
}

bool cfc_task_running(void)
{
    return s_task != NULL;
}

int64_t cfc_task_last_us(void)
{
    return s_last_us;
}

int cfc_task_last_hz(void)
{
    return s_last_hz;
}

int cfc_task_frames(void)
{
    return s_frames;
}

int cfc_task_dropped(void)
{
    return s_dropped;
}
