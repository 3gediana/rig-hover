/*
 * control_arbiter.c -- control source arbitration, safety shell, anti-windup.
 *
 * Data flow (see control_arbiter.h):
 *
 *   cfc_task / HTTP  --seqlock-->  raw output
 *                                     |
 *   arbiter_tick_2ms (robot_control)  |  safety shell  -> published output
 *                                     v
 *   stable_pos / stable_yaw integration + clamping  ->  robot.cc LQR loop
 *
 * The tick is the only writer of the published output, so arbiter_get() is a
 * plain pointer load. Nothing in the tick blocks, allocates or logs.
 */

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_log.h"

#include "robot.h"          /* stable_pos, stable_yaw, yaw_ctrl_time, q_head,
                               target_head_pos, ... */
#include "imu.h"            /* yaw, pitch, roll, dq                     */
#include "control_arbiter.h"
#include "gaze_guard.h"

/* Defined in robot.cc but not declared in robot.h. Keep these names in sync
 * with robot.cc; a rename shows up as a link error, not as wrong behaviour. */
extern float wheel_x;
extern int   robot_state;

static const char *TAG = "Arbiter";

/* ------------------------------------------------------------------ */
/* config                                                              */
/* ------------------------------------------------------------------ */

#define ARB_CFG_MAX_FIELDS 32

typedef struct {
    const char *name;
    uint8_t     is_int;
    uint8_t     idx;
} cfg_field_t;

/* Order matters: idx is the field slot used by the table below. Append new
 * fields at the end: the enum, the table and the two getter switches index the
 * same slot, so inserting in the middle silently remaps every field after it. */
enum {
    C_DT_MS = 0, C_POS_LEAD, C_YAW_LEAD, C_HEAD_MAX, C_VX_MAX, C_VYAW_MAX,
    C_STALE_MS, C_LOST_MS, C_HEAD_LPF, C_CMD_TIMEOUT, C_SRC, C_IMG_FRAME,
    C_GAZE_EN, C_GAZE_HOLD, C_CFC_HZ, C_FOV_H, C_DIST_K, C_DIST_LPF, C_DIST_TARGET,
    C_CLIP_SIGMA, C_UDP_EN, C_UDP_DIV, C_UDP_PORT,
    C_K_DYAW, C_VX_RATE, C_VYAW_RATE, C_VX_STOP_EPS, C_VYAW_STOP_EPS, C_COUNT
};

static const cfg_field_t s_fields[] = {
    { "dt_ms",          0, C_DT_MS },
    { "pos_lead_max",   0, C_POS_LEAD },
    { "yaw_lead_max",   0, C_YAW_LEAD },
    { "head_max",       0, C_HEAD_MAX },
    { "vx_max",         0, C_VX_MAX },
    { "vyaw_max",       0, C_VYAW_MAX },
    { "stale_ms",       0, C_STALE_MS },
    { "lost_ms",        0, C_LOST_MS },
    { "head_lpf",       0, C_HEAD_LPF },
    { "cmd_timeout_ms", 0, C_CMD_TIMEOUT },
    { "src",            1, C_SRC },
    { "img_frame",      1, C_IMG_FRAME },
    { "gaze_en",        1, C_GAZE_EN },
    { "gaze_hold_ms",   0, C_GAZE_HOLD },
    { "cfc_hz",         1, C_CFC_HZ },
    { "fov_h",          0, C_FOV_H },
    { "dist_k",         0, C_DIST_K },
    { "dist_lpf",       0, C_DIST_LPF },
    { "dist_target",    0, C_DIST_TARGET },
    { "clip_sigma",     0, C_CLIP_SIGMA },
    { "udp_en",         1, C_UDP_EN },
    { "udp_div",        1, C_UDP_DIV },
    { "udp_port",       1, C_UDP_PORT },
    { "k_dyaw",         0, C_K_DYAW },
    { "vx_rate_max",    0, C_VX_RATE },
    { "vyaw_rate_max",  0, C_VYAW_RATE },
    { "vx_stop_eps",    0, C_VX_STOP_EPS },
    { "vyaw_stop_eps",  0, C_VYAW_STOP_EPS },
};

#define N_FIELDS ((int)(sizeof(s_fields) / sizeof(s_fields[0])))

static arb_cfg_t s_cfg;
static bool      s_cfg_ready;

static void cfg_defaults(void)
{
    memset(&s_cfg, 0, sizeof(s_cfg));
    s_cfg.dt_ms          = 2.0f;  /* see the dt warning in control_arbiter.h */
    s_cfg.pos_lead_max   = 0.20f;
    s_cfg.yaw_lead_max   = 35.0f;
    s_cfg.head_max       = 45.0f;
    s_cfg.vx_max         = 0.30f;
    s_cfg.vyaw_max       = 90.0f;
    s_cfg.stale_ms       = 300.0f; /* 3 lost frames at 10 Hz, TECH-PLAN 5.4 */
    s_cfg.lost_ms        = 1000.0f;
    s_cfg.head_lpf       = 0.15f;
    s_cfg.cmd_timeout_ms = 500.0f;
    s_cfg.src            = ARB_SRC_MANUAL;
    s_cfg.img_frame      = 0;
    s_cfg.gaze_en        = 1;
    s_cfg.gaze_hold_ms   = 1000.0f;
    s_cfg.cfc_hz         = 50;
    s_cfg.fov_h          = 60.0f;
    s_cfg.dist_k         = 0.90f;
    s_cfg.dist_lpf       = 0.50f;
    s_cfg.dist_target    = 1.00f;
    s_cfg.clip_sigma     = 4.0f;
    s_cfg.udp_en         = 1;
    s_cfg.udp_div        = 1;
    s_cfg.udp_port       = 9001;
    /* Heading loop D gain. 0 keeps the pre-existing pure-P behaviour exactly. */
    s_cfg.k_dyaw         = 0.0f;
    /* Command slew limits: the model output is a 50 Hz staircase, the loop is
     * ~166 Hz. Without a limit a single bad frame is applied in full. Values
     * match the teacher's config.yaml (vyaw_rate_limit_dps2 = 400) so the two
     * control sources saturate the same way; 0 disables the limiter. */
    s_cfg.vx_rate_max    = 1.0f;
    s_cfg.vyaw_rate_max  = 400.0f;
    s_cfg.vx_stop_eps    = 0.005f;
    s_cfg.vyaw_stop_eps  = 0.5f;
    s_cfg_ready          = true;
}

/* ------------------------------------------------------------------ */
/* small helpers                                                       */
/* ------------------------------------------------------------------ */

static float clampf(float v, float lo, float hi)
{
    if (!(v == v)) { /* NaN */
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

static float clamp_lo_hi(float v, float lo, float hi)
{
    return (lo <= hi) ? clampf(v, lo, hi) : 0.0f;
}

static uint32_t now_ms32(void)
{
    return (uint32_t)(esp_timer_get_time() / 1000);
}

/* ------------------------------------------------------------------ */
/* command rate limiter                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    float prev;
    int   init;
} rate_lim_t;

static rate_lim_t s_vx_lim;
static rate_lim_t s_vyaw_lim;

/* Per-tick slew limit of one command channel towards `target`, plus a
 * snap-to-zero deadband.
 *
 * The snap predicate is |target| <= stop_eps, never `target <= lo + eps`.
 * With a single-sided [0, max] range the two look equivalent, but the moment
 * the velocity range is opened to negative values (reverse for wedge escape or
 * ramp creep) the second form classifies a full-speed reverse command as
 * "stop" and snaps it to zero. smartcar/code/lnn/action_shaper.py records this
 * exact bug on ActionRateLimiter.stop_speed; keying the predicate to zero
 * keeps it correct for either sign of the bound.
 *
 * `snap` is passed on ticks where the safety shell forces the channel to zero:
 * a protection must not be delayed behind the ramp.
 *
 * max_step <= 0 disables the limiter (pass-through), including the deadband. */
static float rate_limit(rate_lim_t *s, float target, float max_step,
                        float stop_eps, bool snap)
{
    float d;

    if (!(max_step > 0.0f)) {           /* limiter off */
        s->init = 1;
        s->prev = target;
        return target;
    }
    if (snap || !s->init) {
        s->init = 1;
        s->prev = snap ? 0.0f : target;
        return s->prev;
    }
    d = target - s->prev;
    if (d > max_step) {
        d = max_step;
    } else if (d < -max_step) {
        d = -max_step;
    }
    s->prev += d;
    if (fabsf(target) <= stop_eps) {
        s->prev = 0.0f;
    }
    return s->prev;
}

/* ------------------------------------------------------------------ */
/* measured loop period                                                */
/* ------------------------------------------------------------------ */

/* 0 until the first tick. The loop carries three vTaskDelay(2) plus UART
 * transactions, so it is ~6-10 ms and drifts with motor bus load. A fixed dt
 * integrates the reference at the wrong rate and the robot follows at a
 * fraction of the commanded speed. s_cfg.dt_ms is only the cold-start value. */
static float s_dt = 0.0f;

static float measure_dt(void)
{
    static int64_t last_us;
    const int64_t now_us = esp_timer_get_time();
    float dt = (last_us != 0) ? (float)(now_us - last_us) * 1e-6f
                              : s_cfg.dt_ms * 0.001f;
    last_us = now_us;
    return clampf(dt, 0.001f, 0.030f);
}

/* ------------------------------------------------------------------ */
/* raw output seqlock (producers: cfc_task, HTTP teacher)              */
/* ------------------------------------------------------------------ */

static portMUX_TYPE s_raw_mux = portMUX_INITIALIZER_UNLOCKED;

static struct {
    volatile uint32_t seq;
    ctrl_out_t        v;
} s_raw;

static void raw_write(const ctrl_out_t *in)
{
    taskENTER_CRITICAL(&s_raw_mux);
    s_raw.seq++;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    s_raw.v = *in;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    s_raw.seq++;
    taskEXIT_CRITICAL(&s_raw_mux);
}

/* Returns false when the writer was in the middle of an update. */
static bool raw_read(ctrl_out_t *out)
{
    uint32_t s1 = s_raw.seq;
    uint32_t s2;

    if (s1 & 1u) {
        return false;
    }
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    *out = s_raw.v;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    s2 = s_raw.seq;
    return (s1 == s2);
}

/* ------------------------------------------------------------------ */
/* published output (single writer: the tick)                          */
/* ------------------------------------------------------------------ */

static ctrl_out_t     s_pub[2];
static volatile int   s_pub_idx;
static volatile int   s_block;      /* ARB_BLK_* bitmask, debug only */
static volatile int   s_emote;      /* last emote accepted by the shell */

/* ------------------------------------------------------------------ */
/* feature / command freshness                                         */
/* ------------------------------------------------------------------ */

static volatile uint32_t s_feat_ms;      /* 0 = never received */
static volatile uint32_t s_teacher_ms;   /* 0 = never received */
static volatile uint32_t s_raw_ms;       /* 0 = never received */
static volatile int      s_upload_busy;

/* ------------------------------------------------------------------ */
/* init                                                                */
/* ------------------------------------------------------------------ */

void arbiter_init(void)
{
    if (!s_cfg_ready) {
        cfg_defaults();
    }
    gaze_guard_init();
}

static void ensure_init(void)
{
    if (!s_cfg_ready) {
        cfg_defaults();
    }
}

arb_cfg_t *arbiter_cfg(void)
{
    ensure_init();
    return &s_cfg;
}

/* ------------------------------------------------------------------ */
/* source selection                                                    */
/* ------------------------------------------------------------------ */

bool arbiter_active(void)
{
    return s_cfg.src != ARB_SRC_MANUAL;
}

int arbiter_source(void)
{
    return s_cfg.src;
}

int arbiter_set_source(int src)
{
    if (src < ARB_SRC_MANUAL || src > ARB_SRC_CFC) {
        return -1;
    }
    if (s_cfg.src != src) {
        /* Drop the stale command of the previous source so the new source
         * cannot inherit an old head angle / velocity on its first tick. */
        ctrl_out_t zero;
        memset(&zero, 0, sizeof(zero));
        raw_write(&zero);
        s_teacher_ms = 0;
    }
    s_cfg.src = src;
    ESP_LOGW(TAG, "source=%d", src);
    return 0;
}

const ctrl_out_t *arbiter_get(void)
{
    return &s_pub[s_pub_idx & 1];
}

int arbiter_block_reason(void)
{
    return s_block;
}

int arbiter_emote(void)
{
    return s_emote;
}

void arbiter_set_emote(int emote)
{
    if (emote < 0) {
        emote = 0;
    }
    if (emote > 7) {
        emote = 7;
    }
    s_emote = emote;
}

/* ------------------------------------------------------------------ */
/* producers                                                           */
/* ------------------------------------------------------------------ */

void arbiter_post_raw(const ctrl_out_t *raw)
{
    ctrl_out_t v;

    if (raw == NULL) {
        return;
    }
    ensure_init();
    v = *raw;
    /* Reject non-finite values at the boundary: a NaN must never reach the
     * integrators, even for one tick. */
    if (!isfinite(v.head_des) || !isfinite(v.vx_des) || !isfinite(v.vyaw_des)) {
        memset(&v, 0, sizeof(v));
        s_block |= ARB_BLK_NAN;
        ESP_LOGE(TAG, "non-finite ctrl output dropped");
    }
    if (v.mode < 0 || v.mode > 3) {
        v.mode = 0;
    }
    if (v.emote < 0 || v.emote > 7) {
        v.emote = 0;
    }
    s_raw_ms = now_ms32();
    if (s_raw_ms == 0) {
        s_raw_ms = 1;
    }
    raw_write(&v);
}

void arbiter_set_upload(bool busy)
{
    s_upload_busy = busy ? 1 : 0;
}

void arbiter_set_teacher(float head_des, float vx_des, float vyaw_des,
                         int mode, int emote)
{
    ctrl_out_t v;

    ensure_init();
    if (!isfinite(head_des) || !isfinite(vx_des) || !isfinite(vyaw_des)) {
        memset(&v, 0, sizeof(v));
        s_block |= ARB_BLK_NAN;
    } else {
        v.head_des = head_des;
        v.vx_des = vx_des;
        v.vyaw_des = vyaw_des;
        v.mode = (mode < 0 || mode > 3) ? 0 : mode;
        v.emote = (emote < 0 || emote > 7) ? 0 : emote;
    }
    s_teacher_ms = now_ms32();
    raw_write(&v);
}

void arbiter_note_feat(void)
{
    const uint32_t t = now_ms32();
    if (t != 0) {
        s_feat_ms = t;
    } else {
        s_feat_ms = 1;
    }
}

int64_t arbiter_feature_age_ms(void)
{
    const uint32_t last = s_feat_ms;
    if (last == 0) {
        return -1;
    }
    return (int64_t)(uint32_t)(now_ms32() - last);
}

bool arbiter_feature_fresh(void)
{
    const int64_t age = arbiter_feature_age_ms();
    return (age >= 0) && ((float)age <= s_cfg.stale_ms);
}

/* ------------------------------------------------------------------ */
/* safety shell                                                        */
/* ------------------------------------------------------------------ */

/* Fill `o` with the sanitised output for this tick. Pure computation, no
 * logging in the steady state (only on the edge that starts blocking). */
static void safety_shell(ctrl_out_t *o)
{
    ctrl_out_t raw;
    int block = 0;
    const int64_t age = arbiter_feature_age_ms();

    memset(&raw, 0, sizeof(raw));
    (void)raw_read(&raw);   /* keep the previous published value on a torn read */

    *o = raw;

    /* 1. table edge: highest priority, cannot be vetoed. */
    if (s_cfg.gaze_en && gaze_guard_danger()) {
        o->vx_des = 0.0f;
        o->vyaw_des = 0.0f;
        block |= ARB_BLK_GAZE;
    }

    /* 2. fallen / lifted / torque off. */
    if (robot_state == 0) {
        memset(o, 0, sizeof(*o));
        block |= ARB_BLK_ROBOT_OFF;
    }

    /* 3. non-finite guard (also catches garbage from a corrupted model). */
    if (!isfinite(o->head_des) || !isfinite(o->vx_des) || !isfinite(o->vyaw_des)) {
        memset(o, 0, sizeof(*o));
        block |= ARB_BLK_NAN;
    }

    /* 4. vision freshness ladder (TECH-PLAN 5.4 / 7.5). */
    if (age < 0 || (float)age > s_cfg.lost_ms) {
        o->vx_des = 0.0f;
        o->vyaw_des = 0.0f;
        o->head_des = 0.0f;      /* back to centre, slowly, via the LPF below */
        o->mode = 0;
        block |= ARB_BLK_VISION;
    } else if ((float)age > s_cfg.stale_ms) {
        o->vx_des = 0.0f;
        o->vyaw_des = 0.0f;      /* hold the head where it is */
        block |= ARB_BLK_VISION;
    }

    /* 5. teacher / cfc channel watchdog: the producer may die mid-motion. */
    if (s_cfg.src == ARB_SRC_TEACHER) {
        const uint32_t t = s_teacher_ms;
        if (t == 0 || (uint32_t)(now_ms32() - t) > (uint32_t)s_cfg.cmd_timeout_ms) {
            o->vx_des = 0.0f;
            o->vyaw_des = 0.0f;
            o->head_des = 0.0f;
            block |= ARB_BLK_NOCMD;
        }
    } else if (s_cfg.src == ARB_SRC_CFC) {
        const uint32_t t = s_raw_ms;
        if (t == 0 || (uint32_t)(now_ms32() - t) > (uint32_t)s_cfg.cmd_timeout_ms) {
            o->vx_des = 0.0f;
            o->vyaw_des = 0.0f;
            block |= ARB_BLK_NOCMD;
        }
    }

    /* 5b. flash write in flight: never translate on a stalled loop. */
    if (s_upload_busy) {
        o->vx_des = 0.0f;
        o->vyaw_des = 0.0f;
        block |= ARB_BLK_UPLOAD;
    }

    /* 6. hard clamps. head_max can never exceed the servo travel. */
    {
        float hm = s_cfg.head_max;
        if (hm > 45.0f) {
            hm = 45.0f;
        }
        o->head_des = clampf(o->head_des, -hm, hm);
    }
    o->vx_des   = clamp_lo_hi(o->vx_des, -fabsf(s_cfg.vx_max), fabsf(s_cfg.vx_max));
    o->vyaw_des = clamp_lo_hi(o->vyaw_des, -fabsf(s_cfg.vyaw_max), fabsf(s_cfg.vyaw_max));

    /* 7. command slew limit: the controller's output is a 50 Hz staircase on a
     * ~166 Hz loop. Applied last so the limiter can never widen a bound above,
     * and snapped (not ramped) on every tick where the shell above is
     * protecting, so a stop is immediate. */
    {
        const bool snap = (block != 0);
        o->vx_des = rate_limit(&s_vx_lim, o->vx_des, s_cfg.vx_rate_max * s_dt,
                               s_cfg.vx_stop_eps, snap);
        o->vyaw_des = rate_limit(&s_vyaw_lim, o->vyaw_des,
                                 s_cfg.vyaw_rate_max * s_dt, s_cfg.vyaw_stop_eps,
                                 snap);
    }

    if (o->mode < 0 || o->mode > 3) {
        o->mode = 0;
    }
    if (o->emote < 0 || o->emote > 7) {
        o->emote = 0;
    }
    s_emote = o->emote;

    s_block = block;
}

/* ------------------------------------------------------------------ */
/* the 2 ms hook                                                       */
/* ------------------------------------------------------------------ */

void arbiter_tick_2ms(void)
{
    ctrl_out_t next;
    int idx;

    ensure_init();

    /* Measure the real loop period once per tick: the safety shell's slew
     * limiters and the reference integration below must agree on dt. */
    s_dt = measure_dt();

    /* Publish the sanitised output first so arbiter_get() and the integration
     * below see exactly the same value. */
    safety_shell(&next);
    {
        /* Low pass the head command: the CfC produces a new head angle every
         * 20 ms but the servo is written every 2 ms; without this the servo
         * sees a staircase and buzzes. Applied per tick. */
        static float head_filt;
        static bool  head_init;
        const float a = clampf(s_cfg.head_lpf, 0.0f, 1.0f);
        if (!head_init || a >= 1.0f) {
            head_filt = next.head_des;
            head_init = true;
        } else {
            head_filt += a * (next.head_des - head_filt);
        }
        next.head_des = clampf(head_filt, -45.0f, 45.0f);
    }

    idx = (s_pub_idx & 1) ^ 1;
    s_pub[idx] = next;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    s_pub_idx = idx;

    if (!arbiter_active()) {
        return;
    }

    {
        const ctrl_out_t *c = arbiter_get();
        /* Period measured at the top of this tick (see measure_dt). */
        const float dt = s_dt;
        const float yref = yaw - q_head;

        /* Integrate only when the state the clamp is relative to is usable:
         * a NaN wheel_x would otherwise make the clamp write 0 and command a
         * full-speed run to the origin. */
        if (arbiter_feature_fresh() && isfinite(wheel_x) && isfinite(yref) &&
            isfinite(dt)) {
            const float lo = wheel_x - s_cfg.pos_lead_max;
            const float hi = wheel_x + s_cfg.pos_lead_max;

            stable_pos += c->vx_des * dt;
            stable_pos = clamp_lo_hi(stable_pos, lo, hi);

            stable_yaw += c->vyaw_des * dt;
            stable_yaw = clamp_lo_hi(stable_yaw, yref - s_cfg.yaw_lead_max,
                                     yref + s_cfg.yaw_lead_max);
        }
        /* Refresh unconditionally: robot.cc re-freezes stable_yaw to the
         * current heading once now - yaw_ctrl_time > 3000 ms, which would
         * silently cancel our heading target after 3 s. */
        yaw_ctrl_time = (float)(esp_timer_get_time() / 1000);
        target_head_pos = clampf(c->head_des, -45.0f, 45.0f);
    }
}

/* ------------------------------------------------------------------ */
/* config plumbing for /cfg                                            */
/* ------------------------------------------------------------------ */

int arbiter_cfg_count(void)
{
    return N_FIELDS;
}

const char *arbiter_cfg_name(int i)
{
    if (i < 0 || i >= N_FIELDS) {
        return NULL;
    }
    return s_fields[i].name;
}

int arbiter_cfg_is_int(int i)
{
    if (i < 0 || i >= N_FIELDS) {
        return 0;
    }
    return s_fields[i].is_int;
}

float arbiter_cfg_get_float(int i)
{
    switch (i) {
    case C_DT_MS:        return s_cfg.dt_ms;
    case C_POS_LEAD:     return s_cfg.pos_lead_max;
    case C_YAW_LEAD:     return s_cfg.yaw_lead_max;
    case C_HEAD_MAX:     return s_cfg.head_max;
    case C_VX_MAX:       return s_cfg.vx_max;
    case C_VYAW_MAX:     return s_cfg.vyaw_max;
    case C_STALE_MS:     return s_cfg.stale_ms;
    case C_LOST_MS:      return s_cfg.lost_ms;
    case C_HEAD_LPF:     return s_cfg.head_lpf;
    case C_CMD_TIMEOUT:  return s_cfg.cmd_timeout_ms;
    case C_GAZE_HOLD:    return s_cfg.gaze_hold_ms;
    case C_FOV_H:        return s_cfg.fov_h;
    case C_DIST_K:       return s_cfg.dist_k;
    case C_DIST_LPF:     return s_cfg.dist_lpf;
    case C_DIST_TARGET:  return s_cfg.dist_target;
    case C_CLIP_SIGMA:   return s_cfg.clip_sigma;
    case C_K_DYAW:       return s_cfg.k_dyaw;
    case C_VX_RATE:      return s_cfg.vx_rate_max;
    case C_VYAW_RATE:    return s_cfg.vyaw_rate_max;
    case C_VX_STOP_EPS:  return s_cfg.vx_stop_eps;
    case C_VYAW_STOP_EPS:return s_cfg.vyaw_stop_eps;
    default:             break;
    }
    return (float)arbiter_cfg_get_int(i);
}

int arbiter_cfg_get_int(int i)
{
    switch (i) {
    case C_SRC:       return s_cfg.src;
    case C_IMG_FRAME: return s_cfg.img_frame;
    case C_GAZE_EN:   return s_cfg.gaze_en;
    case C_CFC_HZ:    return s_cfg.cfc_hz;
    case C_UDP_EN:    return s_cfg.udp_en;
    case C_UDP_DIV:   return s_cfg.udp_div;
    case C_UDP_PORT:  return s_cfg.udp_port;
    default:          break;
    }
    return 0;
}

int arbiter_cfg_set(const char *key, float value)
{
    int i;

    if (key == NULL) {
        return -1;
    }
    for (i = 0; i < N_FIELDS; i++) {
        if (strcmp(s_fields[i].name, key) == 0) {
            break;
        }
    }
    if (i >= N_FIELDS) {
        return -1;
    }

    switch (i) {
    case C_DT_MS:       s_cfg.dt_ms = clampf(value, 0.1f, 20.0f); break;
    case C_POS_LEAD:    s_cfg.pos_lead_max = clampf(value, 0.0f, 0.45f); break;
    case C_YAW_LEAD:    s_cfg.yaw_lead_max = clampf(value, 0.0f, 180.0f); break;
    case C_HEAD_MAX:    s_cfg.head_max = clampf(value, 0.0f, 45.0f); break;
    case C_VX_MAX:      s_cfg.vx_max = clampf(value, 0.0f, 1.0f); break;
    case C_VYAW_MAX:    s_cfg.vyaw_max = clampf(value, 0.0f, 90.0f); break;
    case C_STALE_MS:    s_cfg.stale_ms = clampf(value, 20.0f, 5000.0f); break;
    case C_LOST_MS:     s_cfg.lost_ms = clampf(value, 50.0f, 10000.0f); break;
    case C_HEAD_LPF:    s_cfg.head_lpf = clampf(value, 0.0f, 1.0f); break;
    case C_CMD_TIMEOUT: s_cfg.cmd_timeout_ms = clampf(value, 50.0f, 5000.0f); break;
    case C_GAZE_HOLD:   s_cfg.gaze_hold_ms = clampf(value, 100.0f, 10000.0f); break;
    case C_FOV_H:       s_cfg.fov_h = clampf(value, 10.0f, 180.0f); break;
    case C_DIST_K:      s_cfg.dist_k = clampf(value, 0.05f, 10.0f); break;
    case C_DIST_LPF:    s_cfg.dist_lpf = clampf(value, 0.01f, 10.0f); break;
    case C_DIST_TARGET: s_cfg.dist_target = clampf(value, 0.2f, 5.0f); break;
    case C_CLIP_SIGMA:  s_cfg.clip_sigma = clampf(value, 1.0f, 16.0f); break;
    /* Symmetric range on purpose: which sign of k_dyaw damps depends on the
     * robot's yaw polarity, which has never been measured. See robot.cc. */
    case C_K_DYAW:      s_cfg.k_dyaw = clampf(value, -8.0f, 8.0f); break;
    case C_VX_RATE:     s_cfg.vx_rate_max = clampf(value, 0.0f, 20.0f); break;
    case C_VYAW_RATE:   s_cfg.vyaw_rate_max = clampf(value, 0.0f, 5000.0f); break;
    case C_VX_STOP_EPS: s_cfg.vx_stop_eps = clampf(value, 0.0f, 0.1f); break;
    case C_VYAW_STOP_EPS: s_cfg.vyaw_stop_eps = clampf(value, 0.0f, 20.0f); break;
    case C_SRC:         return arbiter_set_source((int)value);
    case C_IMG_FRAME:   s_cfg.img_frame = (value >= 0.5f) ? 1 : 0; break;
    case C_GAZE_EN:     s_cfg.gaze_en = (value >= 0.5f) ? 1 : 0; break;
    case C_CFC_HZ:      s_cfg.cfc_hz = (int)clampf(value, 5.0f, 1000.0f); break;
    case C_UDP_EN:      s_cfg.udp_en = (value >= 0.5f) ? 1 : 0; break;
    case C_UDP_DIV:     s_cfg.udp_div = (int)clampf(value, 1.0f, 100.0f); break;
    case C_UDP_PORT:    s_cfg.udp_port = (int)clampf(value, 1024.0f, 65535.0f); break;
    default:
        return -1;
    }
    return 0;
}
