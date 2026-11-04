#ifndef CONTROL_ARBITER_H
#define CONTROL_ARBITER_H
/*
 * control_arbiter.h -- control source arbitration, safety shell and
 * anti-windup for the RIG-Hover balance loop.
 *
 * The 500 Hz robot loop (robot.cc robot_control) calls arbiter_tick_2ms().
 * Everything the tick touches is either a plain float global defined in
 * robot.cc or a snapshot this module owns, so the tick never blocks: no
 * mutexes, no queues, no heap, no logging in the steady state.
 *
 * Producers (all on other tasks / cores):
 *   - cfc_task        -> arbiter_post_raw()       (CfC inference, ~50 Hz)
 *   - HTTP /mode      -> arbiter_set_teacher()    (PC side teacher rules)
 *   - HTTP /feat      -> arbiter_note_feat()      (vision frame arrival)
 *   - HTTP /gaze      -> gaze_guard_set()         (table-edge danger flag)
 *
 * The tick is the single writer of the published output, so arbiter_get()
 * needs no lock (double buffer + atomic index flip).
 */

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Control source identifiers. */
enum {
    ARB_SRC_MANUAL = 0, /* arbiter inactive: original /cmd behaviour      */
    ARB_SRC_TEACHER = 1,/* PC pushes actions through POST /mode           */
    ARB_SRC_CFC = 2     /* on-board CfC model (slot chosen by /mode)      */
};

/* What robot.cc consumes. Filled by the safety shell, never by a producer. */
typedef struct {
    float head_des;  /* deg, clamped to [-head_max, head_max], head_max<=45 */
    float vx_des;    /* m/s, clamped to [-vx_max, vx_max]                   */
    float vyaw_des;  /* deg/s, clamped to [-vyaw_max, vyaw_max]             */
    int   mode;      /* 0=IDLE 1=TRACK 2=ACK 3=REFRACTORY                  */
    int   emote;     /* 0..7, drives the LCD                                */
} ctrl_out_t;

/*
 * Runtime tunables, all writable from POST /cfg and readable from GET /cfg.
 * Field order/type is duplicated in the table inside control_arbiter.c, which
 * is the single source of truth for the JSON names.
 */
typedef struct {
    float dt_ms;         /* integration dt of arbiter_tick_2ms (see warning below) */
    float pos_lead_max;  /* m,   stable_pos may lead wheel_x by at most this      */
    float yaw_lead_max;  /* deg, stable_yaw may lead (yaw-q_head) by this much    */
    float head_max;      /* deg, hard head clamp (hardware limit is +-45)         */
    float vx_max;        /* m/s, output clamp                                     */
    float vyaw_max;      /* deg/s, output clamp                                   */
    float stale_ms;      /* vision older than this -> stop integrating refs       */
    float lost_ms;       /* vision older than this -> all visual output forced 0  */
    float head_lpf;      /* 0..1, per-tick low pass on head_des                   */
    float cmd_timeout_ms;/* teacher channel goes stale after this                 */
    float k_dyaw;        /* heading loop D gain, torque-count / (deg/s). 0 = pure P */
    float vx_rate_max;   /* m/s^2, per-tick slew limit on vx_des. 0 = disabled     */
    float vyaw_rate_max; /* deg/s^2, per-tick slew limit on vyaw_des. 0 = disabled */
    float vx_stop_eps;   /* m/s,   |vx target| below this snaps to exactly 0      */
    float vyaw_stop_eps; /* deg/s, |vyaw target| below this snaps to exactly 0    */
    int   src;           /* ARB_SRC_*, 0 = manual (arbiter inactive)              */
    int   img_frame;     /* 0 = camera on head, 1 = camera on body                */
    int   gaze_en;       /* 1 = table-edge guard active                           */
    float gaze_hold_ms;  /* danger flag decays if /gaze stops coming             */
    int   cfc_hz;        /* CfC inference rate                                    */
    float fov_h;         /* deg, horizontal FOV used to turn person_cx into theta_img */
    float dist_k;        /* dist_est = dist_k / person_h  (metres)                */
    float dist_lpf;      /* s, first-order distance low pass                      */
    float dist_target;   /* m, observation only; the model decides what to do     */
    float clip_sigma;    /* domain-guard clip, used when the blob says 0          */
    int   udp_en;        /* 1 = allow the UDP telemetry stream                    */
    int   udp_div;       /* 1 = 500 Hz, 4 = 125 Hz, ...                           */
    int   udp_port;      /* UDP telemetry port, default 9001                      */
} arb_cfg_t;

/* ---- Globals in robot.cc / imu.cc that arbiter_tick_2ms() integrates into.
 * They are declared by robot.h / imu.h; listed here only for reference:
 *
 *   float stable_pos;      robot.cc   position reference (m)
 *   float stable_yaw;      robot.cc   heading reference (deg)
 *   float yaw_ctrl_time;   robot.cc   ms; robot.cc re-freezes stable_yaw
 *                                     when now - yaw_ctrl_time > 3000
 *   float wheel_x;         robot.cc   (declared in robot.cc only)
 *   float yaw, q_head;     imu.cc / robot.cc
 *   float target_head_pos; robot.cc   head servo target (deg)
 *   int   robot_state;     robot.cc   0 = fallen / lifted / off
 * ------------------------------------------------------------------------ */

/* Init is idempotent and safe to call from any task. All other entry points
 * call it lazily, so robot.cc may call arbiter_tick_2ms() from boot. */
void arbiter_init(void);

/* ---- the 2 ms hook (called by robot.cc robot_control) -------------------
 *
 * WARNING: robot_control() currently runs 3 x vTaskDelay(pdMS_TO_TICKS(2))
 * per iteration, so the real period is ~6 ms, not 2 ms. dt_ms defaults to
 * 2.0 to match the agreed interface; set POST /cfg dt_ms=6 once the loop
 * period has been measured, otherwise the reference integration runs 3x
 * too slow and the robot follows at ~1/3 of the commanded speed.
 * --------------------------------------------------------------------- */
void arbiter_tick_2ms(void);

/* ---- state queries ---------------------------------------------------- */
bool arbiter_active(void);              /* a non-manual source is selected */
const ctrl_out_t *arbiter_get(void);    /* lock free, points at a stable copy */
int  arbiter_source(void);              /* ARB_SRC_* */
int  arbiter_set_source(int src);       /* 0 ok, -1 rejected */
arb_cfg_t *arbiter_cfg(void);           /* never NULL; may be edited in place */
int  arbiter_cfg_set(const char *key, float value); /* 0 ok, -1 unknown key */

/* Field table for GET /cfg. Iteration order is stable and matches the JSON
 * object the HTTP layer builds; there is no second copy of the field list. */
int         arbiter_cfg_count(void);
const char *arbiter_cfg_name(int index);     /* NULL when out of range */
int         arbiter_cfg_is_int(int index);
float       arbiter_cfg_get_float(int index);
int         arbiter_cfg_get_int(int index);

/* ---- producer entry points -------------------------------------------- */
void arbiter_post_raw(const ctrl_out_t *raw);   /* cfc_task, ~50 Hz */
void arbiter_set_teacher(float head_des, float vx_des, float vyaw_des,
                         int mode, int emote);  /* HTTP /mode */
void arbiter_note_feat(void);                   /* HTTP /feat: refresh age */
bool arbiter_feature_fresh(void);               /* age < cfg.stale_ms */
int64_t arbiter_feature_age_ms(void);           /* -1 when never received */

/* Flash writes for POST /model stall the instruction cache for a few ms per
 * burst. While one is in flight the wheel channels are forced to zero so the
 * balance loop cannot act on an integrated reference it missed. */
void arbiter_set_upload(bool busy);

/* ---- misc used by the HTTP layer / telemetry --------------------------- */
int  arbiter_emote(void);               /* last emote accepted by the shell */
void arbiter_set_emote(int emote);      /* HTTP /emote */
int  arbiter_block_reason(void);        /* bitmask, for /status debug */

/* Block reasons reported through arbiter_block_reason(). */
#define ARB_BLK_NONE      0
#define ARB_BLK_GAZE      (1 << 0)
#define ARB_BLK_ROBOT_OFF (1 << 1)
#define ARB_BLK_NAN       (1 << 2)
#define ARB_BLK_VISION    (1 << 3)
#define ARB_BLK_NOCMD     (1 << 4)
#define ARB_BLK_UPLOAD    (1 << 5)

#ifdef __cplusplus
}
#endif

#endif /* CONTROL_ARBITER_H */
