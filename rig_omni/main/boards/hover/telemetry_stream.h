#ifndef TELEMETRY_STREAM_H
#define TELEMETRY_STREAM_H
/*
 * telemetry_stream.h -- 500 Hz proprioceptive UDP stream for data collection.
 *
 * A dedicated task (core 0, low priority) pushes a fixed binary frame to the
 * last client that sent anything to the telemetry port. It never allocates,
 * never blocks the control loop and stops by itself when no client is around,
 * so it costs nothing during normal operation.
 *
 * Enable/disable: cfg.udp_en (POST /cfg), rate divider cfg.udp_div
 * (1 = 500 Hz, 2 = 250 Hz, ...), port cfg.udp_port (default 9001).
 *
 * FRAME (little endian, no padding, 152 bytes with 34 floats)
 *   0  u32 magic 'TLM1'
 *   4  u16 ver = 1
 *   6  u16 len (bytes)
 *   8  u32 seq
 *   12 u32 t_ms (robot uptime)
 *   16 float imu[8]    roll, pitch, yaw, dq, pit_gyro, ax, ay, az
 *   48 float wheel[6]  v1, v2, x1, x2, wheel_x, wheek_vx
 *   72 float head[2]   q_head, target_head_pos
 *   80 float ctrl[6]   stable_pos, stable_yaw, vx, vyaw, tor1, tor2
 *   104 float st[2]    robot_state, control_mode
 *   112 float arb[6]   head_des, vx_des, vyaw_des, mode, emote, src
 *   136 float feat[4]  age_ms, n_obj, theta_img, seen
 */

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void telemetry_stream_start(void);
void telemetry_stream_stop(void);
bool telemetry_stream_running(void);
bool telemetry_stream_enabled(void);
void telemetry_stream_set_enabled(bool on);

/* Client address as host-order u32, 0 when nobody has ever subscribed. */
uint32_t telemetry_stream_client_ip(void);
uint32_t telemetry_stream_frames(void);

#ifdef __cplusplus
}
#endif

#endif /* TELEMETRY_STREAM_H */
