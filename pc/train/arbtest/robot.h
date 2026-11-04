#ifndef ARB_ROBOT_H
#define ARB_ROBOT_H
/* Minimal shadow of the firmware robot.h: only the globals the safety shell
 * reads. Values are driven by the test through arb_shim.c. */
extern float q_head;
extern float target_head_pos;
extern float stable_pos;
extern float stable_yaw;
extern float yaw_ctrl_time;
extern float wheel_x;
#endif

