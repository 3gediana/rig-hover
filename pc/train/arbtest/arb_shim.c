/* arb_shim.c -- host definitions for everything control_arbiter.c references
 * from outside its own translation unit. The clock is settable so the safety
 * scenarios can drive time deterministically. */
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>

#include "freertos/FreeRTOS.h"
#include "robot.h"
#include "imu.h"
#include "gaze_guard.h"

float q_head = 0.0f;
float target_head_pos = 0.0f;
float stable_pos = 0.0f;
float stable_yaw = 0.0f;
float yaw_ctrl_time = 0.0f;
float wheel_x = 0.0f;
float yaw = 0.0f, pitch = 0.0f, roll = 0.0f, dq = 0.0f;
int   robot_state = 1;

static int64_t s_now_us = 1000000;
static int     s_gaze = 0;
static int64_t s_gaze_us = 0;

void    arb_set_clock_us(int64_t us) { s_now_us = us; }
int64_t arb_clock_us(void) { return s_now_us; }

int64_t esp_timer_get_time(void) { return s_now_us; }
TickType_t xTaskGetTickCount(void) { return (TickType_t)(s_now_us / 1000); }
void vTaskDelay(TickType_t ticks) { s_now_us += (int64_t)ticks * 1000; }

void host_log(const char *level, const char *tag, const char *fmt, ...)
{
    va_list ap;
    (void)level; (void)tag;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

void    gaze_guard_init(void) { }
void    gaze_guard_set(int danger) { s_gaze = danger; s_gaze_us = s_now_us; }
bool    gaze_guard_danger(void) { return s_gaze != 0; }
bool    gaze_guard_latched(void) { return s_gaze != 0; }
int64_t gaze_guard_age_ms(void) { return (s_now_us - s_gaze_us) / 1000; }

