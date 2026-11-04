/*
 * gaze_guard.c -- table-edge emergency stop flag.
 *
 * Single writer (the HTTP server task) and many readers (control loop, cfc
 * task, telemetry). Every field is a naturally aligned 32-bit value, so plain
 * loads and stores are atomic on the ESP32-S3 and no lock is needed.
 */

#include <stdbool.h>
#include <stdint.h>

#include "esp_timer.h"

#include "gaze_guard.h"
#include "control_arbiter.h"

static volatile int      s_danger      = 0;   /* latched flag, 0/1        */
static volatile uint32_t s_events      = 0;   /* rising edges            */
static volatile int64_t  s_last_us     = -1;  /* last /gaze arrival      */
static volatile int64_t  s_danger_us   = 0;   /* when it was last raised */

#define GAZE_DEFAULT_HOLD_MS 1000.0f

void gaze_guard_init(void)
{
    s_danger = 0;
    s_events = 0;
    s_last_us = -1;
}

void gaze_guard_set(int danger)
{
    const int64_t now = esp_timer_get_time();
    s_last_us = now;
    if (danger) {
        if (!s_danger) {
            s_events = s_events + 1;
            s_danger_us = now;
        }
        s_danger = 1;
    } else {
        s_danger = 0;
    }
}

bool gaze_guard_latched(void)
{
    return s_danger != 0;
}

bool gaze_guard_danger(void)
{
    if (!s_danger) {
        return false;
    }
    /* Hold the stop while the flag keeps arriving. If the PC goes silent for
     * longer than gaze_hold_ms the vision channel is dead anyway (the arbiter
     * zeroes every visual output by itself), so drop the latch instead of
     * freezing the robot until the next /gaze frame ever arrives. */
    const float hold_ms = arbiter_cfg()->gaze_hold_ms;
    const int64_t now = esp_timer_get_time();
    if (s_last_us >= 0 && (now - s_last_us) > (int64_t)(hold_ms * 1000.0f)) {
        return false;
    }
    return true;
}

uint32_t gaze_guard_events(void)
{
    return s_events;
}

int64_t gaze_guard_age_ms(void)
{
    if (s_last_us < 0) {
        return -1;
    }
    return (esp_timer_get_time() - s_last_us) / 1000;
}
