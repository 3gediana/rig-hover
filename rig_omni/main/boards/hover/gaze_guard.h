#ifndef GAZE_GUARD_H
#define GAZE_GUARD_H
/*
 * gaze_guard.h -- table-edge (danger) emergency stop, highest priority.
 *
 * The PC perception pipeline looks at the same camera frame it uses for pose
 * detection and decides whether the robot is about to drive off the table.
 * It posts that flag to POST /gaze at 10-15 Hz. This module is the only place
 * that flag is stored; control_arbiter.c asks it before publishing any output
 * and zeroes the translation/rotation commands while it is set.
 *
 * Priority: above everything else, including the CfC and the teacher channel.
 * It cannot be overridden by /cfg, /mode or /cmd -- only a later /gaze with
 * danger=0 clears it.
 */

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void gaze_guard_init(void);

/* Called from POST /gaze. `danger` != 0 latches the stop. */
void gaze_guard_set(int danger);

/* Current state, already accounting for the hold timeout: once the PC stops
 * sending /gaze for gaze_hold_ms the latch clears itself, otherwise a crashed
 * PC would leave the robot frozen forever. */
bool gaze_guard_danger(void);

/* Raw latched value, ignoring the timeout (telemetry / debugging). */
bool gaze_guard_latched(void);

/* Number of danger transitions seen so far, for /status. */
uint32_t gaze_guard_events(void);

/* Milliseconds since the last /gaze frame, -1 if never. */
int64_t gaze_guard_age_ms(void);

#ifdef __cplusplus
}
#endif

#endif /* GAZE_GUARD_H */
