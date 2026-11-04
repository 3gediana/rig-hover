#ifndef ARB_GAZE_GUARD_H
#define ARB_GAZE_GUARD_H
#include <stdbool.h>
#include <stdint.h>
void    gaze_guard_init(void);
void    gaze_guard_set(int danger);
bool    gaze_guard_danger(void);
bool    gaze_guard_latched(void);
int64_t gaze_guard_age_ms(void);
#endif
