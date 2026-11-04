#ifndef CFC_TASK_H
#define CFC_TASK_H
/*
 * cfc_task.h -- vision frame ingestion, observation assembly and the CfC
 * inference task.
 *
 * The inference runs in its own task pinned to core 1 with priority 5, i.e.
 * strictly below robot_control (priority 12). It never touches the 500 Hz
 * loop: it publishes through seqlocks only.
 *
 * Vision frames arrive from the PC at 10-15 Hz on POST /feat (binary, see
 * TECH-PLAN 5.2). The newest frame is kept in a seqlock-protected snapshot;
 * the inference task reads it without blocking, assembles the observation
 * vector and posts the result through arbiter_post_raw().
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Reply of one POST /feat, mirrored into the JSON response. */
typedef struct {
    int      rx_bytes;
    int      n_obj;
    int      n_feat;
    int      ok;        /* 1 parsed, 0 rejected (see err)            */
    const char *err;    /* reason when ok == 0                       */
    int      age_ms;    /* robot_rx_ms - pc_t_ms, 0 when implausible */
    int      dt_ms;     /* robot-side interval to the previous frame  */
    int64_t  cfc_us;    /* duration of the last CfC step, -1 unknown  */
    int      cfc_hz;    /* measured inference rate, 0 when idle       */
} cfc_feat_reply_t;

/* Parse and publish one /feat frame. Safe from the HTTP task. */
void cfc_feat_ingest(const uint8_t *buf, size_t len, cfc_feat_reply_t *reply);

/* Start/stop the inference task (idempotent). Called from the debug server
 * start path; nothing happens until a model is loaded and src == cfc. */
void cfc_task_start(void);
bool cfc_task_running(void);

/* Telemetry helpers for /status. */
int64_t cfc_task_last_us(void);
int     cfc_task_last_hz(void);
int     cfc_task_frames(void);
int     cfc_task_dropped(void);      /* clip-ratio guard trips */

/* Observation length the assembler produces (106 for CfC v1). */
int cfc_task_obs_dim(void);

#ifdef __cplusplus
}
#endif

#endif /* CFC_TASK_H */
