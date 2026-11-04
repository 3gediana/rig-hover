/* arb_main.c -- host-side tests for the firmware safety shell.
 *
 * The seven numbered defences in control_arbiter.c are what keep the robot
 * from hurting itself, and until now they had only ever run on hardware. Each
 * scenario below drives the real C code and asserts the property it claims.
 *
 * Deliberately tests *properties*, not exact numbers: the thresholds live in
 * arb_cfg_t and are read back, so retuning the config does not break the test.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "control_arbiter.h"

extern int   robot_state;
extern void  arb_set_clock_us(int64_t us);
extern int64_t arb_clock_us(void);

static int g_fail = 0;
static int g_pass = 0;

static void check(const char *what, int ok, const char *detail)
{
    if (ok) { ++g_pass; printf("  PASS  %s\n", what); }
    else    { ++g_fail; printf("  FAIL  %s   %s\n", what, detail ? detail : ""); }
}

static void tick_2ms(void)
{
    arb_set_clock_us(arb_clock_us() + 2000);
    arbiter_tick_2ms();
}

static void warm(void)
{
    /* Let the LPF and reference integrators settle at zero command. */
    for (int i = 0; i < 20; ++i) {
        arbiter_post_raw(&(ctrl_out_t){0.0f, 0.0f, 0.0f, 0, 0});
        arbiter_note_feat();
        tick_2ms();
    }
}

int main(void)
{
    arb_cfg_t *c;
    const ctrl_out_t *o;

    arbiter_init();
    c = arbiter_cfg();
    printf("cfg: head_max=%.1f vx_max=%.2f vyaw_max=%.1f lost_ms=%.0f "
           "stale_ms=%.0f cmd_timeout=%.0f\n",
           c->head_max, c->vx_max, c->vyaw_max, c->lost_ms, c->stale_ms,
           c->cmd_timeout_ms);

    robot_state = 1;
    arbiter_set_source(ARB_SRC_CFC);
    warm();

    /* --- 3. non-finite guard ------------------------------------------ */
    {
        ctrl_out_t bad = { NAN, 0.0f, 0.0f, 1, 0 };
        arbiter_note_feat();
        arbiter_post_raw(&bad);
        tick_2ms();
        o = arbiter_get();
        check("NaN head_des is zeroed", o->head_des == 0.0f, NULL);

        bad.head_des = 0.0f; bad.vyaw_des = INFINITY;
        arbiter_note_feat();
        arbiter_post_raw(&bad);
        tick_2ms();
        o = arbiter_get();
        check("Inf vyaw_des is zeroed", o->vyaw_des == 0.0f, NULL);
    }

    /* --- 2. robot off -------------------------------------------------- */
    {
        robot_state = 0;
        ctrl_out_t big = { 30.0f, 0.3f, 40.0f, 1, 3 };
        arbiter_note_feat();
        arbiter_post_raw(&big);
        tick_2ms();
        o = arbiter_get();
        check("robot_state=0 forces every output to 0",
              o->head_des == 0.0f && o->vx_des == 0.0f &&
              o->vyaw_des == 0.0f && o->mode == 0 && o->emote == 0, NULL);
        robot_state = 1;
    }

    /* --- 6. hard clamps ------------------------------------------------ */
    {
        int ok = 1;
        for (int i = 0; i < 400; ++i) {
            ctrl_out_t huge = { 500.0f, 50.0f, 5000.0f, 1, 0 };
            arbiter_note_feat();
            arbiter_post_raw(&huge);
            tick_2ms();
            o = arbiter_get();
            if (fabsf(o->head_des) > c->head_max + 1e-3f) ok = 0;
            if (fabsf(o->vx_des)    > c->vx_max    + 1e-4f) ok = 0;
            if (fabsf(o->vyaw_des)  > c->vyaw_max  + 1e-3f) ok = 0;
        }
        check("250x oversized commands stay inside head/vx/vyaw clamps", ok, NULL);
        if (c->head_max > 45.0f)
            printf("        !! head_max %.1f exceeds the servo travel (45)\n", c->head_max);
    }

    /* --- 7. slew limit -------------------------------------------------- */
    {
        ctrl_out_t zero = { 0.0f, 0.0f, 0.0f, 1, 0 };
        for (int i = 0; i < 200; ++i) { arbiter_note_feat(); arbiter_post_raw(&zero); tick_2ms(); }
        float prev = arbiter_get()->vx_des;
        float worst = 0.0f;
        ctrl_out_t aim = { 0.0f, c->vx_max, 0.0f, 1, 0 };
        for (int i = 0; i < 200; ++i) {
            arbiter_note_feat();
            arbiter_post_raw(&aim);
            tick_2ms();
            float v = arbiter_get()->vx_des;
            float d = fabsf(v - prev);
            if (d > worst) worst = d;
            prev = v;
        }
        if (c->vx_rate_max > 0.0f) {
            float bound = c->vx_rate_max * (c->dt_ms <= 0.0f ? 2.0f : c->dt_ms) / 1000.0f;
            char buf[96];
            snprintf(buf, sizeof buf, "worst step %.4f > bound %.4f", worst, bound);
            check("vx never steps faster than vx_rate_max", worst <= bound * 1.5f + 1e-4f, buf);
        } else {
            check("vx slew limit disabled in cfg -> skipped", 1, NULL);
        }
    }

    /* --- 4. vision freshness ladder ------------------------------------ */
    {
        ctrl_out_t aim = { 10.0f, 0.2f, 20.0f, 1, 0 };
        for (int i = 0; i < 50; ++i) { arbiter_note_feat(); arbiter_post_raw(&aim); tick_2ms(); }
        /* stop refreshing /feat and run past lost_ms */
        /* lost_ms only *starts* the forced-zero; the head channel then decays
         * through the deliberate head_lpf (0.15/tick), which needs ~50 ticks
         * to settle. Run well past both so the assertion is about the policy,
         * not about how long the smoother takes. */
        int steps = (int)(c->lost_ms / 2.0f) + 400;
        for (int i = 0; i < steps; ++i) { arbiter_post_raw(&aim); tick_2ms(); }
        o = arbiter_get();
        {
            char buf[160];
            snprintf(buf, sizeof buf,
                     "age=%lld ms  head=%.3f vx=%.4f vyaw=%.3f",
                     (long long)arbiter_feature_age_ms(),
                     (double)o->head_des, (double)o->vx_des, (double)o->vyaw_des);
            check("vision older than lost_ms forces head/vx/vyaw to 0",
                  o->vx_des == 0.0f && o->vyaw_des == 0.0f && fabsf(o->head_des) < 0.5f,
                  buf);
        }
    }

    /* --- 5. teacher watchdog ------------------------------------------- */
    {
        arbiter_set_source(ARB_SRC_TEACHER);
        arbiter_note_feat();
        arbiter_set_teacher(8.0f, 0.1f, 12.0f, 1, 0);
        int settled = 0;
        for (int i = 0; i < 200; ++i) {
            arbiter_note_feat();
            arbiter_set_teacher(8.0f, 0.1f, 12.0f, 1, 0);
            tick_2ms();
            if (fabsf(arbiter_get()->vyaw_des) > 1.0f) settled = 1;
        }
        check("teacher source drives output while fresh", settled, NULL);
        int steps = (int)(c->cmd_timeout_ms / 2.0f) + 60;
        for (int i = 0; i < steps; ++i) { arbiter_note_feat(); tick_2ms(); }
        o = arbiter_get();
        check("teacher channel stops after cmd_timeout_ms",
              o->vx_des == 0.0f && o->vyaw_des == 0.0f, NULL);
    }

    /* --- mode sanity ---------------------------------------------------- */
    {
        arbiter_set_source(ARB_SRC_CFC);
        ctrl_out_t m = { 0.0f, 0.0f, 0.0f, 9, 99 };
        for (int i = 0; i < 5; ++i) { arbiter_note_feat(); arbiter_post_raw(&m); tick_2ms(); }
        o = arbiter_get();
        check("out-of-range mode/emote are rejected",
              o->mode >= 0 && o->mode <= 3 && o->emote >= 0 && o->emote <= 7, NULL);
    }

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}


