/*
 * cfc_engine.c -- plain C99 implementation of the CfC single-step engine.
 *
 * Math, memory layout, stack use, thread safety and build flags are documented
 * in cfc_engine.h. Operator-by-operator this mirrors the numpy reference
 * (cfc_ref.py) and ncps.torch.CfCCell(mode="default"):
 *
 *   lecun(1.7159 * tanh(0.666 * v)) -> tanh -> sigmoid -> h_new = ff1*(1-t) + t*ff2
 *   -> optional retention gate
 *
 * INVARIANTS (keep them when touching this file)
 *   1. Single precision throughout, never widened to double: the ESP32-S3 only
 *      has a single-precision FPU, and staying in float32 lets the PC parity
 *      harness compare float32 bit patterns directly.
 *   2. The LeCun constant is 0.666, not 2/3. That is what the ncps source has.
 *   3. When lam == 0 the retention term is not added at all, so "gate enabled
 *      with lambda = 0" is bit-identical to "gate disabled" (this also covers
 *      the pathological case where h_new is exactly -0.0f and adding +0.0f
 *      would flip the sign bit).
 *   4. Every check happens before h_io is written: on error h_io keeps its
 *      previous value instead of holding a half-updated state.
 */

#include <math.h>
#include <string.h>

#include "cfc_engine.h"

/* LeCun tanh: 1.7159 * tanh(0.666 * v). */
static float cfc_lecun(float v)
{
    return 1.7159f * tanhf(0.666f * v);
}

/* 1 / (1 + exp(-v)); for very negative v, expf overflows to +inf and the
 * result converges to 0 without extra handling. */
static float cfc_sigmoid(float v)
{
    return 1.0f / (1.0f + expf(-v));
}

/* Dimension of z: backbone_units when a backbone exists, else concat(in, h). */
static int cfc_z_units(const cfc_cell_t *c)
{
    return (c->backbone_units > 0) ? c->backbone_units : (c->input_dim + c->hidden_dim);
}

int cfc_validate(const cfc_cell_t *c)
{
    if (c == NULL) {
        return -1;
    }
    if (c->input_dim < 0 || c->hidden_dim <= 0 || c->backbone_units < 0) {
        return -1;
    }
    if (c->ff1_w == NULL || c->ff1_b == NULL || c->ff2_w == NULL || c->ff2_b == NULL ||
        c->ta_w == NULL || c->ta_b == NULL || c->tb_w == NULL || c->tb_b == NULL) {
        return -1;
    }
    if (c->backbone_units > 0 && (c->bb_w == NULL || c->bb_b == NULL)) {
        return -1;
    }
    if (cfc_z_units(c) > CFC_MAX_Z) {
        return -2;
    }
    if (c->use_retention != 0 && c->ret == NULL) {
        return -3;
    }
    if (c->use_retention == 0 && c->ret != NULL) {
        return -4;
    }
    return 0;
}

void cfc_init_state(const cfc_cell_t *c, float *h)
{
    if (c == NULL || h == NULL || c->hidden_dim <= 0) {
        return;
    }
    memset(h, 0, (size_t)c->hidden_dim * sizeof(float));
}

void cfc_step(const cfc_cell_t *c, const float *in, float *h_io)
{
    float z[CFC_MAX_Z];
    int in_dim, hid, bu, zu, i, j;

    if (c == NULL || h_io == NULL) {
        return;
    }
    in_dim = c->input_dim;
    hid = c->hidden_dim;
    bu = c->backbone_units;
    if (in_dim < 0 || hid <= 0 || bu < 0) {
        return;
    }
    if (in_dim > 0 && in == NULL) {
        return;
    }
    zu = cfc_z_units(c);
    if (zu > CFC_MAX_Z) {
        /* Configuration exceeds the compile-time stack limit: leave h_io
         * untouched and let cfc_validate() catch it at init time. */
        return;
    }

    /* --- 1. z --- */
    if (bu > 0) {
        const int k = in_dim + hid;
        for (i = 0; i < bu; ++i) {
            const float *w = c->bb_w + (size_t)i * (size_t)k;
            float acc = c->bb_b[i];
            for (j = 0; j < in_dim; ++j) {
                acc += w[j] * in[j];
            }
            for (j = 0; j < hid; ++j) {
                acc += w[in_dim + j] * h_io[j];
            }
            z[i] = cfc_lecun(acc);
        }
    } else {
        /* No backbone: z = concat(in, h). Copy h into z first so h_io can be
         * overwritten safely below. */
        for (j = 0; j < in_dim; ++j) {
            z[j] = in[j];
        }
        for (j = 0; j < hid; ++j) {
            z[in_dim + j] = h_io[j];
        }
    }

    /* --- 2. ff1 / ff2 / t -> h_new -> retention gate ---
     * Unit-by-unit, in place: h_new[i] depends only on z and on h_io[i] itself,
     * and z is already complete, so reading h_io[i] (old state, needed by the
     * retention gate) and writing h_io[i] (new state) can happen in the same
     * iteration. */
    for (i = 0; i < hid; ++i) {
        const float *w1 = c->ff1_w + (size_t)i * (size_t)zu;
        const float *w2 = c->ff2_w + (size_t)i * (size_t)zu;
        const float *wa = c->ta_w + (size_t)i * (size_t)zu;
        const float *wb = c->tb_w + (size_t)i * (size_t)zu;
        float a1 = c->ff1_b[i];
        float a2 = c->ff2_b[i];
        float ta = c->ta_b[i];
        float tb = c->tb_b[i];
        float t, f1, f2, h_new;

        for (j = 0; j < zu; ++j) {
            const float zj = z[j];
            a1 += w1[j] * zj;
            a2 += w2[j] * zj;
            ta += wa[j] * zj;
            tb += wb[j] * zj;
        }
        f1 = tanhf(a1);
        f2 = tanhf(a2);
        t = cfc_sigmoid(ta * c->ts + tb);
        h_new = f1 * (1.0f - t) + t * f2;

        if (c->use_retention != 0 && c->ret != NULL) {
            const float lam = cfc_sigmoid(c->ret[i]);
            if (lam != 0.0f) {
                h_new += lam * (h_io[i] - h_new);
            }
        }
        h_io[i] = h_new;
    }
}
