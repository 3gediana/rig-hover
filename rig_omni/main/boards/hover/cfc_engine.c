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
 *   5. cfc_step() stays libm-exact. The LUT and the int8 kernels live behind
 *      cfc_step_q8() only, so the float32 path keeps its 1e-5 parity with the
 *      training reference.
 *
 * INT8 KERNEL
 * -----------
 * The int32 dot product runs on the ESP32-S3 PIE int8 SIMD unit when the
 * translation unit is built for that target: EE.VMULAS.S8.ACCX multiplies 16
 * int8 pairs per instruction and accumulates into the 40-bit ACCX register, so
 * one 64-tap row costs 4 instructions instead of 64. esp-dsp ships only
 * dspi_dotprod_s8(), an image-2D s8 kernel that returns a shifted int8 and
 * therefore cannot serve an int32 accumulation, which is why the four-instruction
 * sequence is written here instead.
 *
 * ACCX is a hardware register the compiler does not model, so the kernel keeps
 * q0/q1 and ACCX to itself, runs inside one uninterrupted asm block per row,
 * and is validated against the scalar reference by cfc_q8_selftest() before it
 * is ever used. AMS: ACCX is not saved across interrupts by the FreeRTOS
 * context switch, but this kernel is a leaf sequence with no call and no yield,
 * so only an interrupt could disturb it -- the same exposure esp-dsp and esp-nn
 * int8 kernels accept, and the reason the self-test runs before first use.
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "cfc_engine.h"

/* Unconditional force-inline: the firmware builds with -Os, which turns a plain
 * inline request into a real call, and a call per output row costs more than
 * the row's own arithmetic. */
#if defined(_MSC_VER)
#define CFC_ALWAYS_INLINE static __forceinline
#else
#define CFC_ALWAYS_INLINE static inline __attribute__((always_inline))
#endif

#if defined(__XTENSA__)
#include "sdkconfig.h"
#endif
#if defined(__XTENSA__) && defined(CONFIG_IDF_TARGET_ESP32S3) && \
    !defined(CFC_NO_PIE) && !defined(CFC_USE_PIE)
#define CFC_USE_PIE 1
#endif
#ifndef CFC_USE_PIE
#define CFC_USE_PIE 0
#endif

/* ------------------------------------------------------------------ */
/* libm activations: the float32 path stays bit-exact                   */
/* ------------------------------------------------------------------ */

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

/* ------------------------------------------------------------------ */
/* tanh / sigmoid lookup table                                         */
/* ------------------------------------------------------------------ */

/* tanh over [-8, 8] in CFC_TANH_LUT_N evenly spaced points; 1 KB of .rodata.
 * Values are tanh(-8 + i * 16 / 255) as doubles rounded to float. Only the
 * int8 path reads it; linear interpolation between neighbours keeps the error
 * at 3.8e-4 (measured, see cfc_selftest). */
static const float s_tanh_lut[CFC_TANH_LUT_N] = {
    -0.999999775f, -0.999999745f, -0.999999711f, -0.999999672f,
    -0.999999628f, -0.999999578f, -0.999999522f, -0.999999458f,
    -0.999999386f, -0.999999304f, -0.999999211f, -0.999999105f,
    -0.999998985f, -0.99999885f, -0.999998696f, -0.999998522f,
    -0.999998324f, -0.9999981f, -0.999997846f, -0.999997558f,
    -0.999997231f, -0.999996861f, -0.999996441f, -0.999995965f,
    -0.999995426f, -0.999994814f, -0.999994121f, -0.999993335f,
    -0.999992444f, -0.999991433f, -0.999990288f, -0.99998899f,
    -0.999987517f, -0.999985848f, -0.999983956f, -0.999981811f,
    -0.999979379f, -0.999976622f, -0.999973497f, -0.999969953f,
    -0.999965936f, -0.999961381f, -0.999956218f, -0.999950364f,
    -0.999943728f, -0.999936204f, -0.999927675f, -0.999918005f,
    -0.999907043f, -0.999894615f, -0.999880525f, -0.999864552f,
    -0.999846443f, -0.999825914f, -0.99980264f, -0.999776255f,
    -0.999746343f, -0.999712433f, -0.999673991f, -0.99963041f,
    -0.999581005f, -0.999524997f, -0.999461504f, -0.999389527f,
    -0.999307932f, -0.999215436f, -0.999110583f, -0.998991725f,
    -0.998856991f, -0.998704265f, -0.998531147f, -0.998334919f,
    -0.998112502f, -0.997860405f, -0.99757468f, -0.99725085f,
    -0.996883851f, -0.996467945f, -0.99599664f, -0.995462589f,
    -0.994857478f, -0.994171906f, -0.993395239f, -0.992515461f,
    -0.991518991f, -0.990390494f, -0.989112659f, -0.987665954f,
    -0.986028363f, -0.984175079f, -0.982078183f, -0.979706277f,
    -0.977024095f, -0.973992071f, -0.970565882f, -0.966695955f,
    -0.962326946f, -0.957397202f, -0.951838204f, -0.945574013f,
    -0.938520729f, -0.930585994f, -0.921668554f, -0.911657936f,
    -0.900434264f, -0.887868289f, -0.873821687f, -0.858147698f,
    -0.840692204f, -0.82129532f, -0.799793599f, -0.776022946f,
    -0.749822297f, -0.721038129f, -0.689529807f, -0.655175705f,
    -0.617880007f, -0.577579944f, -0.534253191f, -0.487924983f,
    -0.438674475f, -0.386639779f, -0.332021088f, -0.27508138f,
    -0.216144257f, -0.155588699f, -0.0938407261f, -0.0313622604f,
    0.0313622604f, 0.0938407261f, 0.155588699f, 0.216144257f,
    0.27508138f, 0.332021088f, 0.386639779f, 0.438674475f,
    0.487924983f, 0.534253191f, 0.577579944f, 0.617880007f,
    0.655175705f, 0.689529807f, 0.721038129f, 0.749822297f,
    0.776022946f, 0.799793599f, 0.82129532f, 0.840692204f,
    0.858147698f, 0.873821687f, 0.887868289f, 0.900434264f,
    0.911657936f, 0.921668554f, 0.930585994f, 0.938520729f,
    0.945574013f, 0.951838204f, 0.957397202f, 0.962326946f,
    0.966695955f, 0.970565882f, 0.973992071f, 0.977024095f,
    0.979706277f, 0.982078183f, 0.984175079f, 0.986028363f,
    0.987665954f, 0.989112659f, 0.990390494f, 0.991518991f,
    0.992515461f, 0.993395239f, 0.994171906f, 0.994857478f,
    0.995462589f, 0.99599664f, 0.996467945f, 0.996883851f,
    0.99725085f, 0.99757468f, 0.997860405f, 0.998112502f,
    0.998334919f, 0.998531147f, 0.998704265f, 0.998856991f,
    0.998991725f, 0.999110583f, 0.999215436f, 0.999307932f,
    0.999389527f, 0.999461504f, 0.999524997f, 0.999581005f,
    0.99963041f, 0.999673991f, 0.999712433f, 0.999746343f,
    0.999776255f, 0.99980264f, 0.999825914f, 0.999846443f,
    0.999864552f, 0.999880525f, 0.999894615f, 0.999907043f,
    0.999918005f, 0.999927675f, 0.999936204f, 0.999943728f,
    0.999950364f, 0.999956218f, 0.999961381f, 0.999965936f,
    0.999969953f, 0.999973497f, 0.999976622f, 0.999979379f,
    0.999981811f, 0.999983956f, 0.999985848f, 0.999987517f,
    0.99998899f, 0.999990288f, 0.999991433f, 0.999992444f,
    0.999993335f, 0.999994121f, 0.999994814f, 0.999995426f,
    0.999995965f, 0.999996441f, 0.999996861f, 0.999997231f,
    0.999997558f, 0.999997846f, 0.9999981f, 0.999998324f,
    0.999998522f, 0.999998696f, 0.99999885f, 0.999998985f,
    0.999999105f, 0.999999211f, 0.999999304f, 0.999999386f,
    0.999999458f, 0.999999522f, 0.999999578f, 0.999999628f,
    0.999999672f, 0.999999711f, 0.999999745f, 0.999999775f,
};

#define CFC_TANH_LO    (-8.0f)
#define CFC_TANH_SCALE (255.0f / 16.0f)

float cfc_tanh_fast(float x)
{
    float f, d;
    int i;

    if (x <= CFC_TANH_LO) {
        return -1.0f;
    }
    if (x >= 8.0f) {
        return 1.0f;
    }
    if (!(x == x)) {          /* NaN: a neutral output beats propagating it */
        return 0.0f;
    }
    f = (x - CFC_TANH_LO) * CFC_TANH_SCALE;
    i = (int)f;
    if (i < 0) {
        i = 0;
    } else if (i > CFC_TANH_LUT_N - 2) {
        i = CFC_TANH_LUT_N - 2;
    }
    d = f - (float)i;
    return s_tanh_lut[i] + d * (s_tanh_lut[i + 1] - s_tanh_lut[i]);
}

float cfc_sigmoid_fast(float x)
{
    /* sigmoid(x) = 0.5 + 0.5 * tanh(x/2); |x| > 16 saturates anyway. */
    return 0.5f + 0.5f * cfc_tanh_fast(0.5f * x);
}

/* LeCun tanh on the LUT: 1.7159 * tanh(0.666 * v). */
static float cfc_lecun_fast(float v)
{
    return 1.7159f * cfc_tanh_fast(0.666f * v);
}

/* ------------------------------------------------------------------ */
/* int8 dot product                                                    */
/* ------------------------------------------------------------------ */

/* Portable reference. Every other path must agree with this bit for bit. */
static int32_t q8_dot_c(const int8_t *w, const int8_t *qx, int n)
{
    int32_t acc = 0;
    int j;

    for (j = 0; j < n; ++j) {
        acc += (int32_t)w[j] * (int32_t)qx[j];
    }
    return acc;
}

#if CFC_USE_PIE
/* Force-inlined: this firmware builds with -Os, which neither inlines these
 * calls nor expands the 4-byte copies below, and three nested calls per output
 * row cost more than the 16 SIMD instructions they wrap. */
#define CFC_INLINE static inline __attribute__((always_inline))

/* 16 int8 MACs per instruction. Both pointers must be 16-byte aligned; the
 * scalar tail of `n & 15` taps is finished in C. */
CFC_INLINE int32_t q8_half_pie(const int8_t *w, const int8_t *x, int n)
{
    int32_t acc;
    uint32_t clr = 0;
    const int8_t *wp = w;
    const int8_t *xp = x;
    int rem = n & ~15;

    __asm__ __volatile__("movi %0, 0\n\t"
                         "wur.accx_0 %0\n\t"
                         "wur.accx_1 %0"
                         : "=&r"(clr)
                         :
                         : "memory");
    while (rem >= 16) {
        __asm__ __volatile__("ee.vld.128.ip q0, %0, 16\n\t"
                             "ee.vld.128.ip q1, %1, 16\n\t"
                             "ee.vmulas.s8.accx q0, q1"
                             : "+r"(wp), "+r"(xp)
                             :
                             : "memory");
        rem -= 16;
    }
    __asm__ __volatile__("rur.accx_0 %0" : "=r"(acc) : : "memory");
    rem = n & 15;
    while (rem-- > 0) {
        acc += (int32_t)(*wp++) * (int32_t)(*xp++);
    }
    return acc;
}

/* Word-wise copy of a weight row into an aligned stack slot. The blob format
 * makes layer_off a multiple of 4, so the source is 4-byte aligned and a load/
 * store pair per 4 taps is safe; this is the only reason the 106-wide
 * projection layer cannot load its rows directly. Written as a cast rather
 * than memcpy() because -Os emits a real call for size-4 memcpy. */
CFC_INLINE void q8_stage_copy(int8_t *dst, const int8_t *src, int n)
{
    int i = 0;
    uint32_t *d32 = (uint32_t *)(void *)dst;
    const uint32_t *s32 = (const uint32_t *)(const void *)src;

    for (; i + 4 <= n; i += 4) {
        d32[i >> 2] = s32[i >> 2];
    }
    for (; i < n; ++i) {
        dst[i] = src[i];
    }
}
#endif /* CFC_USE_PIE */

/* -1 = not probed yet, 0 = scalar, 1 = SIMD. Written once, then read-only. */
static int s_pie_state = -1;

int cfc_q8_selftest(void)
{
    if (s_pie_state >= 0) {
        return s_pie_state ? 0 : -1;
    }
#if CFC_USE_PIE
    {
        /* 128 taps covers every row this firmware can hold (projection 106,
         * backbone 128, cell 64); the buffers stay small enough to run from
         * cfc_model_init() on a 4 KB stack. */
        enum { TEST_MAX = 128 };
        static const int lens[] = {16, 32, 48, 64, 96, 106, 128};
        /* static, not stack: the 16-byte alignment was computed against the
         * caller's stack pointer and panicked with Load/StoreProhibited when
         * this ran from cfc_model_init(). One-shot probe, so not reentrant. */
        static CFC_ALIGN16 int8_t wa[TEST_MAX + 16];
        static CFC_ALIGN16 int8_t xa[TEST_MAX + 16];
        static CFC_ALIGN16 int8_t tail[TEST_MAX + 16];
        uint32_t s = 0x1234ABCDu;
        int i, t, ok = 1;

        for (i = 0; i < TEST_MAX + 16; ++i) {
            s = s * 1664525u + 1013904223u;
            wa[i] = (int8_t)((int)((s >> 16) & 0xFFu) - 128);
            s = s * 1664525u + 1013904223u;
            xa[i] = (int8_t)((int)((s >> 16) & 0xFFu) - 128);
        }
        for (t = 0; t < (int)(sizeof(lens) / sizeof(lens[0])); ++t) {
            const int n = lens[t];
            /* SIMD on 16-byte aligned operands must equal the scalar reference,
             * both directly and through cfc_q8_dot() so the alignment-staging
             * path is covered too. */
            if (q8_half_pie(wa, xa, n) != q8_dot_c(wa, xa, n)) {
                ok = 0;
                break;
            }
            if (cfc_q8_dot(wa, xa, n) != q8_dot_c(wa, xa, n)) {
                ok = 0;
                break;
            }
            /* 4-byte aligned (not 16) row start: exercises q8_stage_copy. */
            memcpy(tail, wa, (size_t)(n + 4));
            if (cfc_q8_dot(tail + 4, xa, n) != q8_dot_c(tail + 4, xa, n)) {
                ok = 0;
                break;
            }
        }
        s_pie_state = ok ? 1 : 0;
    }
#else
    s_pie_state = 0;
#endif
    return s_pie_state ? 0 : -1;
}

int cfc_q8_simd_enabled(void)
{
    if (s_pie_state < 0) {
        cfc_q8_selftest();
    }
    return s_pie_state > 0 ? 1 : 0;
}

int32_t cfc_q8_dot(const int8_t *w, const int8_t *qx, int n)
{
    if (w == NULL || qx == NULL || n <= 0) {
        return 0;
    }
#if CFC_USE_PIE
    if (s_pie_state < 0) {
        cfc_q8_selftest();
    }
    if (s_pie_state > 0) {
        /* The SIMD loads need 16-byte alignment on both sides. qx always comes
         * from this module and is aligned; a weight row is only 4-byte aligned
         * when the row width is not a multiple of 16 (the 106-wide projection
         * layer), so those rows are staged through an aligned copy. */
        if (((uintptr_t)qx & 15u) == 0u) {
            if (((uintptr_t)w & 15u) == 0u) {
                return q8_half_pie(w, qx, n);
            }
            if (n <= CFC_Q8_ROW_MAX) {
                CFC_ALIGN16 int8_t stage[CFC_Q8_ROW_MAX];
                q8_stage_copy(stage, w, n);
                return q8_half_pie(stage, qx, n);
            }
        }
    }
#endif
    return q8_dot_c(w, qx, n);
}

/* Quantise `na` floats from a then `nb` floats from b into one contiguous int8
 * buffer, sharing a single scale. Returns 1 when a scale was written. */
static int q8_quantize2(const float *a, int na, const float *b, int nb,
                        int8_t *q, float *scale)
{
    float amax = 0.0f;
    float inv;
    int j;

    for (j = 0; j < na; ++j) {
        const float v = fabsf(a[j]);
        if (v > amax) {
            amax = v;
        }
    }
    for (j = 0; j < nb; ++j) {
        const float v = fabsf(b[j]);
        if (v > amax) {
            amax = v;
        }
    }
    if (!(amax > 1e-20f)) {
        memset(q, 0, (size_t)(na + nb));
        *scale = 0.0f;
        return 0;
    }
    inv = 127.0f / amax;
    for (j = 0; j < na; ++j) {
        const float v = a[j] * inv;
        int k = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
        if (k > 127) {
            k = 127;
        } else if (k < -127) {
            k = -127;
        }
        q[j] = (int8_t)k;
    }
    for (j = 0; j < nb; ++j) {
        const float v = b[j] * inv;
        int k = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
        if (k > 127) {
            k = 127;
        } else if (k < -127) {
            k = -127;
        }
        q[na + j] = (int8_t)k;
    }
    *scale = amax / 127.0f;
    return 1;
}

int cfc_q8_quantize(const float *x, int n, int8_t *qx, float *scale)
{
    if (scale == NULL) {
        return 0;
    }
    if (x == NULL || qx == NULL || n <= 0) {
        *scale = 0.0f;
        return 0;
    }
    return q8_quantize2(x, n, NULL, 0, qx, scale);
}

/* ------------------------------------------------------------------ */
/* self test                                                           */
/* ------------------------------------------------------------------ */

void cfc_selftest(cfc_selftest_t *out)
{
    int k;

    if (out == NULL) {
        return;
    }
    memset(out, 0, sizeof(*out));

    /* LUT tanh: dense sweep of [-16, 16] against libm, plus the saturation
     * points, plus sigmoid over [-20, 20]. */
    for (k = -8000; k <= 8000; ++k) {
        const float x = (float)k * 0.002f;
        const float e = fabsf(cfc_tanh_fast(x) - tanhf(x));
        if (e > out->tanh_max_err) {
            out->tanh_max_err = e;
        }
    }
    {
        const float ep = fabsf(cfc_tanh_fast(8.0f) - tanhf(8.0f));
        const float en = fabsf(cfc_tanh_fast(-8.0f) - tanhf(-8.0f));
        out->tanh_sat_err = (ep > en) ? ep : en;
    }
    for (k = -20000; k <= 20000; ++k) {
        const float x = (float)k * 0.001f;
        const float e = fabsf(cfc_sigmoid_fast(x) - (1.0f / (1.0f + expf(-x))));
        if (e > out->sigmoid_max_err) {
            out->sigmoid_max_err = e;
        }
    }

    /* Per-output-channel int8 weight x int8 activation error against the exact
     * float32 dot, expressed relative to the largest output of the layer
     * (dividing by each row's own value would blow up on rows that happen to
     * land near zero). This is the "int8 path vs float32 reference" relative
     * error of one 64x64 W8A8 layer, measured on 16 synthetic rows. */
    {
        enum { K_OUT = 16, K_IN = 64 };
        CFC_ALIGN16 int8_t qx[K_IN];
        int8_t qw[K_IN];
        float w[K_IN];
        float x[K_IN];
        float exact[K_OUT];
        float approx[K_OUT];
        uint32_t s = 0x9E3779B9u;
        float xmax = 0.0f, worst = 0.0f, xs = 0.0f;
        int r;

        for (k = 0; k < K_IN; ++k) {
            s = s * 1664525u + 1013904223u;
            x[k] = ((float)(int)((s >> 8) & 0xFFFFu) / 32768.0f) - 1.0f;
        }
        cfc_q8_quantize(x, K_IN, qx, &xs);
        for (r = 0; r < K_OUT; ++r) {
            float amax = 0.0f, ws = 1.0f, e = 0.0f;
            for (k = 0; k < K_IN; ++k) {
                s = s * 1664525u + 1013904223u;
                w[k] = ((float)(int)((s >> 8) & 0xFFFFu) / 32768.0f) - 1.0f;
                if (fabsf(w[k]) > amax) {
                    amax = fabsf(w[k]);
                }
            }
            if (amax > 1e-20f) {
                ws = amax / 127.0f;
                for (k = 0; k < K_IN; ++k) {
                    const float v = w[k] / ws;
                    int q = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
                    if (q > 127) {
                        q = 127;
                    } else if (q < -127) {
                        q = -127;
                    }
                    qw[k] = (int8_t)q;
                }
            } else {
                memset(qw, 0, (size_t)K_IN);
            }
            for (k = 0; k < K_IN; ++k) {
                e += w[k] * x[k];
            }
            exact[r] = e;
            if (fabsf(e) > xmax) {
                xmax = fabsf(e);
            }
            approx[r] = (float)cfc_q8_dot(qw, qx, K_IN) * ws * xs;
        }
        if (xmax > 1e-20f) {
            for (r = 0; r < K_OUT; ++r) {
                const float d = fabsf(approx[r] - exact[r]) / xmax;
                if (d > worst) {
                    worst = d;
                }
            }
        }
        out->q8_dot_max_rel = worst;
    }

    out->q8_simd = (cfc_q8_selftest() == 0 && s_pie_state > 0) ? 1 : 0;
}

/* ------------------------------------------------------------------ */
/* step                                                                */
/* ------------------------------------------------------------------ */

/* Dimension of z: backbone_units when a backbone exists, else concat(in, h). */
static int cfc_z_units(const cfc_cell_t *c)
{
    return (c->backbone_units > 0) ? c->backbone_units
                                   : (c->input_dim + c->hidden_dim);
}

int cfc_validate(const cfc_cell_t *c)
{
    const unsigned m = (c == NULL) ? 0u : c->int8_mask;

    if (c == NULL) {
        return -1;
    }
    if (c->input_dim < 0 || c->hidden_dim <= 0 || c->backbone_units < 0) {
        return -1;
    }
    if (m & ~CFC_W8_ALL) {
        return -5;                       /* unknown bit: the mask is corrupt */
    }
    if (m & CFC_W8_BB) {
        if (c->backbone_units > 0 && (c->bb_w8 == NULL || c->bb_ws == NULL)) {
            return -5;
        }
    } else if (c->backbone_units > 0 && c->bb_w == NULL) {
        return -1;                       /* cfc_step_q8 needs bb_w when bb is float */
    }
    if (((m & CFC_W8_FF1) && (c->ff1_w8 == NULL || c->ff1_ws == NULL)) ||
        ((m & CFC_W8_FF2) && (c->ff2_w8 == NULL || c->ff2_ws == NULL)) ||
        ((m & CFC_W8_TA)  && (c->ta_w8 == NULL || c->ta_ws == NULL)) ||
        ((m & CFC_W8_TB)  && (c->tb_w8 == NULL || c->tb_ws == NULL))) {
        return -5;
    }
    if (!(m & CFC_W8_FF1) && c->ff1_w == NULL) {
        return -1;
    }
    if (!(m & CFC_W8_FF2) && c->ff2_w == NULL) {
        return -1;
    }
    if (!(m & CFC_W8_TA) && c->ta_w == NULL) {
        return -1;
    }
    if (!(m & CFC_W8_TB) && c->tb_w == NULL) {
        return -1;
    }
    /* Biases and the retention gate are float32 in both modes. */
    if (c->ff1_b == NULL || c->ff2_b == NULL || c->ta_b == NULL ||
        c->tb_b == NULL) {
        return -1;
    }
    if (c->backbone_units > 0 && c->bb_b == NULL) {
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

    if (c == NULL || h_io == NULL || c->int8_mask != 0) {
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

/* Row accumulator: bias + sum(w[k] * x[k]). `w8 != NULL` selects the int8 form,
 * which dequantises with wscale[row] * xscale; otherwise the float32 matrix and
 * the float vector are used. One int8 row is exactly one EE.VMULAS accumulator
 * session, which is the cost this kernel is organised around. */
CFC_ALWAYS_INLINE float cfc_row_dot(const float *wf, const int8_t *w8,
                                    const float *ws, int row, int n,
                                    const float *x, const int8_t *qx,
                                    float xscale, float bias)
{
    if (w8 != NULL) {
        if (xscale != 0.0f) {
            return bias + (float)cfc_q8_dot(w8 + (size_t)row * (size_t)n, qx, n) *
                          (ws[row] * xscale);
        }
        return bias;
    }
    {
        const float *w = wf + (size_t)row * (size_t)n;
        float acc = bias;
        int j;
        for (j = 0; j < n; ++j) {
            acc += w[j] * x[j];
        }
        return acc;
    }
}

void cfc_step_q8(const cfc_cell_t *c, const float *in, float *h_io)
{
    CFC_ALIGN16 int8_t qx[CFC_MAX_Z + 16];
    float zbuf[CFC_MAX_Z];
    float su = 0.0f, sz = 0.0f;
    int in_dim, hid, bu, zu, i;
    unsigned mask;
    int bb8, z8, zfloat;

    if (c == NULL || h_io == NULL || c->int8_mask == 0) {
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
    if (zu > CFC_MAX_Z || (bu > 0 && in_dim + hid > CFC_MAX_Z)) {
        return;                 /* same contract as cfc_step */
    }
    mask = c->int8_mask;
    bb8 = (mask & CFC_W8_BB) != 0;
    z8 = (mask & (CFC_W8_FF1 | CFC_W8_FF2 | CFC_W8_TA | CFC_W8_TB)) != 0;
    /* True when at least one z consumer keeps a float32 matrix, which means z
     * must also exist as floats. */
    zfloat = (mask & (CFC_W8_FF1 | CFC_W8_FF2 | CFC_W8_TA | CFC_W8_TB)) !=
             (CFC_W8_FF1 | CFC_W8_FF2 | CFC_W8_TA | CFC_W8_TB);

    /* --- 1. z ---
     * qx holds concat(in, h) while the backbone runs and is re-used for z
     * afterwards: the two vectors never overlap in time and the second one is
     * never longer, so one buffer serves both. */
    if (bu > 0) {
        const int k = in_dim + hid;
        if (bb8) {
            q8_quantize2(in, in_dim, h_io, hid, qx, &su);
        }
        for (i = 0; i < bu; ++i) {
            float acc;
            if (bb8) {
                acc = cfc_row_dot(NULL, c->bb_w8, c->bb_ws, i, k, NULL, qx, su,
                                  c->bb_b[i]);
            } else {
                /* Float32 backbone: same two-segment loop as cfc_step(), so a
                 * float matrix costs neither a concat buffer nor a copy. */
                const float *w = c->bb_w + (size_t)i * (size_t)k;
                int j;
                acc = c->bb_b[i];
                for (j = 0; j < in_dim; ++j) {
                    acc += w[j] * in[j];
                }
                for (j = 0; j < hid; ++j) {
                    acc += w[in_dim + j] * h_io[j];
                }
            }
            zbuf[i] = cfc_lecun_fast(acc);
        }
    } else {
        /* No backbone: z is concat(in, h). With a float32 consumer the concat
         * has to materialise; with four int8 consumers it is quantised straight
         * out of the two halves. */
        if (zfloat || !z8) {
            int j;
            for (j = 0; j < in_dim; ++j) {
                zbuf[j] = in[j];
            }
            for (j = 0; j < hid; ++j) {
                zbuf[in_dim + j] = h_io[j];
            }
        }
    }
    if (z8) {
        if (bu > 0 || zfloat) {
            cfc_q8_quantize(zbuf, zu, qx, &sz);
        } else {
            q8_quantize2(in, in_dim, h_io, hid, qx, &sz);
        }
    }

    /* --- 2. ff1 / ff2 / t -> h_new -> retention gate --- */
    for (i = 0; i < hid; ++i) {
        float a1 = cfc_row_dot(c->ff1_w, c->ff1_w8, c->ff1_ws, i, zu, zbuf, qx,
                               sz, c->ff1_b[i]);
        float a2 = cfc_row_dot(c->ff2_w, c->ff2_w8, c->ff2_ws, i, zu, zbuf, qx,
                               sz, c->ff2_b[i]);
        float ta = cfc_row_dot(c->ta_w, c->ta_w8, c->ta_ws, i, zu, zbuf, qx,
                               sz, c->ta_b[i]);
        float tb = cfc_row_dot(c->tb_w, c->tb_w8, c->tb_ws, i, zu, zbuf, qx,
                               sz, c->tb_b[i]);
        float t, f1, f2, h_new;

        f1 = cfc_tanh_fast(a1);
        f2 = cfc_tanh_fast(a2);
        t = cfc_sigmoid_fast(ta * c->ts + tb);
        h_new = f1 * (1.0f - t) + t * f2;

        if (c->use_retention != 0 && c->ret != NULL) {
            const float lam = cfc_sigmoid_fast(c->ret[i]);
            if (lam != 0.0f) {
                h_new += lam * (h_io[i] - h_new);
            }
        }
        h_io[i] = h_new;
    }
}
