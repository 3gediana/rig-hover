#ifndef CFC_ENGINE_H
#define CFC_ENGINE_H
/*
 * cfc_engine.h -- CfC single-step inference engine (plain C99, no dynamic memory,
 * no VLA, no platform-specific headers).
 *
 * MATH (CONTRACT 5.2; ground truth = ncps.torch.CfCCell, mode="default"):
 *
 *     x     = concat(in, h)                          (input_dim + hidden_dim)
 *     z     = lecun(bb_w @ x + bb_b)                 (backbone_units)
 *     ff1   = tanh(ff1_w @ z + ff1_b)                (hidden_dim)
 *     ff2   = tanh(ff2_w @ z + ff2_b)                (hidden_dim)
 *     t     = sigmoid((ta_w @ z + ta_b) * ts + (tb_w @ z + tb_b))
 *     h_new = ff1 * (1 - t) + t * ff2                (there is NO h self-loop)
 *
 *     retention gate (local extension; enabled only when use_retention != 0
 *     AND ret != NULL):
 *     lam   = sigmoid(ret)                           (hidden_dim)
 *     h_out = h_new + lam * (h - h_new)
 *
 * LeCun activation uses the literal constant 0.666 (NOT 2/3), which is what
 * the ncps source contains. Do not "fix" it.
 *
 * MEMORY LAYOUT
 * -------------
 * Every weight array is float32, row-major (out_features, in_features), i.e.
 * weights[i * in_features + j]. Each `const float *` points at the first
 * element of a contiguous array; no file format and no packed struct is
 * assumed. Loading and layout are the caller's responsibility.
 *
 *     field    shape                                             elements
 *     bb_w     backbone_units x (input_dim + hidden_dim)         bu*(in+hid)
 *     bb_b     backbone_units                                   bu
 *     ff1_w    hidden_dim x backbone_units                      hid*bu
 *     ff1_b    hidden_dim                                       hid
 *     ff2_w / ff2_b / ta_w / ta_b / tb_w / tb_b                 same as ff1
 *     ret      hidden_dim (read only when the retention gate is on)
 *
 * BACKBONE-LESS FORM (equivalent to ncps backbone_layers=0, the shape AutoNCP
 * produces): set backbone_units = 0. Then z = concat(in, h) directly, bb_w and
 * bb_b may be NULL, and the four weight matrices take (input_dim+hidden_dim)
 * input features.
 *
 * SPARSITY MASKS (AutoNCP wiring) are NOT a runtime input: ncps applies them as
 * weight[i] *= mask[i] inside F.linear, so the exported dense arrays must
 * already hold the masked weights (the cell builds the mask transpose-abs'd at
 * construction). IMPORTANT: ncps masks ONLY ff1 and ff2; time_a and time_b stay
 * dense even for AutoNCP wiring. Folding the mask into ta_w/tb_w as well makes
 * the hidden state wrong by O(0.6) -- test_parity.py [2b] guards this. The
 * engine needs no mask field; validate the export recipe there instead.
 *
 * INT8 WEIGHT PATH (optional, per layer, flags bit2 + layer_prec in the blob)
 * --------------------------------------------------------------------------
 * Each of the five weight matrices can independently be per-output-channel
 * symmetric int8, evaluated by cfc_step_q8(); a layer that is not marked int8
 * keeps its float32 matrix and is evaluated with the plain float loop, so a
 * cell may mix the two (cfc_cell_t.int8_mask). The float32 arrays of an int8
 * layer are unused and may be NULL, and vice versa; biases (bb_b, ff1_b,
 * ff2_b, ta_b, tb_b) and `ret` stay float32 in both cases, because they are
 * added after the dequantisation and carry no multiply.
 *
 *   W[i][:] ~= q[i][:] * wscale[i],  wscale[i] = max|W[i][:]| / 127
 *   x       ~= qx * xscale,          xscale   = max|x| / 127   (per step)
 *   acc     = sum(q[i][j] * qx[j])                             (int32)
 *   y[i]    = acc * wscale[i] * xscale + bias[i]
 *
 * The activation is a SINGLE int8 vector: one weight row costs exactly one
 * EE.VMULAS accumulator session. That is the whole design constraint of this
 * kernel: ACCX is a single 40-bit hardware accumulator, so a row needs a
 * "zero -> accumulate -> read back" round trip, measured at ~300 cycles on the
 * real board against ~20 cycles of actual SIMD work for a 64-tap row. Halving
 * the sessions per row (this change, from 768 to 384 per step) is therefore
 * worth far more than any arithmetic inside them.
 *
 * The price is activation resolution: 8-bit activations alone double the
 * tanh-head error of the previous 15-bit (hi/lo int8 pair) scheme, from 3.4e-3
 * to 6.3e-3 in normalised units (measured on 1669 frames of held-out sessions,
 * see cfc_selftest/q8_dot_max_rel and pc/train/README.md). Per-layer weight
 * precision is unaffected, so the output heads can still be float32.
 *
 * The int8 path also replaces every tanh/sigmoid with a 256-entry LUT plus
 * linear interpolation (cfc_tanh_fast); the float32 path keeps libm tanhf so
 * its bit-for-bit parity against the training reference stays intact.
 *
 * STACK USE
 * ---------
 * One float scratch buffer of max(backbone_units, input_dim+hidden_dim) is
 * needed, sized by the compile-time constant CFC_MAX_Z (default 256, i.e. at
 * most 1 KB of stack). The deployment shape (in=106, hid=64, bu=64) only uses
 * 64 floats; firmware can shrink it with -DCFC_MAX_Z=64. If a configuration
 * exceeds the limit, cfc_step returns without touching h_io (keeping the
 * previous state instead of writing garbage); call cfc_validate() at init to
 * reject such a configuration up front.
 *
 * cfc_step_q8 additionally keeps the int8 copy of its activation vectors on
 * the stack (CFC_MAX_Z bytes each, 16-byte aligned); it is a separate function
 * so the float32 path never pays for that frame.
 *
 * THREAD SAFETY
 * -------------
 * cfc_step / cfc_step_q8 use no static or global data and are reentrant. A
 * single cfc_cell_t can be read by several tasks concurrently, but each task
 * needs its own h buffer. The only module state is the one-shot result of
 * cfc_q8_selftest(), which is written once at init and then read-only.
 *
 * BUILD
 * -----
 *     cl /c /O2 /fp:precise /W3 cfc_engine.c        (MSVC, C mode)
 *     gcc -std=c99 -O2 -c cfc_engine.c              (C99)
 * Depends only on <math.h>, <stdint.h> and <string.h>.
 *
 * The ESP32-S3 int8 SIMD kernel is selected automatically when the translation
 * unit is built for an Xtensa target whose sdkconfig.h says ESP32-S3
 * (CFC_USE_PIE). -DCFC_NO_PIE forces the portable scalar kernel, which is also
 * what the PC parity harness compiles.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Stack scratch limit: the engine keeps one float buffer of
 * max(backbone_units, input_dim+hidden_dim) floats on the stack, bounded by
 * this compile-time constant. Default 256 floats (1 KB); the deployment shape
 * (in=106, hid=64, bu=64) only needs 64. Firmware may shrink or grow it with
 * -DCFC_MAX_Z=<n> (must be >= 1). */
#ifndef CFC_MAX_Z
#define CFC_MAX_Z 256
#endif

/* Longest int8 weight row the aligned-copy fallback can stage on the stack.
 * Rows longer than this go through the scalar kernel instead of failing. */
#ifndef CFC_Q8_ROW_MAX
#define CFC_Q8_ROW_MAX 256
#endif

/* Number of points in the tanh lookup table (1 KB of .rodata). */
#define CFC_TANH_LUT_N 256

/* 16-byte alignment for the int8 activation buffers the SIMD kernel loads. */
#if defined(_MSC_VER)
#define CFC_ALIGN16 __declspec(align(16))
#else
#define CFC_ALIGN16 __attribute__((aligned(16)))
#endif

/* cfc_cell_t.int8_mask bits: one bit per weight matrix, set when that matrix is
 * stored as int8 (and its `*_w8`/`*_ws` pointers are the valid ones). */
#define CFC_W8_BB   0x01u
#define CFC_W8_FF1  0x02u
#define CFC_W8_FF2  0x04u
#define CFC_W8_TA   0x08u
#define CFC_W8_TB   0x10u
#define CFC_W8_ALL  0x1Fu

typedef struct {
    int input_dim, hidden_dim, backbone_units;
    int use_retention;
    const float *bb_w, *bb_b;        /* backbone_units x (input_dim+hidden_dim), backbone_units */
    const float *ff1_w, *ff1_b;      /* hidden x backbone_units */
    const float *ff2_w, *ff2_b;
    const float *ta_w,  *ta_b;
    const float *tb_w,  *tb_b;
    const float *ret;                /* hidden, may be NULL */
    float ts;

    /* Per-layer int8 selection: bit n set -> that matrix is read from its
     * `*_w8` pointer with the per-output-row scales in `*_ws`, and the float
     * `*_w` pointer for it is ignored (may be NULL). A clear bit means the
     * opposite: the float matrix is used and the int8 pointers may be NULL.
     * Shapes are the same in both forms, only the element type differs:
     * bb_w8 is backbone_units*zu bytes, ff1_w8 is hidden_dim*zu bytes, and
     * bb_ws / ff1_ws hold one scale per output row.
     *
     * 0 means "all float32" and selects cfc_step(); any other value selects
     * cfc_step_q8(), which evaluates the int8 layers with a single int8
     * activation vector and the float layers with the plain float loop. */
    unsigned int8_mask;
    const int8_t *bb_w8, *ff1_w8, *ff2_w8, *ta_w8, *tb_w8;
    const float  *bb_ws, *ff1_ws, *ff2_ws, *ta_ws, *tb_ws;
} cfc_cell_t;

/* Zero the hidden state (h holds at least hidden_dim floats). */
void cfc_init_state(const cfc_cell_t *c, float *h);

/* One step: read input_dim floats from `in`; h_io is updated in place to the
 * new hidden state (hidden_dim floats). h_io is both the previous state and
 * the output, no extra copy is made.
 *
 * Does nothing when c->int8_mask != 0: a cell with any int8 matrix must be
 * evaluated with cfc_step_q8(), which also handles its float32 layers. */
void cfc_step(const cfc_cell_t *c, const float *in, float *h_io);

/* Same step for a cell with int8_mask != 0: int8 matrices use one int8
 * activation vector per layer input, float32 matrices keep the float dot.
 * `in` and `h_io` mean exactly what they do in cfc_step. */
void cfc_step_q8(const cfc_cell_t *c, const float *in, float *h_io);

/* Optional configuration self-check (call it once at init; cfc_step does not
 * depend on it):
 *     0   usable
 *    -1   NULL pointer or invalid dimension (in_dim<0, hidden_dim<=0,
 *         or a required weight pointer is NULL)
 *    -2   z dimension exceeds the compile-time limit CFC_MAX_Z
 *         (rebuild with -DCFC_MAX_Z=<larger>)
 *    -3   contradictory retention config (use_retention!=0 but ret==NULL):
 *         the gate cannot be evaluated, so cfc_step would silently run the
 *         baseline formula
 *    -4   ret != NULL while use_retention == 0: the parameters would be
 *         silently ignored, which usually means the loader forgot to set the
 *         flag. Not fatal for cfc_step (it keeps use_retention as the master
 *         switch), but worth logging.
 *    -5   a layer is marked int8 in int8_mask but its int8 matrix or scale
 *         pointer is NULL (or the float32 pointer of a float layer is missing
 *         while int8_mask is set, which cfc_step_q8 needs for that layer)
 */
int cfc_validate(const cfc_cell_t *c);

/* ---- int8 primitives (shared with the model wrapper) ------------------- */

/* Quantise the float vector x[0..n) to signed 8-bit and return the scale, with
 * x[j] ~= qx[j] * (*scale).  `qx` must have room for n bytes and should be
 * 16-byte aligned; the SIMD kernel reads it in 16-byte chunks.  Values are
 * clamped to [-127, 127] so a saturated entry can never overflow the int32
 * accumulator (the worst case, 127*127*256, stays far below INT32_MAX).
 *
 * Returns 1 when a scale was written, 0 when every entry was ~0 (`scale` is
 * then 0 and xscale would be meaningless). */
int cfc_q8_quantize(const float *x, int n, int8_t *qx, float *scale);

/* acc = sum(w[j] * qx[j]) in int32. `w` may be only byte aligned; `qx` should
 * be 16-byte aligned. Both must hold n bytes. Uses the ESP32-S3 SIMD kernel on
 * device and the portable scalar kernel everywhere else; the two produce
 * identical int32 results (cfc_q8_selftest verifies this at boot).
 *
 * One call is one accumulator session, i.e. one output row of an int8 layer. */
int32_t cfc_q8_dot(const int8_t *w, const int8_t *qx, int n);

/* tanh / sigmoid through the 256-point LUT (domain [-8, 8], saturating).
 * |fast - libm| < 1e-3 over the whole real line; see cfc_selftest(). */
float cfc_tanh_fast(float x);
float cfc_sigmoid_fast(float x);

/* Run the SIMD-vs-scalar comparison once. Returns 0 when the device kernel is
 * usable, -1 when it disagreed with the scalar reference (the engine then
 * permanently falls back to scalar). Safe to call repeatedly and from any
 * task; the first result wins. */
int cfc_q8_selftest(void);

/* 1 when the int8 dot runs on the ESP32-S3 SIMD kernel, 0 when it runs on the
 * portable scalar kernel (every non-Xtensa build, and any device where the
 * self-test disagreed). Call it after cfc_q8_selftest(). */
int cfc_q8_simd_enabled(void);

/* Numeric self-check, used by the PC parity harness and worth running once on a
 * bench board. NOT called at boot: it sweeps a few 10^4 transcendentals and
 * keeps ~1 KB of buffers, which does not belong on the small task that first
 * touches the model (the ESP-IDF Wi-Fi event loop task). cfc_q8_selftest(), the
 * only thing the boot path needs, is a few hundred bytes. Every field is a
 * measured worst case, not a tolerance. */
typedef struct {
    float tanh_max_err;      /* max |cfc_tanh_fast - tanhf| over [-16, 16] */
    float tanh_sat_err;      /* |cfc_tanh_fast(+-8) - tanhf(+-8)| */
    float sigmoid_max_err;   /* max |cfc_sigmoid_fast - 1/(1+exp(-x))| over [-20, 20] */
    float q8_dot_max_rel;    /* max |int8 W8A8 dot - f64 dot| / max|f64 dot| */
    int   q8_simd;           /* 1 when the SIMD kernel was selected and passed */
} cfc_selftest_t;

void cfc_selftest(cfc_selftest_t *out);

#ifdef __cplusplus
}
#endif

#endif /* CFC_ENGINE_H */
