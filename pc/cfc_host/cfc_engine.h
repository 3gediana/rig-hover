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
 * THREAD SAFETY
 * -------------
 * cfc_step uses no static or global data and is reentrant. A single
 * cfc_cell_t can be read by several tasks concurrently, but each task needs
 * its own h buffer.
 *
 * BUILD
 * -----
 *     cl /c /O2 /fp:precise /W3 cfc_engine.c        (MSVC, C mode)
 *     gcc -std=c99 -O2 -c cfc_engine.c              (C99)
 * Depends only on <math.h> (tanhf, expf) and <string.h> (memset, size_t).
 */

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
} cfc_cell_t;

/* Zero the hidden state (h holds at least hidden_dim floats). */
void cfc_init_state(const cfc_cell_t *c, float *h);

/* One step: read input_dim floats from `in`; h_io is updated in place to the
 * new hidden state (hidden_dim floats). h_io is both the previous state and
 * the output, no extra copy is made. */
void cfc_step(const cfc_cell_t *c, const float *in, float *h_io);

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
 */
int cfc_validate(const cfc_cell_t *c);

#ifdef __cplusplus
}
#endif

#endif /* CFC_ENGINE_H */
