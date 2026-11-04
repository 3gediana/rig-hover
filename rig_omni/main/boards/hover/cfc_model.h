#ifndef CFC_MODEL_H
#define CFC_MODEL_H
/*
 * cfc_model.h -- self-describing CfC weight blob: parsing, validation,
 * storage and atomic slot switching.
 *
 * STORAGE
 * -------
 * The firmware has no SPIFFS filesystem: the `assets` partition holds a
 * memory-mapped packed asset image (header at offset 0: u32 n_files,
 * u32 checksum, u32 stored_len; payload starts at offset 12) consumed by
 * assets.cc / esp_mmap_assets. Measured image size on this board: 4,974,631 B
 * of an 8 MB partition, so the last 3.3 MB of the partition is free.
 *
 * Model blobs are stored with the raw esp_partition API in a reserved region
 * at the end of that partition, which leaves the asset image untouched:
 *
 *   region_base = part_size - CFC_STORE_REGION_SIZE     (CFC_STORE_REGION_SIZE)
 *     +0x0000  superblock A   (4 KB sector, guards against power loss)
 *     +0x1000  superblock B   (4 KB sector, second copy)
 *     +0x2000  slot 0 blob    (<= CFC_SLOT_MAX_BYTES)
 *     +0x42000 slot 1 blob    (<= CFC_SLOT_MAX_BYTES)
 *
 * On init the region is only accepted when the asset image really ends before
 * region_base; otherwise model storage stays disabled and every model
 * endpoint reports the reason. Nothing is ever written outside the region.
 *
 * ATOMICITY
 * ---------
 * A blob is never used while it is being written:
 *   1. POST /model streams into a PSRAM staging buffer, no flash access yet.
 *   2. magic / version / dimensions / per-layer sizes / CRC32 are validated.
 *   3. the staging buffer is written to its slot, read back and re-validated.
 *   4. the superblock copy with the higher sequence number is rewritten
 *      (B first, then A) so a power cut always leaves one valid copy.
 *   5. the blob is loaded into the *inactive* RAM instance and the active
 *      index is flipped. The instance that is currently being read by
 *      cfc_task is never written to and never freed.
 * Any failure at any step leaves the running model completely unaffected.
 *
 * BLOB FORMAT (little endian, all floats are IEEE-754 binary32)
 * ------------------------------------------------------------
 *   off   size  field
 *   0     4     magic        'CFCM' = 0x4D434643
 *   4     2     ver          1
 *   6     2     flags        bit0 = projection layer present
 *                            bit1 = retention gate present
 *                            bit2 = at least one weight matrix is int8
 *                            bit3 = the layer_prec[] table is present
 *   8     2     input_dim    raw observation length (e.g. 106)
 *   10    2     cell_in_dim  CfC cell input (projection output, else input_dim)
 *   12    2     hidden_dim
 *   14    2     backbone_units  (0 = backbone-less form of cfc_engine)
 *   16    2     n_heads      <= 8
 *   18    2     reserved     0
 *   20    16    head_dim[8]  u16 each
 *   36    16    head_act[8]  u16 each: 0=linear 1=tanh 2=softmax
 *   52    64    layer_off[16] u32 each, byte offset from blob start, %4==0
 *   116   64    layer_len[16] u32 each, byte length, %4==0
 *   180   1024  input_mean[256]
 *   1204  1024  input_std[256]
 *   2228  4     clip_sigma   domain guard, 4.0 when 0
 *   2232  4     ts           CfC timescale, 1.0 when 0
 *   2236  4     crc32        over the whole blob with these 4 bytes zeroed
 *   2240  16    layer_prec[16] u8 each: 0=float32 1=int8, one entry per layer id
 *   2256  ...   payload
 *
 * layer_prec was appended at the end of the 2240-byte header on purpose: every
 * field above keeps its offset, `layer_off` is absolute, and a blob written by
 * the previous exporter (whose payload starts at 2240) therefore still parses.
 * Nothing outside this file may assume the header length; CFC_HDR_BYTES is the
 * single source of truth and the exporter's --check-offsets cross-checks it.
 *
 * Layer ids (offsets are absolute, so the payload may be reordered):
 *   0  bb_w    backbone_units * (cell_in_dim + hidden_dim)
 *   1  bb_b    backbone_units
 *   2  ff1_w   hidden_dim * zu
 *   3  ff1_b   hidden_dim
 *   4  ff2_w, 5 ff2_b, 6 ta_w, 7 ta_b, 8 tb_w, 9 tb_b   (same shape as ff1)
 *   10 ret     hidden_dim                      optional, flags bit1
 *   11 proj_w  cell_in_dim * input_dim         optional, flags bit0
 *   12 proj_b  cell_in_dim                     optional, flags bit0
 *   13 heads   hidden_dim * sum(head_dim) floats, then sum(head_dim) biases
 *   with zu = backbone_units when backbone_units > 0 else cell_in_dim+hidden_dim.
 *
 * PER-LAYER PRECISION (layer_prec, flags bit2)
 * --------------------------------------------
 * `layer_prec[id]` selects the storage form of that layer, independently for
 * every id:
 *
 *   0  float32: `layer_len` = element count * 4, the floats start at `layer_off`.
 *   1  int8:    per-output-channel symmetric,
 *               W[i][:] ~= q[i][:] * scale[i],  scale[i] = max|W[i][:]| / 127.
 *               `layer_len` = the element count rounded up to 4 (one byte per
 *               element), those int8 bytes start at `layer_off`, and the float32
 *               `scale[out_dim]` array follows immediately after them.
 *
 * Only the six weight matrices (bb_w, ff1_w, ff2_w, ta_w, tb_w, proj_w) may be
 * marked int8; the biases, `ret` and the `heads` layer are always float32 (they
 * are added after dequantisation, carry no multiply, and `heads` produces the
 * 15 values the controller actually consumes, so quantising it is the single
 * most damaging choice available -- measured 2.8x worse on the head outputs for
 * 8% less volume). A layer marked int8 whose id cannot be int8, or an int8
 * marking without the matching layer, is rejected rather than reinterpreted.
 *
 * BACKWARD COMPATIBILITY
 * ----------------------
 * The layer_prec table is only consulted when flags bit3 says it exists. That
 * bit is what makes the older blobs readable: in a blob whose payload starts at
 * 2240, the 16 bytes at 2240 are the first 16 bytes of bb_w's payload, not a
 * zeroed table, so a reader that trusted the offset alone would see garbage. A
 * blob with bit3 clear is therefore resolved by the old rule -- bit2 set means
 * the six weight matrices are int8 and everything else float32, bit2 clear means
 * all float32 -- and is only required to be CFC_HDR_BYTES_LEGACY long.
 *
 * This preserves the *layout*, not the arithmetic: such a blob is evaluated by
 * the current int8 kernel, which quantises activations to 8 bits, so its output
 * differs from the old 15-bit kernel by the tanh-head error quoted in
 * cfc_engine.h. Re-export to get the layer table in writing.
 *
 * Everything else in the blob is unchanged, so a reader that ignores bit2 (and
 * therefore rejects an int8 blob on the length check) sees a well-formed but
 * unsupported file rather than misinterpreting bytes.
 *
 * INFERENCE (int8)
 * ----------------
 * Same pipeline; cfc_step_q8() replaces cfc_step() for the cell (it evaluates
 * any float32 layer of a mixed cell with the float loop), and the projection
 * uses cfc_q8_dot(). The float32 arrays of the int8 layers are then unused.
 *
 * INFERENCE
 * ---------
 *   x = (clip(obs, mean +- clip_sigma*std) - mean) / std
 *   z = SiLU(proj_w @ x + proj_b)            (when the projection exists)
 *   cfc_step(cell, z, h)                      (cfc_engine.h, CONTRACT 5.3)
 *   out[i] = act_i(head_i @ h + b_i)
 *   head 0 head_yaw: tanh, caller scales by head_max degrees
 *   head 1 vyaw    : tanh, caller scales by vyaw_max deg/s
 *   head 2 mode    : softmax(4), caller takes argmax
 *   head 3 emote   : softmax(8), caller takes argmax
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "cfc_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CFC_BLOB_MAGIC      0x4D434643u /* 'CFCM' */
#define CFC_BLOB_VER        1
#define CFC_HDR_BYTES       2256
#define CFC_MAX_LAYERS      16
#define CFC_MAX_HEADS       8
#define CFC_MAX_HEAD_DIM    16
#define CFC_MAX_INPUT_DIM   256
#define CFC_MAX_HIDDEN_DIM  128
#define CFC_MAX_BACKBONE    256

/* Blob header flags (H_FLAGS). */
#define CFC_FLAG_PROJ       0x0001u
#define CFC_FLAG_RET        0x0002u
#define CFC_FLAG_INT8       0x0004u
/* bit3: the header carries the layer_prec[] table. Its absence is what makes a
 * blob from the exporter that predates layer_prec readable: those bytes are not
 * a zeroed table there, they are the first 16 bytes of bb_w's payload, so the
 * reader must not look at them at all. */
#define CFC_FLAG_LAYERPREC  0x0008u

/* Header length before layer_prec[] existed. A blob without
 * CFC_FLAG_LAYERPREC is only required to be this long. */
#define CFC_HDR_BYTES_LEGACY 2240

/* layer_prec[] values (H_PREC, one byte per layer id). */
#define CFC_PREC_F32        0
#define CFC_PREC_INT8       1

#define CFC_SLOT_MAX_BYTES  (256 * 1024)
#define CFC_STORE_REGION_SZ (0x88000) /* 544 KB: 2 superblocks + 2 slots */

/* cfc_model_runtime_flags() bits. */
#define CFC_RT_INT8         0x01 /* active model uses int8 weight matrices */
#define CFC_RT_SIMD         0x02 /* int8 dot runs on the ESP32-S3 SIMD kernel */

/* Parsed model. All weight pointers point into `blob`, which lives in PSRAM. */
typedef struct {
    int       input_dim;
    int       cell_in_dim;
    int       hidden_dim;
    int       backbone_units;
    int       n_heads;
    int       head_dim[CFC_MAX_HEADS];
    int       head_act[CFC_MAX_HEADS];
    bool      has_proj;
    bool      has_ret;
    bool      use_int8;
    float     clip_sigma;
    float     ts;

    const float *input_mean; /* input_dim valid entries */
    const float *input_std;

    cfc_cell_t cell;         /* ready for cfc_step() / cfc_step_q8() */
    const float *proj_w;     /* cell_in_dim x input_dim, or NULL */
    const float *proj_b;     /* cell_in_dim, or NULL */
    const int8_t *proj_w8;   /* int8 form, or NULL */
    const float  *proj_ws;   /* cell_in_dim scales, or NULL */
    const float *head_w[CFC_MAX_HEADS];
    const float *head_b[CFC_MAX_HEADS];

    void     *blob;          /* PSRAM, owned by this module */
    size_t    blob_len;
    uint32_t  crc;
    int       slot;
    int       in_sram;       /* 1 when `blob` sits in internal SRAM, 0 in PSRAM */
    int       param_count;   /* floats of real weights (telemetry only) */
} cfc_model_t;

/* ---- storage lifecycle ------------------------------------------------ */

/* Locate the partition and reserve the storage region. Cheap, no PSRAM use.
 * Returns false when model storage is unavailable (no partition, no free
 * space) -- every other entry point then degrades to "no model". */
bool cfc_model_init(void);

bool cfc_model_storage_ok(void);
const char *cfc_model_last_error(void);

/* Slot bookkeeping, no inference involved. */
int  cfc_model_slot_valid(int slot);      /* 1 valid, 0 empty/invalid, -1 bad arg */
uint32_t cfc_model_slot_crc(int slot);    /* 0 when empty */
int  cfc_model_slot_dims(int slot, int *input_dim, int *hidden_dim, int *n_heads);
uint32_t cfc_model_slot_bytes(int slot);
int  cfc_model_active_slot(void);         /* -1 when no model is loaded */

/* How the active model is being evaluated: CFC_RT_INT8 when it carries int8
 * weight matrices, CFC_RT_SIMD when those run on the ESP32-S3 int8 SIMD
 * kernel. Returns 0 when no model is loaded. Reported by /status so a field
 * measurement never has to assume which path produced the number. */
int  cfc_model_runtime_flags(void);

/* 1 when the active model's weights are in internal SRAM, 0 when they had to go
 * to PSRAM (or no model is loaded). Weight traffic is the largest term of a
 * step, and internal SRAM is one to two orders of magnitude faster to read than
 * PSRAM, so a timing result is only interpretable next to this bit. */
int  cfc_model_blob_in_sram(void);

/* Upload a complete blob (already fully received) to `slot`. Validates,
 * writes flash, verifies by read-back, updates the superblock and, when the
 * slot is the selected one, swaps the RAM instance. Atomic: on any error the
 * running model is untouched. */
int cfc_model_upload(int slot, const void *blob, size_t len,
                     char *err, size_t err_len);

/* Load `slot` and make it the active model. Returns 0 on success. */
int cfc_model_select(int slot);

/* Drop the active model (keeps the flash copy). */
void cfc_model_unload(void);

/* The model cfc_task must use, or NULL. acquire/release bracket a single
 * inference so a concurrent load can never free a blob mid-computation. */
const cfc_model_t *cfc_model_acquire(void);
void cfc_model_release(void);

/* ---- inference -------------------------------------------------------- */

/* One CfC step. `obs` has model->input_dim floats, `h` has hidden_dim floats
 * and is updated in place. `out` must hold n_heads * CFC_MAX_HEAD_DIM floats
 * (head i, output k lives at out[i*CFC_MAX_HEAD_DIM + k]); rows past
 * head_dim[i] are zeroed. The head activation is already applied except for
 * softmax heads, which keep their logits so the caller can argmax or softmax.
 * Returns the number of heads written, or -1. */
int cfc_model_forward(const cfc_model_t *m, const float *obs, float *h,
                      float *out, int out_cap);

/* Zero the hidden state. */
void cfc_model_reset_state(const cfc_model_t *m, float *h);

/* Segment timings of the last cfc_model_forward(), in microseconds:
 * [0] normalisation, [1] projection, [2] cfc_step, [3] heads. Diagnostic only;
 * a torn read across two frames costs one misleading sample. */
void cfc_model_last_split(int64_t *out);

/* Highest input_dim any loaded/known model needs, for caller-side buffers. */
int cfc_model_max_input_dim(void);

/* ---- helpers usable from the HTTP layer ------------------------------- */

uint32_t cfc_crc32(uint32_t crc, const void *data, size_t len); /* zlib flavour */

#ifdef __cplusplus
}
#endif

#endif /* CFC_MODEL_H */
