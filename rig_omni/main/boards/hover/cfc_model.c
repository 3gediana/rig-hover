/*
 * cfc_model.c -- CfC weight blob storage, validation and inference wrapper.
 *
 * See cfc_model.h for the blob layout, the storage region and the atomicity
 * argument. This file owns the only code in the firmware that writes the
 * `assets` partition; it never writes outside the reserved tail region.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "cfc_engine.h"
#include "cfc_model.h"

static const char *TAG = "CfcModel";

/* ------------------------------------------------------------------ */
/* tuning                                                              */
/* ------------------------------------------------------------------ */

#define PARTITION_LABEL   "assets"
#define SB_A_OFF          0x0000u
#define SB_B_OFF          0x1000u
#define SLOT_OFF(i)       (0x2000u + (uint32_t)(i) * CFC_SLOT_MAX_BYTES)
#define FLASH_SECTOR      4096u
#define SB_MAGIC          0x53434643u /* 'CFCS' */
#define SB_BYTES          32u
#define CFG_MARGIN        0x1000u     /* keep 4 KB clear of the asset image */

/* Blobs at or below this size are loaded into internal SRAM instead of PSRAM;
 * see instance_load(). A float32 deployment blob (133 KB) is well above it, an
 * int8 one (41 KB) well below. */
#define CFC_SRAM_BLOB_MAX (64u * 1024u)

/* Field offsets inside the blob header (see cfc_model.h). */
#define H_MAGIC      0
#define H_VER        4
#define H_FLAGS      6
#define H_IN_DIM     8
#define H_CELL_IN    10
#define H_HID        12
#define H_BB         14
#define H_NHEADS     16
#define H_RESERVED   18
#define H_HEADDIM    20
#define H_HEADACT    36
#define H_LAYEROFF   52
#define H_LAYERLEN   116
#define H_MEAN       180
#define H_STD        1204
#define H_CLIPSIG    2228
#define H_TS         2232
#define H_CRC        2236
#define H_PREC       2240

#define FLAG_PROJ    CFC_FLAG_PROJ
#define FLAG_RET     CFC_FLAG_RET
#define FLAG_INT8    CFC_FLAG_INT8
#define FLAG_LAYERPREC CFC_FLAG_LAYERPREC

/* Layer ids. */
enum {
    L_BB_W = 0, L_BB_B,
    L_FF1_W, L_FF1_B,
    L_FF2_W, L_FF2_B,
    L_TA_W, L_TA_B,
    L_TB_W, L_TB_B,
    L_RET, L_PROJ_W, L_PROJ_B, L_HEADS,
    L_COUNT
};

typedef struct {
    uint32_t seq;
    uint32_t bytes[2];
    uint32_t crc[2];
    uint8_t  valid[2];
    uint8_t  active;
} superblock_t;

/* ------------------------------------------------------------------ */
/* state                                                               */
/* ------------------------------------------------------------------ */

static bool                 s_inited;
static bool                 s_storage_ok;
static char                 s_err[96] = "not initialized";
static const esp_partition_t *s_part;
static uint32_t             s_region_off;
static uint32_t             s_region_len;
static superblock_t         s_sb;

static cfc_model_t          s_model[2];   /* RAM instances, never both active */
static volatile int         s_active = -1;
static const cfc_model_t   *s_busy;       /* instance currently in cfc_step  */

/* Scratch for read-back verification, only used by the HTTP task. */
static uint8_t              s_chk[1024];

/* ------------------------------------------------------------------ */
/* little endian helpers                                               */
/* ------------------------------------------------------------------ */

static uint16_t rd_u16(const uint8_t *p)
{
    uint16_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static uint32_t rd_u32(const uint8_t *p)
{
    uint32_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static float rd_f32(const uint8_t *p)
{
    float v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static void wr_u32(uint8_t *p, uint32_t v)
{
    memcpy(p, &v, sizeof(v));
}

/* ------------------------------------------------------------------ */
/* CRC32 (IEEE 802.3, same polynomial/init as zlib.crc32)              */
/* ------------------------------------------------------------------ */

static uint32_t s_crc_tab[256];
static bool     s_crc_tab_ready;

static void crc32_tab_init(void)
{
    uint32_t i, j, c;
    for (i = 0; i < 256; i++) {
        c = i;
        for (j = 0; j < 8; j++) {
            c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        }
        s_crc_tab[i] = c;
    }
    s_crc_tab_ready = true;
}

uint32_t cfc_crc32(uint32_t crc, const void *data, size_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    uint32_t c;
    size_t i;

    if (!s_crc_tab_ready) {
        crc32_tab_init();
    }
    c = crc ^ 0xFFFFFFFFu;
    for (i = 0; i < len; i++) {
        c = s_crc_tab[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

/* CRC of a blob with its own crc32 field treated as zero. */
static uint32_t blob_crc(const uint8_t *blob, size_t len)
{
    uint32_t c = cfc_crc32(0, blob, H_CRC);
    uint8_t zero[4] = {0, 0, 0, 0};
    c = cfc_crc32(c, zero, sizeof(zero));
    if (len > H_CRC + 4u) {
        c = cfc_crc32(c, blob + H_CRC + 4u, len - (H_CRC + 4u));
    }
    return c;
}

/* ------------------------------------------------------------------ */
/* superblock                                                          */
/* ------------------------------------------------------------------ */

static void sb_decode(const uint8_t *raw, superblock_t *sb)
{
    int i;
    sb->seq = rd_u32(raw + 4);
    for (i = 0; i < 2; i++) {
        sb->bytes[i] = rd_u32(raw + 8 + 4 * i);
        sb->crc[i]   = rd_u32(raw + 16 + 4 * i);
        sb->valid[i] = raw[24 + i];
    }
    sb->active = raw[26];
}

static void sb_encode(const superblock_t *sb, uint8_t *raw)
{
    int i;
    memset(raw, 0xFF, SB_BYTES);
    wr_u32(raw + 0, SB_MAGIC);
    wr_u32(raw + 4, sb->seq);
    for (i = 0; i < 2; i++) {
        wr_u32(raw + 8 + 4 * i, sb->bytes[i]);
        wr_u32(raw + 16 + 4 * i, sb->crc[i]);
        raw[24 + i] = sb->valid[i];
    }
    raw[26] = (uint8_t)sb->active;
    wr_u32(raw + 28, cfc_crc32(0, raw, 28));
}

/* Read one superblock copy. Returns true when the copy is intact. */
static bool sb_read(uint32_t off, superblock_t *out)
{
    uint8_t raw[SB_BYTES];
    if (s_part == NULL) {
        return false;
    }
    if (esp_partition_read(s_part, s_region_off + off, raw, sizeof(raw)) != ESP_OK) {
        return false;
    }
    if (rd_u32(raw) != SB_MAGIC) {
        return false;
    }
    if (rd_u32(raw + 28) != cfc_crc32(0, raw, 28)) {
        return false;
    }
    sb_decode(raw, out);
    return true;
}

static void sb_load(void)
{
    superblock_t a, b;
    const bool oka = sb_read(SB_A_OFF, &a);
    const bool okb = sb_read(SB_B_OFF, &b);

    memset(&s_sb, 0, sizeof(s_sb));
    if (oka && okb) {
        s_sb = (a.seq >= b.seq) ? a : b;
    } else if (oka) {
        s_sb = a;
    } else if (okb) {
        s_sb = b;
    } else {
        s_sb.seq = 0;
    }
}

/* Write the superblock into the copy that is behind, then the other one. */
static int sb_commit(const superblock_t *sb)
{
    uint8_t raw[SB_BYTES];
    superblock_t a, b;
    const bool oka = sb_read(SB_A_OFF, &a);
    const bool okb = sb_read(SB_B_OFF, &b);
    uint32_t first;
    uint32_t second;

    if (!oka && !okb) {
        first = SB_A_OFF;
        second = SB_B_OFF;
    } else if (oka && okb) {
        first = (a.seq <= b.seq) ? SB_A_OFF : SB_B_OFF;
        second = (first == SB_A_OFF) ? SB_B_OFF : SB_A_OFF;
    } else if (oka) {
        first = SB_B_OFF;
        second = SB_A_OFF;
    } else {
        first = SB_A_OFF;
        second = SB_B_OFF;
    }

    sb_encode(sb, raw);
    /* Write the stale copy first: a power cut in between still leaves the
     * previous sequence number intact in the other copy. */
    if (esp_partition_erase_range(s_part, s_region_off + first, FLASH_SECTOR) != ESP_OK) {
        snprintf(s_err, sizeof(s_err), "erase superblock failed");
        return -1;
    }
    if (esp_partition_write(s_part, s_region_off + first, raw, SB_BYTES) != ESP_OK) {
        snprintf(s_err, sizeof(s_err), "write superblock failed");
        return -1;
    }
    if (esp_partition_erase_range(s_part, s_region_off + second, FLASH_SECTOR) != ESP_OK) {
        snprintf(s_err, sizeof(s_err), "erase superblock failed");
        return -1;
    }
    if (esp_partition_write(s_part, s_region_off + second, raw, SB_BYTES) != ESP_OK) {
        snprintf(s_err, sizeof(s_err), "write superblock failed");
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* init                                                                */
/* ------------------------------------------------------------------ */

static void set_err(const char *msg)
{
    snprintf(s_err, sizeof(s_err), "%s", msg);
}

bool cfc_model_init(void)
{
    uint8_t hdr[12];
    uint32_t used = 0;

    if (s_inited) {
        return s_storage_ok;
    }
    s_inited = true;
    if (!s_crc_tab_ready) {
        crc32_tab_init();
    }

    s_part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA,
                                      ESP_PARTITION_SUBTYPE_ANY,
                                      PARTITION_LABEL);
    if (s_part == NULL) {
        set_err("assets partition not found");
        ESP_LOGE(TAG, "%s", s_err);
        return false;
    }
    if (s_part->size <= CFC_STORE_REGION_SZ) {
        set_err("assets partition too small");
        ESP_LOGE(TAG, "%s", s_err);
        return false;
    }

    s_region_len = CFC_STORE_REGION_SZ;
    s_region_off = s_part->size - CFC_STORE_REGION_SZ;

    /* Refuse to touch the region while the packed asset image still reaches
     * into it. Header: u32 n_files | u32 checksum | u32 stored_len | table. */
    if (esp_partition_read(s_part, 0, hdr, sizeof(hdr)) == ESP_OK) {
        const uint32_t n_files = rd_u32(hdr + 0);
        const uint32_t stored  = rd_u32(hdr + 8);
        if (n_files > 0 && n_files < 4096u && stored < s_part->size) {
            used = 12u + stored;
        }
    }
    if (used > 0 && used + CFG_MARGIN > s_region_off) {
        set_err("asset image overlaps model region");
        ESP_LOGE(TAG, "%s (used=%u region=%u)", s_err, (unsigned)used,
                 (unsigned)s_region_off);
        return false;
    }

    ESP_LOGW(TAG, "init: stage=pre_sb_load");
    s_storage_ok = true;
    sb_load();
    ESP_LOGW(TAG, "init: stage=post_sb_load");
    set_err("ok");
    {
        /* Decide the int8 kernel once, loudly: a silent fall back to the scalar
         * kernel would look exactly like "quantisation did not help". This runs
         * on whichever task first touches the model (the ESP-IDF Wi-Fi event
         * loop task, ~4 KB of stack), so only the few-hundred-byte kernel probe
         * runs here; the full numeric sweep is cfc_selftest(), driven from the
         * PC harness. */
        /* This used to call cfc_q8_simd_enabled(), which runs the full SIMD
         * numeric sweep (7 lengths x 3 paths) and overflowed this task's
         * stack, crash-looping the robot. The sweep is a diagnostic: the
         * decision is taken lazily on the first inference, which runs on
         * cfc_task's stack instead of the boot task's. */
        ESP_LOGW(TAG, "int8 kernel: probe deferred to first inference");
    }
    ESP_LOGW(TAG, "model storage ok: part=%uB region_off=%u used=%u slots=%d%d",
             (unsigned)s_part->size, (unsigned)s_region_off, (unsigned)used,
             s_sb.valid[0], s_sb.valid[1]);
    return true;
}

bool cfc_model_storage_ok(void)
{
    cfc_model_init();
    return s_storage_ok;
}

const char *cfc_model_last_error(void)
{
    return s_err;
}

int cfc_model_slot_valid(int slot)
{
    if (slot < 0 || slot > 1) {
        return -1;
    }
    cfc_model_init();
    if (!s_storage_ok) {
        return 0;
    }
    return s_sb.valid[slot] ? 1 : 0;
}

uint32_t cfc_model_slot_crc(int slot)
{
    if (slot < 0 || slot > 1 || !cfc_model_slot_valid(slot)) {
        return 0;
    }
    return s_sb.crc[slot];
}

uint32_t cfc_model_slot_bytes(int slot)
{
    if (slot < 0 || slot > 1 || !cfc_model_slot_valid(slot)) {
        return 0;
    }
    return s_sb.bytes[slot];
}

/* Dimensions of a stored slot without loading the weights: read the ten bytes
 * of the header that matter. Keeps the HTTP task stack small. */
int cfc_model_slot_dims(int slot, int *input_dim, int *hidden_dim, int *n_heads)
{
    uint8_t raw[20];

    if (!cfc_model_slot_valid(slot) || s_sb.bytes[slot] < CFC_HDR_BYTES_LEGACY) {
        return -1;
    }
    if (esp_partition_read(s_part, s_region_off + SLOT_OFF(slot), raw, sizeof(raw)) !=
        ESP_OK) {
        return -1;
    }
    if (rd_u32(raw + H_MAGIC) != CFC_BLOB_MAGIC) {
        return -1;
    }
    if (input_dim != NULL) {
        *input_dim = rd_u16(raw + H_IN_DIM);
    }
    if (hidden_dim != NULL) {
        *hidden_dim = rd_u16(raw + H_HID);
    }
    if (n_heads != NULL) {
        *n_heads = rd_u16(raw + H_NHEADS);
    }
    return 0;
}

int cfc_model_active_slot(void)
{
    const int a = s_active;
    if (a < 0) {
        return -1;
    }
    return s_model[a].slot;
}

int cfc_model_runtime_flags(void)
{
    const int a = s_active;
    int f = 0;

    if (a < 0) {
        return 0;
    }
    if (s_model[a].use_int8) {
        f |= CFC_RT_INT8;
        if (cfc_q8_simd_enabled()) {
            f |= CFC_RT_SIMD;
        }
    }
    return f;
}

int cfc_model_blob_in_sram(void)
{
    const int a = s_active;
    if (a < 0) {
        return 0;
    }
    return s_model[a].in_sram ? 1 : 0;
}

/* ------------------------------------------------------------------ */
/* blob parsing                                                        */
/* ------------------------------------------------------------------ */

/* Validate one layer's byte length against the expected float count. */
static bool layer_ok(const uint8_t *blob, size_t len, int id, uint32_t expect_floats)
{
    const uint32_t off = rd_u32(blob + H_LAYEROFF + 4 * id);
    const uint32_t blen = rd_u32(blob + H_LAYERLEN + 4 * id);

    if (blen != expect_floats * 4u) {
        return false;
    }
    if (off % 4u != 0u) {
        return false;
    }
    if ((size_t)off + blen > len) {
        return false;
    }
    return true;
}

static const float *layer_ptr(const uint8_t *blob, int id)
{
    return (const float *)(const void *)(blob + rd_u32(blob + H_LAYEROFF + 4 * id));
}

/* Validate one int8 weight layer: `expect_elems` bytes of int8 (rounded up to 4)
 * followed by `out_dim` float32 scales, all inside the blob. */
static bool layer_ok_q8(const uint8_t *blob, size_t len, int id,
                        uint32_t expect_elems, uint32_t out_dim)
{
    const uint32_t off = rd_u32(blob + H_LAYEROFF + 4 * id);
    const uint32_t blen = rd_u32(blob + H_LAYERLEN + 4 * id);
    const uint32_t want = (expect_elems + 3u) & ~3u;

    if (blen != want) {
        return false;
    }
    if (off % 4u != 0u) {
        return false;
    }
    if ((size_t)off + (size_t)blen + (size_t)out_dim * 4u > len) {
        return false;
    }
    return true;
}

static const int8_t *layer_w8(const uint8_t *blob, int id)
{
    return (const int8_t *)(const void *)(blob + rd_u32(blob + H_LAYEROFF + 4 * id));
}

/* Scale array of an int8 layer: right after its `layer_len` int8 bytes. Both
 * offsets are multiples of 4, so the cast is aligned. */
static const float *layer_wscale(const uint8_t *blob, int id)
{
    const uint32_t off = rd_u32(blob + H_LAYEROFF + 4 * id);
    const uint32_t blen = rd_u32(blob + H_LAYERLEN + 4 * id);
    return (const float *)(const void *)(blob + off + blen);
}

/* Layers that may be stored as int8. Everything else -- the biases, the
 * retention gate and the output heads -- is float32 in every blob: quantising
 * `heads` costs 2.8x on the head outputs (it is 3% of the parameters, and the
 * most sensitive one) for 8% of the volume, so the format does not offer it. */
static bool layer_can_be_int8(int id)
{
    return id == L_BB_W || id == L_FF1_W || id == L_FF2_W || id == L_TA_W ||
           id == L_TB_W || id == L_PROJ_W;
}

/* Validate one weight layer in whichever form layer_prec selected. */
static bool weight_layer_ok(const uint8_t *blob, size_t len, int id,
                            uint32_t elems, uint32_t out_dim, uint8_t prec)
{
    return (prec == CFC_PREC_INT8) ? layer_ok_q8(blob, len, id, elems, out_dim)
                                   : layer_ok(blob, len, id, elems);
}

/* Resolve the storage precision of all 16 layer ids into `prec`.
 *
 * bit3 (CFC_FLAG_LAYERPREC) clear -> the blob predates the table and its bytes
 *                at H_PREC belong to the payload, so they are not read at all:
 *                bit2 set means the six weight matrices are int8, bit2 clear
 *                means all float32.
 * bit3 set       -> use the table, require it to agree with bit2, and reject any
 *                int8 marking the engine cannot honour rather than silently
 *                running that layer as float32. */
static bool resolve_layer_prec(const uint8_t *blob, uint16_t flags,
                               uint8_t prec[CFC_MAX_LAYERS])
{
    const bool has_table = (flags & FLAG_LAYERPREC) != 0;
    const bool is_int8 = (flags & FLAG_INT8) != 0;
    int i, any = 0;

    /* Every caller reads all 16 entries, and the legacy branch below only fills
     * the six weight ones, so start from "all float32". */
    memset(prec, 0, CFC_MAX_LAYERS);
    if (!has_table) {
        if (is_int8) {
            prec[L_BB_W] = prec[L_FF1_W] = prec[L_FF2_W] = CFC_PREC_INT8;
            prec[L_TA_W] = prec[L_TB_W] = prec[L_PROJ_W] = CFC_PREC_INT8;
        }
        return true;
    }
    for (i = 0; i < CFC_MAX_LAYERS; i++) {
        prec[i] = blob[H_PREC + i];        /* caller checked len >= CFC_HDR_BYTES */
        if (prec[i] != CFC_PREC_F32 && prec[i] != CFC_PREC_INT8) {
            set_err("bad layer_prec value");
            return false;
        }
        any |= prec[i];
    }
    if ((any != 0) != is_int8) {
        set_err("layer_prec contradicts flags bit2");
        return false;
    }
    for (i = 0; i < CFC_MAX_LAYERS; i++) {
        if (prec[i] != CFC_PREC_INT8) {
            continue;
        }
        if (!layer_can_be_int8(i)) {
            set_err("layer_prec marks a float32-only layer as int8");
            return false;
        }
        if (rd_u32(blob + H_LAYERLEN + 4 * i) == 0) {
            set_err("layer_prec marks an absent layer");
            return false;
        }
    }
    return true;
}

/* Parse and validate `blob` into `m`. The blob memory is not owned yet. */
static int blob_parse(const uint8_t *blob, size_t len, cfc_model_t *m)
{
    const uint32_t magic = rd_u32(blob + H_MAGIC);
    const uint16_t ver   = rd_u16(blob + H_VER);
    const uint16_t flags = rd_u16(blob + H_FLAGS);
    uint8_t prec[CFC_MAX_LAYERS];
    int in_dim, cell_in, hid, bu, n_heads, zu, i;
    uint32_t head_floats = 0, head_bias = 0;

    memset(m, 0, sizeof(*m));

    /* A blob that predates layer_prec[] is only as long as the old header. */
    if (len < ((flags & CFC_FLAG_LAYERPREC) ? CFC_HDR_BYTES
                                            : CFC_HDR_BYTES_LEGACY)) {
        set_err("blob shorter than header");
        return -1;
    }
    if (magic != CFC_BLOB_MAGIC) {
        set_err("bad magic");
        return -1;
    }
    if (ver != CFC_BLOB_VER) {
        set_err("unsupported version");
        return -1;
    }

    in_dim  = rd_u16(blob + H_IN_DIM);
    cell_in = rd_u16(blob + H_CELL_IN);
    hid     = rd_u16(blob + H_HID);
    bu      = rd_u16(blob + H_BB);
    n_heads = rd_u16(blob + H_NHEADS);

    if (in_dim <= 0 || in_dim > CFC_MAX_INPUT_DIM) {
        set_err("bad input_dim");
        return -1;
    }
    if (cell_in <= 0 || cell_in > CFC_MAX_INPUT_DIM) {
        set_err("bad cell_in_dim");
        return -1;
    }
    if (hid <= 0 || hid > CFC_MAX_HIDDEN_DIM) {
        set_err("bad hidden_dim");
        return -1;
    }
    if (bu < 0 || bu > CFC_MAX_BACKBONE) {
        set_err("bad backbone_units");
        return -1;
    }
    if (n_heads <= 0 || n_heads > CFC_MAX_HEADS) {
        set_err("bad n_heads");
        return -1;
    }

    zu = (bu > 0) ? bu : (cell_in + hid);

    if (!resolve_layer_prec(blob, flags, prec)) {
        return -1;
    }

    /* Per-layer shape checks: an int8 weight matrix shrinks to one byte per
     * element and drags a float32 scale array behind it; biases, `ret` and
     * `heads` keep the float32 length in every case. */
    if (!weight_layer_ok(blob, len, L_FF1_W, (uint32_t)(hid * zu), (uint32_t)hid,
                         prec[L_FF1_W]) ||
        !weight_layer_ok(blob, len, L_FF2_W, (uint32_t)(hid * zu), (uint32_t)hid,
                         prec[L_FF2_W]) ||
        !weight_layer_ok(blob, len, L_TA_W, (uint32_t)(hid * zu), (uint32_t)hid,
                         prec[L_TA_W]) ||
        !weight_layer_ok(blob, len, L_TB_W, (uint32_t)(hid * zu), (uint32_t)hid,
                         prec[L_TB_W])) {
        set_err("cell weight layer size mismatch");
        return -1;
    }
    if (!layer_ok(blob, len, L_FF1_B, (uint32_t)hid) ||
        !layer_ok(blob, len, L_FF2_B, (uint32_t)hid) ||
        !layer_ok(blob, len, L_TA_B, (uint32_t)hid) ||
        !layer_ok(blob, len, L_TB_B, (uint32_t)hid)) {
        set_err("cell bias layer size mismatch");
        return -1;
    }
    if (bu > 0) {
        const uint32_t bwe = (uint32_t)(bu * (cell_in + hid));
        if (!weight_layer_ok(blob, len, L_BB_W, bwe, (uint32_t)bu,
                             prec[L_BB_W])) {
            set_err("backbone weight layer size mismatch");
            return -1;
        }
        if (!layer_ok(blob, len, L_BB_B, (uint32_t)bu)) {
            set_err("backbone bias layer size mismatch");
            return -1;
        }
    }
    if ((flags & FLAG_RET) && !layer_ok(blob, len, L_RET, (uint32_t)hid)) {
        set_err("retention layer size mismatch");
        return -1;
    }
    if (flags & FLAG_PROJ) {
        const uint32_t pwe = (uint32_t)(cell_in * in_dim);
        if (!weight_layer_ok(blob, len, L_PROJ_W, pwe, (uint32_t)cell_in,
                             prec[L_PROJ_W])) {
            set_err("projection weight layer size mismatch");
            return -1;
        }
        if (!layer_ok(blob, len, L_PROJ_B, (uint32_t)cell_in)) {
            set_err("projection bias layer size mismatch");
            return -1;
        }
    }
    if (!(flags & FLAG_PROJ) && cell_in != in_dim) {
        /* Without a projection the cell reads the raw observation directly. */
        set_err("cell_in_dim must equal input_dim without a projection");
        return -1;
    }

    for (i = 0; i < n_heads; i++) {
        const int d = rd_u16(blob + H_HEADDIM + 2 * i);
        const int a = rd_u16(blob + H_HEADACT + 2 * i);
        if (d <= 0 || d > CFC_MAX_HEAD_DIM) {
            set_err("bad head_dim");
            return -1;
        }
        if (a < 0 || a > 2) {
            set_err("bad head_act");
            return -1;
        }
        m->head_dim[i] = d;
        m->head_act[i] = a;
        head_floats += (uint32_t)(d * hid);
        head_bias += (uint32_t)d;
    }
    if (!layer_ok(blob, len, L_HEADS, head_floats + head_bias)) {
        set_err("head layer size mismatch");
        return -1;
    }

    m->input_dim      = in_dim;
    m->cell_in_dim    = cell_in;
    m->hidden_dim     = hid;
    m->backbone_units = bu;
    m->n_heads        = n_heads;
    m->has_proj       = (flags & FLAG_PROJ) != 0;
    m->has_ret        = (flags & FLAG_RET) != 0;
    m->input_mean     = (const float *)(const void *)(blob + H_MEAN);
    m->input_std      = (const float *)(const void *)(blob + H_STD);
    m->clip_sigma     = rd_f32(blob + H_CLIPSIG);
    m->ts             = rd_f32(blob + H_TS);

    if (!(m->clip_sigma > 0.5f && m->clip_sigma < 16.0f)) {
        m->clip_sigma = 4.0f;
    }
    if (!(m->ts > 1e-6f && m->ts < 1e6f)) {
        m->ts = 1.0f;
    }
    /* mean/std must be finite; a zero/NaN sigma would divide by zero. */
    for (i = 0; i < in_dim; i++) {
        const float mu = m->input_mean[i];
        const float sd = m->input_std[i];
        if (!isfinite(mu) || !isfinite(sd) || sd < 1e-6f) {
            set_err("bad input statistics");
            return -1;
        }
    }

    m->cell.input_dim      = cell_in;
    m->cell.hidden_dim     = hid;
    m->cell.backbone_units = bu;
    m->cell.use_retention  = m->has_ret ? 1 : 0;
    m->cell.bb_b = (bu > 0) ? layer_ptr(blob, L_BB_B) : NULL;
    m->cell.ff1_b = layer_ptr(blob, L_FF1_B);
    m->cell.ff2_b = layer_ptr(blob, L_FF2_B);
    m->cell.ta_b  = layer_ptr(blob, L_TA_B);
    m->cell.tb_b  = layer_ptr(blob, L_TB_B);
    m->cell.ret   = m->has_ret ? layer_ptr(blob, L_RET) : NULL;
    m->cell.ts    = m->ts;

    /* Bind each weight layer in the form layer_prec selected: an int8 layer
     * leaves its float32 pointer NULL, a float32 layer leaves the int8 pair
     * NULL, and cfc_step_q8() picks per row from whichever one is set. */
    if (bu > 0) {
        if (prec[L_BB_W] == CFC_PREC_INT8) {
            m->cell.bb_w8 = layer_w8(blob, L_BB_W);
            m->cell.bb_ws = layer_wscale(blob, L_BB_W);
            m->cell.int8_mask |= CFC_W8_BB;
        } else {
            m->cell.bb_w = layer_ptr(blob, L_BB_W);
        }
    }
    if (prec[L_FF1_W] == CFC_PREC_INT8) {
        m->cell.ff1_w8 = layer_w8(blob, L_FF1_W);
        m->cell.ff1_ws = layer_wscale(blob, L_FF1_W);
        m->cell.int8_mask |= CFC_W8_FF1;
    } else {
        m->cell.ff1_w = layer_ptr(blob, L_FF1_W);
    }
    if (prec[L_FF2_W] == CFC_PREC_INT8) {
        m->cell.ff2_w8 = layer_w8(blob, L_FF2_W);
        m->cell.ff2_ws = layer_wscale(blob, L_FF2_W);
        m->cell.int8_mask |= CFC_W8_FF2;
    } else {
        m->cell.ff2_w = layer_ptr(blob, L_FF2_W);
    }
    if (prec[L_TA_W] == CFC_PREC_INT8) {
        m->cell.ta_w8 = layer_w8(blob, L_TA_W);
        m->cell.ta_ws = layer_wscale(blob, L_TA_W);
        m->cell.int8_mask |= CFC_W8_TA;
    } else {
        m->cell.ta_w = layer_ptr(blob, L_TA_W);
    }
    if (prec[L_TB_W] == CFC_PREC_INT8) {
        m->cell.tb_w8 = layer_w8(blob, L_TB_W);
        m->cell.tb_ws = layer_wscale(blob, L_TB_W);
        m->cell.int8_mask |= CFC_W8_TB;
    } else {
        m->cell.tb_w = layer_ptr(blob, L_TB_W);
    }

    if (cfc_validate(&m->cell) != 0) {
        set_err("cfc engine rejected the configuration");
        return -1;
    }

    if (m->has_proj) {
        if (prec[L_PROJ_W] == CFC_PREC_INT8) {
            m->proj_w8 = layer_w8(blob, L_PROJ_W);
            m->proj_ws = layer_wscale(blob, L_PROJ_W);
        } else {
            m->proj_w = layer_ptr(blob, L_PROJ_W);
        }
        m->proj_b = layer_ptr(blob, L_PROJ_B);
    }
    m->use_int8 = (m->cell.int8_mask != 0 || m->proj_w8 != NULL);

    {
        const float *hp = layer_ptr(blob, L_HEADS);
        const float *bp = hp + head_floats;
        for (i = 0; i < n_heads; i++) {
            m->head_w[i] = hp;
            hp += (size_t)m->head_dim[i] * (size_t)hid;
        }
        for (i = 0; i < n_heads; i++) {
            m->head_b[i] = bp;
            bp += m->head_dim[i];
        }
    }

    m->param_count = (int)(head_floats + head_bias) + hid * zu * 4 + hid * 4;
    if (bu > 0) {
        m->param_count += bu * (cell_in + hid) + bu;
    }
    if (m->has_proj) {
        m->param_count += cell_in * in_dim + cell_in;
    }
    set_err("ok");
    return 0;
}

/* ------------------------------------------------------------------ */
/* RAM instance management                                             */
/* ------------------------------------------------------------------ */

/* Read a slot into RAM instance pick and validate it. */
static int instance_load(int slot, int pick)
{
    cfc_model_t *m = &s_model[pick];
    uint32_t bytes;
    uint32_t crc;
    uint8_t *buf;
    int in_sram = 0;
    esp_err_t err;

    if (!s_storage_ok) {
        set_err("storage not available");
        return -1;
    }
    if (!s_sb.valid[slot]) {
        set_err("slot empty");
        return -1;
    }
    bytes = s_sb.bytes[slot];
    crc = s_sb.crc[slot];
    if (bytes < CFC_HDR_BYTES_LEGACY || bytes > CFC_SLOT_MAX_BYTES) {
        set_err("stored size out of range");
        return -1;
    }

    /* Never rewrite an instance a reader is inside of. The inference takes
     * 1-2 ms at 50 Hz, so a short sleep-based wait is enough; never spin, the
     * caller shares core 0 with the 500 Hz loop. */
    if (s_busy == m) {
        const int64_t deadline = esp_timer_get_time() + 200 * 1000;
        while (s_busy == m && esp_timer_get_time() < deadline) {
            vTaskDelay(1);
        }
        if (s_busy == m) {
            set_err("model busy");
            return -1;
        }
    }

    buf = NULL;
    if (bytes <= CFC_SRAM_BLOB_MAX) {
        /* Internal SRAM first: the 64 KB L1 data cache is flushed by the
         * camera, LVGL and Wi-Fi between inference frames, so an int8 blob read
         * from PSRAM costs on the order of a millisecond of cache misses per
         * step while the same bytes in SRAM are essentially free. Aligned to 16
         * so the SIMD kernel can load weight rows without staging. */
        buf = (uint8_t *)heap_caps_aligned_alloc(16, bytes,
                                                 MALLOC_CAP_INTERNAL |
                                                 MALLOC_CAP_8BIT);
        in_sram = (buf != NULL);
    } else {
        in_sram = 0;
    }
    if (buf == NULL) {
        buf = (uint8_t *)heap_caps_aligned_alloc(16, bytes,
                                                 MALLOC_CAP_SPIRAM |
                                                 MALLOC_CAP_8BIT);
    }
    if (buf == NULL) {
        set_err("out of memory for the model");
        return -1;
    }
    err = esp_partition_read(s_part, s_region_off + SLOT_OFF(slot), buf, bytes);
    if (err != ESP_OK) {
        heap_caps_free(buf);
        set_err("flash read failed");
        return -1;
    }
    if (blob_crc(buf, bytes) != crc) {
        heap_caps_free(buf);
        set_err("stored blob crc mismatch");
        return -1;
    }
    {
        cfc_model_t tmp;
        if (blob_parse(buf, bytes, &tmp) != 0) {
            heap_caps_free(buf);
            return -1;
        }
        tmp.blob = buf;
        tmp.blob_len = bytes;
        tmp.crc = crc;
        tmp.slot = slot;
        tmp.in_sram = in_sram;
        /* Old blob of this instance: safe to free, we are not busy on it. */
        if (m->blob != NULL) {
            heap_caps_free(m->blob);
        }
        *m = tmp;
    }
    return 0;
}

int cfc_model_select(int slot)
{
    int other;
    const int cur = s_active;

    if (slot < 0 || slot > 1) {
        set_err("slot must be 0 or 1");
        return -1;
    }
    cfc_model_init();
    if (!s_storage_ok) {
        return -1;
    }
    if (cur >= 0 && s_model[cur].slot == slot) {
        return 0; /* already active */
    }
    other = (cur == 0) ? 1 : 0;
    if (instance_load(slot, other) != 0) {
        return -1;
    }
    s_active = other;
    ESP_LOGW(TAG, "active slot=%d instance=%d in=%d hid=%d heads=%d",
             slot, other, s_model[other].input_dim, s_model[other].hidden_dim,
             s_model[other].n_heads);
    return 0;
}

void cfc_model_unload(void)
{
    s_active = -1;
}

const cfc_model_t *cfc_model_acquire(void)
{
    const int a = s_active;
    if (a < 0) {
        return NULL;
    }
    s_busy = &s_model[a];
    return &s_model[a];
}

void cfc_model_release(void)
{
    s_busy = NULL;
}

int cfc_model_max_input_dim(void)
{
    return CFC_MAX_INPUT_DIM;
}

/* ------------------------------------------------------------------ */
/* upload                                                              */
/* ------------------------------------------------------------------ */

int cfc_model_upload(int slot, const void *blob, size_t len,
                     char *err, size_t err_len)
{
    cfc_model_t probe;
    uint32_t crc;
    uint32_t erase_len;
    uint32_t off;
    size_t done;

    if (err != NULL && err_len > 0) {
        err[0] = '\0';
    }
    if (slot < 0 || slot > 1) {
        set_err("slot must be 0 or 1");
        goto fail;
    }
    if (blob == NULL || len < CFC_HDR_BYTES_LEGACY || len > CFC_SLOT_MAX_BYTES) {
        set_err("blob size out of range");
        goto fail;
    }
    cfc_model_init();
    if (!s_storage_ok) {
        goto fail;
    }

    crc = blob_crc((const uint8_t *)blob, len);
    if (crc != rd_u32((const uint8_t *)blob + H_CRC)) {
        set_err("crc32 mismatch");
        goto fail;
    }
    if (blob_parse((const uint8_t *)blob, len, &probe) != 0) {
        goto fail; /* set_err already explains which field */
    }

    /* Write the slot. The active instance was already copied to PSRAM, so a
     * flash write here cannot disturb the running model. */
    off = s_region_off + SLOT_OFF(slot);
    erase_len = (uint32_t)((len + FLASH_SECTOR - 1u) / FLASH_SECTOR) * FLASH_SECTOR;
    if (erase_len > CFC_SLOT_MAX_BYTES) {
        set_err("blob does not fit the slot");
        goto fail;
    }
    if (esp_partition_erase_range(s_part, off, erase_len) != ESP_OK) {
        set_err("flash erase failed");
        goto fail;
    }
    done = 0;
    while (done < len) {
        size_t chunk = len - done;
        if (chunk > 4096u) {
            chunk = 4096u;
        }
        if (esp_partition_write(s_part, off + (uint32_t)done,
                                (const uint8_t *)blob + done, chunk) != ESP_OK) {
            set_err("flash write failed");
            goto fail;
        }
        done += chunk;
    }

    /* Read back and re-validate: only a byte-identical copy counts. */
    {
        uint32_t back_crc = 0;
        size_t got = 0;
        while (got < len) {
            size_t chunk = len - got;
            if (chunk > sizeof(s_chk)) {
                chunk = sizeof(s_chk);
            }
            if (esp_partition_read(s_part, off + (uint32_t)got, s_chk, chunk) != ESP_OK) {
                set_err("flash read-back failed");
                goto fail;
            }
            /* blob_crc() hashes the blob with the crc32 field excluded, so the
             * read-back has to cover exactly the same bytes. Hashing the raw
             * stored image here would include the stored crc value and always
             * mismatch. */
            for (size_t b = 0; b < 4u; b++) {
                const size_t abs = (size_t)H_CRC + b;
                if (abs >= got && abs < got + chunk) {
                    s_chk[abs - got] = 0;
                }
            }
            back_crc = cfc_crc32(back_crc, s_chk, chunk);
            got += chunk;
        }
        if (back_crc != crc) {
            set_err("read-back crc mismatch");
            goto fail;
        }
    }

    {
        superblock_t next = s_sb;
        next.seq += 1;
        next.bytes[slot] = (uint32_t)len;
        next.crc[slot] = crc;
        next.valid[slot] = 1;
        next.active = (s_active >= 0) ? (uint8_t)s_model[s_active].slot : (uint8_t)slot;
        if (sb_commit(&next) != 0) {
            goto fail;
        }
        s_sb = next;
    }

    /* Swap only if this slot is the one in use, otherwise leave the running
     * model alone. Reload is forced: uploading into the active slot must pick
     * up the new weights, not keep the previous RAM copy. */
    if (s_active < 0 || s_model[s_active].slot == slot) {
        cfc_model_unload();          /* pointer only, the blob stays valid */
        if (cfc_model_select(slot) != 0) {
            goto fail;
        }
    }

    ESP_LOGW(TAG, "slot%d committed: %u bytes crc=%08x in=%d hid=%d heads=%d",
             slot, (unsigned)len, (unsigned)crc, probe.input_dim,
             probe.hidden_dim, probe.n_heads);
    set_err("ok");
    if (err != NULL && err_len > 0) {
        snprintf(err, err_len, "ok");
    }
    return 0;

fail:
    if (err != NULL && err_len > 0) {
        snprintf(err, err_len, "%s", s_err);
    }
    ESP_LOGE(TAG, "slot%d upload rejected: %s", slot, s_err);
    return -1;
}

/* ------------------------------------------------------------------ */
/* inference                                                           */
/* ------------------------------------------------------------------ */

static float silu(float v)
{
    /* v / (1 + exp(-v)); for very negative v this tends to 0 without
     * overflowing because expf(+|v|) saturates to +inf -> 1/inf = 0. */
    return v / (1.0f + expf(-v));
}

/* Segment timings of the last cfc_model_forward(), in microseconds:
 * [0] normalisation, [1] projection, [2] cfc_step, [3] heads. Written by the
 * inference task only, read by /status; a torn read costs one misleading
 * sample, never a wrong control output. */
static volatile int64_t s_split_us[4];

void cfc_model_last_split(int64_t *out)
{
    int i;
    if (out == NULL) {
        return;
    }
    for (i = 0; i < 4; i++) {
        out[i] = s_split_us[i];
    }
}

void cfc_model_reset_state(const cfc_model_t *m, float *h)
{
    if (m == NULL || h == NULL) {
        return;
    }
    cfc_init_state(&m->cell, h);
}

int cfc_model_forward(const cfc_model_t *m, const float *obs, float *h,
                      float *out, int out_cap)
{
    float x[CFC_MAX_INPUT_DIM];
    float zp[CFC_MAX_INPUT_DIM];
    int i, j, k;
    int64_t t_norm, t_proj, t_cell, t_end;

    if (m == NULL || obs == NULL || h == NULL || out == NULL) {
        return -1;
    }
    if (out_cap < m->n_heads * CFC_MAX_HEAD_DIM) {
        return -1;
    }

    t_norm = esp_timer_get_time();

    /* 1. domain guard + normalisation. */
    for (j = 0; j < m->input_dim; j++) {
        const float mu = m->input_mean[j];
        const float sd = m->input_std[j];
        float v = obs[j];
        const float lo = mu - m->clip_sigma * sd;
        const float hi = mu + m->clip_sigma * sd;
        if (!isfinite(v)) {
            v = mu;
        }
        if (v < lo) {
            v = lo;
        } else if (v > hi) {
            v = hi;
        }
        x[j] = (v - mu) / sd;
    }

    /* 2. optional projection + SiLU. */
    t_proj = esp_timer_get_time();
    if (m->has_proj) {
        if (m->proj_w8 != NULL) {
            CFC_ALIGN16 int8_t qx[CFC_MAX_INPUT_DIM + 16];
            float sx = 0.0f;
            cfc_q8_quantize(x, m->input_dim, qx, &sx);
            for (i = 0; i < m->cell_in_dim; i++) {
                float acc = m->proj_b[i];
                if (sx != 0.0f) {
                    acc += (float)cfc_q8_dot(
                               m->proj_w8 + (size_t)i * (size_t)m->input_dim, qx,
                               m->input_dim) * m->proj_ws[i] * sx;
                }
                /* SiLU keeps libm even on the int8 path: its relative error
                 * would be multiplied by |v|, and 64 expf calls per step are
                 * cheap next to the weight traffic. */
                zp[i] = silu(acc);
            }
        } else {
            for (i = 0; i < m->cell_in_dim; i++) {
                const float *w = m->proj_w + (size_t)i * (size_t)m->input_dim;
                float acc = m->proj_b[i];
                for (j = 0; j < m->input_dim; j++) {
                    acc += w[j] * x[j];
                }
                zp[i] = silu(acc);
            }
        }
    }

    t_cell = esp_timer_get_time();

    /* cfc_step_q8() also runs a cell whose int8_mask is partial: any float32
     * layer in it keeps the float loop. */
    if (m->cell.int8_mask != 0) {
        cfc_step_q8(&m->cell, m->has_proj ? zp : x, h);
    } else {
        cfc_step(&m->cell, m->has_proj ? zp : x, h);
    }

    t_end = esp_timer_get_time();

    /* 3. heads. */
    for (i = 0; i < m->n_heads; i++) {
        const int d = m->head_dim[i];
        for (k = 0; k < d; k++) {
            const float *w = m->head_w[i] + (size_t)k * (size_t)m->hidden_dim;
            float acc = m->head_b[i][k];
            for (j = 0; j < m->hidden_dim; j++) {
                acc += w[j] * h[j];
            }
            switch (m->head_act[i]) {
            case 1:
                acc = tanhf(acc);
                break;
            case 2:
                /* softmax is applied by the consumer over the whole head;
                 * store the logits so argmax and softmax both work. */
                break;
            default:
                break;
            }
            out[i * CFC_MAX_HEAD_DIM + k] = acc;
        }
        /* Mark unused slots of this head so the consumer cannot read stale
         * values if it indexes past head_dim. */
        for (k = d; k < CFC_MAX_HEAD_DIM; k++) {
            out[i * CFC_MAX_HEAD_DIM + k] = 0.0f;
        }
    }
    {
        const int64_t t_head = esp_timer_get_time();
        s_split_us[0] = t_proj - t_norm;
        s_split_us[1] = t_cell - t_proj;
        s_split_us[2] = t_end - t_cell;
        s_split_us[3] = t_head - t_end;
    }
    return m->n_heads;
}