/*
 * host_blob_main.c -- 用**固件原版** cfc_model.c 在 PC 上加载 blob 并推理。
 *
 * 子命令：
 *   info   <blob>                          上传 -> 激活 -> 把解析出来的每个字段
 *                                          和每层指针相对 blob 的偏移打成 JSON
 *   run    <blob> <frames.bin> <out.bin>   逐帧推理；frames 是纯 float32
 *                                          (T, input_dim)，out 是 (T, n_heads*16)
 *                                          与 cfc_model_forward 的输出布局一致
 *   reject <blob>                          期望被拒绝（返回 0 = 确实拒绝了）
 *
 * 退出码：0 成功；1 用法错误；2 文件读不了；3 upload 被接受（reject 模式失败）；
 *         4 推理失败。
 */

/* fopen 在这类一次性工具里够用；必须在任何标准头之前定义，才能压掉 MSVC 的 C4996 */
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfc_model.h"

#define OUT_STRIDE (CFC_MAX_HEADS * CFC_MAX_HEAD_DIM)

static uint8_t *read_file(const char *path, size_t *len_out, size_t limit)
{
    FILE *f;
    long sz;
    uint8_t *buf;

    f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    sz = ftell(f);
    if (sz <= 0 || (size_t)sz > limit) {
        fprintf(stderr, "bad size %ld for %s (limit %lu)\n", sz, path,
                (unsigned long)limit);
        fclose(f);
        return NULL;
    }
    rewind(f);
    buf = (uint8_t *)malloc((size_t)sz);
    if (buf == NULL) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fprintf(stderr, "short read %s\n", path);
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *len_out = (size_t)sz;
    return buf;
}

/* 上传到 slot 0。成功返回 0 并打印上传日志到 stderr。 */
static int upload(const char *blob_path)
{
    size_t len = 0;
    uint8_t *buf = read_file(blob_path, &len, CFC_SLOT_MAX_BYTES);
    char err[128] = "";
    int rc;

    if (buf == NULL) {
        return 2;
    }
    rc = cfc_model_upload(0, buf, len, err, sizeof(err));
    fprintf(stderr, "upload: len=%lu rc=%d err=\"%s\"\n",
            (unsigned long)len, rc, err);
    free(buf);
    return rc == 0 ? 0 : 5;
}

static void dump_offsets_json(const cfc_model_t *m)
{
    const uint8_t *base = (const uint8_t *)m->blob;
    int i;

#define OFF(p) ((p) != NULL ? (long)((const uint8_t *)(p) - base) : -1L)
    printf("{\n");
    printf("  \"input_dim\": %d,\n", m->input_dim);
    printf("  \"cell_in_dim\": %d,\n", m->cell_in_dim);
    printf("  \"hidden_dim\": %d,\n", m->hidden_dim);
    printf("  \"backbone_units\": %d,\n", m->backbone_units);
    printf("  \"n_heads\": %d,\n", m->n_heads);
    printf("  \"has_proj\": %d,\n", m->has_proj ? 1 : 0);
    printf("  \"has_ret\": %d,\n", m->has_ret ? 1 : 0);
    printf("  \"clip_sigma\": %.9g,\n", (double)m->clip_sigma);
    printf("  \"ts\": %.9g,\n", (double)m->ts);
    printf("  \"crc\": %lu,\n", (unsigned long)m->crc);
    printf("  \"blob_len\": %lu,\n", (unsigned long)m->blob_len);
    printf("  \"param_count\": %d,\n", m->param_count);
    printf("  \"use_int8\": %d,\n", m->use_int8 ? 1 : 0);
    printf("  \"int8_mask\": %u,\n", (unsigned)m->cell.int8_mask);
    printf("  \"in_sram\": %d,\n", m->in_sram ? 1 : 0);
    printf("  \"slot\": %d,\n", m->slot);
    /* 逐层精度（固件解析后的视角）：权重层看 int8 指针是否为 NULL，其余层恒 0 */
    {
        const int prec[14] = {
            m->cell.bb_w8 != NULL, 0,
            m->cell.ff1_w8 != NULL, 0,
            m->cell.ff2_w8 != NULL, 0,
            m->cell.ta_w8 != NULL, 0,
            m->cell.tb_w8 != NULL, 0,
            0, m->proj_w8 != NULL, 0, 0
        };
        printf("  \"layer_prec\": [");
        for (i = 0; i < CFC_MAX_LAYERS; i++) {
            printf("%s%d", i ? ", " : "", i < 14 ? prec[i] : 0);
        }
        printf("],\n");
    }
    printf("  \"head_dim\": [");
    for (i = 0; i < m->n_heads; i++) {
        printf("%s%d", i ? ", " : "", m->head_dim[i]);
    }
    printf("],\n");
    printf("  \"head_act\": [");
    for (i = 0; i < m->n_heads; i++) {
        printf("%s%d", i ? ", " : "", m->head_act[i]);
    }
    printf("],\n");
    printf("  \"off\": {\"bb_w\": %ld, \"bb_b\": %ld, \"ff1_w\": %ld, \"ff1_b\": %ld,\n",
           OFF(m->cell.bb_w), OFF(m->cell.bb_b), OFF(m->cell.ff1_w), OFF(m->cell.ff1_b));
    printf("          \"ff2_w\": %ld, \"ff2_b\": %ld, \"ta_w\": %ld, \"ta_b\": %ld,\n",
           OFF(m->cell.ff2_w), OFF(m->cell.ff2_b), OFF(m->cell.ta_w), OFF(m->cell.ta_b));
    printf("          \"tb_w\": %ld, \"tb_b\": %ld, \"ret\": %ld,\n",
           OFF(m->cell.tb_w), OFF(m->cell.tb_b), OFF(m->cell.ret));
    printf("          \"proj_w\": %ld, \"proj_b\": %ld,\n",
           OFF(m->proj_w), OFF(m->proj_b));
    printf("          \"input_mean\": %ld, \"input_std\": %ld",
           OFF(m->input_mean), OFF(m->input_std));
    for (i = 0; i < m->n_heads; i++) {
        printf(",\n          \"head_w_%d\": %ld, \"head_b_%d\": %ld",
               i, OFF(m->head_w[i]), i, OFF(m->head_b[i]));
    }
    printf("}");            /* 关掉 "off" 对象 */
    if (m->use_int8) {
        int t;
        /* int8 权重指针 + 紧随其后的 float32 scale 指针，都相对 blob 起点 */
        static const char *names[6] = {"bb_w", "ff1_w", "ff2_w", "ta_w",
                                       "tb_w", "proj_w"};
        const int8_t *w8[6];
        const float *ws[6];
#define OFF8(p) ((p) != NULL ? (long)((const uint8_t *)(const void *)(p) - base) : -1L)
        w8[0] = m->cell.bb_w8;   ws[0] = m->cell.bb_ws;
        w8[1] = m->cell.ff1_w8;  ws[1] = m->cell.ff1_ws;
        w8[2] = m->cell.ff2_w8;  ws[2] = m->cell.ff2_ws;
        w8[3] = m->cell.ta_w8;   ws[3] = m->cell.ta_ws;
        w8[4] = m->cell.tb_w8;   ws[4] = m->cell.tb_ws;
        w8[5] = m->proj_w8;      ws[5] = m->proj_ws;
        printf(",\n  \"off8\": {");
        for (t = 0; t < 6; t++) {
            printf("%s\"%s\": %ld, \"%s_s\": %ld", t ? ", " : "",
                   names[t], OFF8(w8[t]), names[t], OFF8(ws[t]));
        }
        printf("}");        /* 关掉 "off8" 对象 */
#undef OFF8
    }
    printf("\n}\n");        /* 关掉根对象 */
#undef OFF
}

static int cmd_info(const char *blob_path)
{
    const cfc_model_t *m;
    int rc = upload(blob_path);
    if (rc != 0) {
        return rc;
    }
    m = cfc_model_acquire();
    if (m == NULL) {
        fprintf(stderr, "no active model after upload\n");
        return 4;
    }
    dump_offsets_json(m);
    cfc_model_release();
    return 0;
}

static int cmd_run(const char *blob_path, const char *frames_path, const char *out_path)
{
    const cfc_model_t *m;
    size_t flen = 0, n_frames, i;
    uint8_t *fdata;
    float *h;
    float out[OUT_STRIDE];
    FILE *fo;
    double maxabs = 0.0;
    int in_dim, rc;

    rc = upload(blob_path);
    if (rc != 0) {
        return rc;
    }
    m = cfc_model_acquire();
    if (m == NULL) {
        fprintf(stderr, "no active model after upload\n");
        return 4;
    }
    in_dim = m->input_dim;
    if (in_dim <= 0) {
        cfc_model_release();
        return 4;
    }
    fdata = read_file(frames_path, &flen, (size_t)64 * 1024 * 1024);
    if (fdata == NULL) {
        cfc_model_release();
        return 2;
    }
    if (flen % (4u * (size_t)in_dim) != 0) {
        fprintf(stderr, "frames file %lu bytes is not a multiple of %d floats\n",
                (unsigned long)flen, in_dim);
        free(fdata);
        cfc_model_release();
        return 2;
    }
    n_frames = flen / (4u * (size_t)in_dim);

    h = (float *)malloc((size_t)m->hidden_dim * sizeof(float));
    if (h == NULL) {
        free(fdata);
        cfc_model_release();
        return 4;
    }
    cfc_model_reset_state(m, h);   /* 每条序列（会话）开头必须清零 */

    fo = fopen(out_path, "wb");
    if (fo == NULL) {
        fprintf(stderr, "cannot write %s\n", out_path);
        free(h);
        free(fdata);
        cfc_model_release();
        return 2;
    }
    for (i = 0; i < n_frames; i++) {
        const float *obs = ((const float *)fdata) + i * (size_t)in_dim;
        int n;
        int k;
        memset(out, 0, sizeof(out));       /* 没被写的槽位保持 0，别把上一帧的残留写出去 */
        n = cfc_model_forward(m, obs, h, out, OUT_STRIDE);
        if (n <= 0) {
            fprintf(stderr, "cfc_model_forward failed at frame %lu\n",
                    (unsigned long)i);
            fclose(fo);
            free(h);
            free(fdata);
            cfc_model_release();
            return 4;
        }
        /* 每帧只写 n_heads*CFC_MAX_HEAD_DIM 个 float（消费方按这个跨度读） */
        if (fwrite(out, sizeof(float), (size_t)(n * CFC_MAX_HEAD_DIM), fo) !=
            (size_t)(n * CFC_MAX_HEAD_DIM)) {
            fprintf(stderr, "write failed\n");
            fclose(fo);
            free(h);
            free(fdata);
            cfc_model_release();
            return 2;
        }
        for (k = 0; k < n * CFC_MAX_HEAD_DIM; k++) {
            double a = (double)out[k];
            if (a < 0) {
                a = -a;
            }
            if (a > maxabs) {
                maxabs = a;
            }
        }
    }
    fclose(fo);
    printf("{\"frames\": %lu, \"n_heads\": %d, \"input_dim\": %d, \"hidden_dim\": %d, "
           "\"max_abs_out\": %.9g, \"h0\": %.9g}\n",
           (unsigned long)n_frames, m->n_heads, in_dim, m->hidden_dim, maxabs,
           (double)h[0]);
    free(h);
    free(fdata);
    cfc_model_release();
    return 0;
}

static int cmd_reject(const char *blob_path)
{
    size_t len = 0;
    uint8_t *buf = read_file(blob_path, &len, CFC_SLOT_MAX_BYTES);
    char err[128] = "";
    int rc;

    if (buf == NULL) {
        return 2;
    }
    rc = cfc_model_upload(0, buf, len, err, sizeof(err));
    free(buf);
    if (rc == 0) {
        fprintf(stderr, "upload ACCEPTED a blob that must be rejected\n");
        return 3;
    }
    printf("{\"rejected\": 1, \"err\": \"%s\"}\n", err);
    return 0;
}

static int cmd_selftest(void)
{
    cfc_selftest_t st;

    cfc_selftest(&st);
    printf("{\"tanh_max_err\": %.9g, \"tanh_sat_err\": %.9g, "
           "\"sigmoid_max_err\": %.9g, \"q8_dot_max_rel\": %.9g, "
           "\"q8_simd\": %d}\n",
           (double)st.tanh_max_err, (double)st.tanh_sat_err,
           (double)st.sigmoid_max_err, (double)st.q8_dot_max_rel, st.q8_simd);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && strcmp(argv[1], "selftest") == 0) {
        return cmd_selftest();
    }
    if (argc < 3) {
        fprintf(stderr,
                "usage:\n"
                "  %s info   <blob>\n"
                "  %s run    <blob> <frames.bin> <out.bin>\n"
                "  %s reject <blob>\n"
                "  %s selftest\n", argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }
    if (!cfc_model_init()) {
        fprintf(stderr, "cfc_model_init failed: %s\n", cfc_model_last_error());
        return 4;
    }
    if (strcmp(argv[1], "info") == 0) {
        return cmd_info(argv[2]);
    }
    if (strcmp(argv[1], "run") == 0) {
        if (argc < 5) {
            fprintf(stderr, "run needs <blob> <frames.bin> <out.bin>\n");
            return 1;
        }
        return cmd_run(argv[2], argv[3], argv[4]);
    }
    if (strcmp(argv[1], "reject") == 0) {
        return cmd_reject(argv[2]);
    }
    fprintf(stderr, "unknown command %s\n", argv[1]);
    return 1;
}
