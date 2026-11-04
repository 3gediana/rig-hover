/*
 * test_engine.c -- PC-side test driver for the C CfC engine (NOT firmware).
 *
 * It contains no parsing logic of its own: weights, initial state and the
 * input sequence all come from testdata.bin written by Python (little-endian,
 * contiguous float32 arrays, layout below), and the results go straight back
 * to out_c.bin for Python to read. On the C side only compute + copy remain,
 * so a parity failure never has to be blamed on two different parsers.
 *
 * Usage:
 *     test_engine.exe [testdata.bin] [out_c.bin] [--bench N]
 *
 * testdata.bin layout (all little-endian, contiguous float32):
 *     char    magic[8] = "CFCBIN01"
 *     int32   n_steps, input_dim, hidden_dim, backbone_units, use_retention
 *     float32 ts
 *     float32 bb_w[bu * (in + hid)]      // empty when bu == 0
 *     float32 bb_b[bu]
 *     float32 ff1_w[hid * zu], ff1_b[hid]
 *     float32 ff2_w[hid * zu], ff2_b[hid]
 *     float32 ta_w [hid * zu], ta_b [hid]
 *     float32 tb_w [hid * zu], tb_b [hid]
 *     float32 ret[hid]                   // always present; ignored when use_retention==0
 *     float32 h0[hid]
 *     float32 inputs[n_steps * in]
 *   where zu = (bu > 0) ? bu : (in + hid).
 *
 * out_c.bin layout:
 *     char    magic[8] = "CFCOUT01"
 *     int32   n_steps, hidden_dim
 *     float32 hs[n_steps * hid]           // state after each step
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "cfc_engine.h"

#if defined(_WIN32)
#include <windows.h>
static double now_seconds(void)
{
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
}
#else
#include <time.h>
static double now_seconds(void)
{
    return (double)clock() / (double)CLOCKS_PER_SEC;
}
#endif

static void die(const char *msg)
{
    fprintf(stderr, "test_engine: %s\n", msg);
    exit(2);
}

/* ------------------------------------------------------------------ */
/* Engine self-test (no data file needed): validates the fail-safe      */
/* paths and one hand-computed numeric step.                            */
/* ------------------------------------------------------------------ */
#define ST_CHECK(cond, msg)                                            \
    do {                                                               \
        if (!(cond)) {                                                 \
            printf("SELFTEST_FAIL=%s\n", (msg));                       \
            ++fails;                                                   \
        }                                                              \
    } while (0)

static int run_selftest(void)
{
    /* in=3, hid=2, bu=0  ->  zu = 5 */
    float w1[2 * 5] = {0.0f}, b1[2] = {0.0f, 0.0f};
    float w2[2 * 5] = {0.0f}, b2[2] = {0.0f, 0.0f};
    float w3[2 * 5] = {0.0f}, b3[2] = {0.0f, 0.0f};
    float w4[2 * 5] = {0.0f}, b4[2] = {0.0f, 0.0f};
    float ret[2] = {-1.0e4f, -1.0e4f}; /* sigmoid -> exactly 0 */
    float h[2], ref[2], big_in[CFC_MAX_Z];
    const float in[3] = {1.0f, 2.0f, 3.0f};
    cfc_cell_t c;
    int fails = 0;
    int i;

    for (i = 0; i < CFC_MAX_Z; ++i) {
        big_in[i] = 1.0f;
    }
    memset(&c, 0, sizeof(c));
    c.input_dim = 3;
    c.hidden_dim = 2;
    c.backbone_units = 0;
    c.use_retention = 0;
    c.ff1_w = w1; c.ff1_b = b1;
    c.ff2_w = w2; c.ff2_b = b2;
    c.ta_w = w3;  c.ta_b = b3;
    c.tb_w = w4;  c.tb_b = b4;
    c.ret = NULL;
    c.ts = 1.0f;

    ST_CHECK(cfc_validate(&c) == 0, "valid config rejected");

    /* 1. init_state zeroes a dirty state */
    h[0] = 1.5f; h[1] = -2.5f;
    cfc_init_state(&c, h);
    ST_CHECK(h[0] == 0.0f && h[1] == 0.0f, "init_state did not zero");

    /* 2. all-zero weights: tanh(0)=0, t=0.5 -> h_new = 0 */
    h[0] = 0.7f; h[1] = -0.3f;
    cfc_step(&c, in, h);
    ST_CHECK(h[0] == 0.0f && h[1] == 0.0f, "zero-weight step is not 0");

    /* 3. hand-computed step: ff1_b = 1 -> h_new = tanh(1)*0.5 = 0.38079708 */
    b1[0] = 1.0f; b1[1] = 1.0f;
    h[0] = 0.0f; h[1] = 0.0f;
    cfc_step(&c, in, h);
    ST_CHECK(fabsf(h[0] - 0.38079708f) < 1.0e-6f && fabsf(h[1] - 0.38079708f) < 1.0e-6f,
             "hand-computed step mismatch");

    /* 4. retention enabled with lambda == 0 is bit-identical to disabled */
    b1[0] = 0.25f; b1[1] = -0.75f; b3[0] = 0.5f; b4[1] = -1.25f;
    h[0] = 0.5f; h[1] = 0.25f;
    cfc_step(&c, in, h);
    memcpy(ref, h, sizeof(ref));
    c.use_retention = 1;
    c.ret = ret;
    h[0] = 0.5f; h[1] = 0.25f;
    cfc_step(&c, in, h);
    ST_CHECK(memcmp(ref, h, sizeof(ref)) == 0, "lambda=0 is not bit-identical to no-retention");

    /* 5. contradictory retention config (ret == NULL) is reported */
    c.ret = NULL;
    ST_CHECK(cfc_validate(&c) == -3, "use_retention without ret not rejected");
    c.use_retention = 0;

    /* 5b. ret set while the master switch is off is reported (loader mismatch) */
    c.ret = ret;
    ST_CHECK(cfc_validate(&c) == -4, "ret without use_retention not reported");
    c.ret = NULL;

    /* 6. NULL input leaves the state untouched */
    h[0] = 1.5f; h[1] = -1.5f;
    cfc_step(&c, NULL, h);
    ST_CHECK(h[0] == 1.5f && h[1] == -1.5f, "NULL input modified the state");

    /* 7. z dimension above CFC_MAX_Z is rejected and the state is untouched */
    c.input_dim = CFC_MAX_Z;
    c.hidden_dim = 1;
    c.backbone_units = 0;
    ST_CHECK(cfc_validate(&c) == -2, "oversized z not rejected");
    h[0] = 1.5f;
    cfc_step(&c, big_in, h);
    ST_CHECK(h[0] == 1.5f, "oversized step modified the state");

    printf("SELFTEST=%d\n", fails);
    printf("CFC_MAX_Z=%d\n", CFC_MAX_Z);
    return (fails == 0) ? 0 : 1;
}

static unsigned char *read_file(const char *path, size_t *len_out)
{
    FILE *f = fopen(path, "rb");
    long n;
    unsigned char *buf;
    size_t got;

    if (f == NULL) {
        fprintf(stderr, "test_engine: cannot open %s\n", path);
        exit(2);
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        die("fseek failed");
    }
    n = ftell(f);
    if (n < 0) {
        die("ftell failed");
    }
    rewind(f);
    buf = (unsigned char *)malloc((size_t)n + 1);
    if (buf == NULL) {
        die("malloc failed");
    }
    got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) {
        die("short fread");
    }
    *len_out = (size_t)n;
    return buf;
}

static void write_file(const char *path, const char *magic, int32_t a, int32_t b,
                       const float *data, size_t nfloats)
{
    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        fprintf(stderr, "test_engine: cannot write %s\n", path);
        exit(2);
    }
    if (fwrite(magic, 1, 8, f) != 8 || fwrite(&a, 4, 1, f) != 1 || fwrite(&b, 4, 1, f) != 1 ||
        (nfloats > 0 && fwrite(data, sizeof(float), nfloats, f) != nfloats)) {
        fclose(f);
        die("write of output file failed");
    }
    fclose(f);
}

int main(int argc, char **argv)
{
    const char *in_path = "testdata.bin";
    const char *out_path = "out_c.bin";
    int bench_steps = 0;
    unsigned char *buf;
    size_t len, need;
    const unsigned char *p;
    int32_t n_steps, in_dim, hid, bu, use_ret;
    int zu, t, i, rep, k;
    float ts;
    const float *bb_w, *bb_b, *ff1_w, *ff1_b, *ff2_w, *ff2_b;
    const float *ta_w, *ta_b, *tb_w, *tb_b, *ret, *h0, *inputs;
    cfc_cell_t cell;
    float *h = NULL;
    float *hs = NULL;
    int rc;
    int arg;

    if (sizeof(float) != 4 || sizeof(int32_t) != 4) {
        die("float/int32_t are not 4 bytes on this platform");
    }

    for (arg = 1; arg < argc; ++arg) {
        if (strcmp(argv[arg], "--selftest") == 0) {
            return run_selftest(); /* engine self-test, no data file needed */
        }
    }

    if (argc > 1) {
        in_path = argv[1];
    }
    if (argc > 2) {
        out_path = argv[2];
    }
    for (arg = 3; arg < argc; ++arg) {
        if (strcmp(argv[arg], "--bench") == 0 && arg + 1 < argc) {
            bench_steps = atoi(argv[++arg]);
        } else {
            fprintf(stderr, "test_engine: unknown argument %s\n", argv[arg]);
            exit(2);
        }
    }

    buf = read_file(in_path, &len);
    if (len < 8 + 5 * 4 + 4 || memcmp(buf, "CFCBIN01", 8) != 0) {
        die("bad magic or truncated testdata.bin");
    }
    p = buf + 8;
    memcpy(&n_steps, p, 4); p += 4;
    memcpy(&in_dim, p, 4);  p += 4;
    memcpy(&hid, p, 4);     p += 4;
    memcpy(&bu, p, 4);      p += 4;
    memcpy(&use_ret, p, 4); p += 4;
    memcpy(&ts, p, 4);      p += 4;

    if (n_steps < 1 || in_dim < 1 || hid < 1 || bu < 0) {
        die("invalid dimensions");
    }
    zu = (bu > 0) ? (int)bu : (int)(in_dim + hid);

    /* Check the file length before slicing pointers: a mismatch dies here
     * instead of reading out of bounds. */
    need = 8 + 5 * 4 + 4;
    need += (size_t)((int64_t)bu * (in_dim + hid) + bu + 4 * (int64_t)hid * zu +
                     4 * (int64_t)hid + hid + hid + (int64_t)n_steps * in_dim) * sizeof(float);
    if (len != need) {
        fprintf(stderr, "test_engine: file length %llu != expected %llu\n",
                (unsigned long long)len, (unsigned long long)need);
        exit(2);
    }

    bb_w  = (const float *)p; p += (size_t)bu * (size_t)(in_dim + hid) * sizeof(float);
    bb_b  = (const float *)p; p += (size_t)bu * sizeof(float);
    ff1_w = (const float *)p; p += (size_t)hid * (size_t)zu * sizeof(float);
    ff1_b = (const float *)p; p += (size_t)hid * sizeof(float);
    ff2_w = (const float *)p; p += (size_t)hid * (size_t)zu * sizeof(float);
    ff2_b = (const float *)p; p += (size_t)hid * sizeof(float);
    ta_w  = (const float *)p; p += (size_t)hid * (size_t)zu * sizeof(float);
    ta_b  = (const float *)p; p += (size_t)hid * sizeof(float);
    tb_w  = (const float *)p; p += (size_t)hid * (size_t)zu * sizeof(float);
    tb_b  = (const float *)p; p += (size_t)hid * sizeof(float);
    ret   = (const float *)p; p += (size_t)hid * sizeof(float);
    h0    = (const float *)p; p += (size_t)hid * sizeof(float);
    inputs = (const float *)p;

    memset(&cell, 0, sizeof(cell));
    cell.input_dim = (int)in_dim;
    cell.hidden_dim = (int)hid;
    cell.backbone_units = (int)bu;
    cell.use_retention = (int)use_ret;
    cell.bb_w = (bu > 0) ? bb_w : NULL;
    cell.bb_b = (bu > 0) ? bb_b : NULL;
    cell.ff1_w = ff1_w;
    cell.ff1_b = ff1_b;
    cell.ff2_w = ff2_w;
    cell.ff2_b = ff2_b;
    cell.ta_w = ta_w;
    cell.ta_b = ta_b;
    cell.tb_w = tb_w;
    cell.tb_b = tb_b;
    cell.ret = (use_ret != 0) ? ret : NULL;
    cell.ts = ts;

    rc = cfc_validate(&cell);
    printf("VALIDATE=%d\n", rc);
    printf("N_STEPS=%d\nINPUT_DIM=%d\nHIDDEN_DIM=%d\nBACKBONE_UNITS=%d\nTS=%.9g\n",
           (int)n_steps, (int)in_dim, (int)hid, (int)bu, (double)ts);
    printf("MACS_PER_STEP=%d\n", (int)(bu * (in_dim + hid) + 4 * hid * zu));
    if (rc != 0) {
        free(buf);
        die("cfc_validate rejected the configuration");
    }

    h = (float *)calloc((size_t)hid, sizeof(float));
    hs = (float *)calloc((size_t)n_steps * (size_t)hid, sizeof(float));
    if (h == NULL || hs == NULL) {
        die("calloc failed");
    }

    cfc_init_state(&cell, h);
    for (i = 0; i < hid; ++i) {
        if (h[i] != 0.0f) {
            die("cfc_init_state did not zero the state");
        }
    }
    for (i = 0; i < hid; ++i) {
        h[i] = h0[i]; /* non-zero initial states are allowed */
    }

    for (t = 0; t < n_steps; ++t) {
        cfc_step(&cell, inputs + (size_t)t * (size_t)in_dim, h);
        memcpy(hs + (size_t)t * (size_t)hid, h, (size_t)hid * sizeof(float));
    }
    write_file(out_path, "CFCOUT01", n_steps, hid, hs, (size_t)n_steps * (size_t)hid);

    if (bench_steps > 0) {
        /* Best of 5 repeats to drop scheduler noise; one repeat is long enough
         * (100k steps, sub-millisecond per step) to swamp timer resolution. */
        double best = 1e30;
        for (rep = 0; rep < 5; ++rep) {
            double t0, t1;
            cfc_init_state(&cell, h);
            t0 = now_seconds();
            for (k = 0; k < bench_steps; ++k) {
                cfc_step(&cell, inputs + (size_t)(k % (int)n_steps) * (size_t)in_dim, h);
            }
            t1 = now_seconds();
            if (t1 - t0 < best) {
                best = t1 - t0;
            }
        }
        {
            double checksum = 0.0;
            for (i = 0; i < hid; ++i) {
                checksum += (double)h[i];
            }
            printf("BENCH_STEPS=%d\n", bench_steps);
            printf("BENCH_MIN_SECONDS=%.9f\n", best);
            printf("BENCH_US_PER_STEP=%.6f\n", best * 1e6 / (double)bench_steps);
            printf("BENCH_CHECKSUM=%.9g\n", checksum);
        }
    }

    free(hs);
    free(h);
    free(buf);
    return 0;
}
