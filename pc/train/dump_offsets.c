/*
 * dump_offsets.c -- 用 offsetof()/sizeof() 打印固件 blob 头的**真实**字段偏移。
 *
 * 为什么要有这个程序：blob 头的布局是 cfc_model.c 里一组 #define H_* 的硬偏移，
 * 不是 C 结构体的对齐规则。Python 侧如果"按经验猜"对齐/填充，就可能在某个字段
 * 上错 2~4 字节，而固件只会在真机上以 "bad magic" / "cell layer size mismatch"
 * 这种不指向根因的方式失败。所以这里用真正的编译器算出偏移，和 Python 打包代码
 * 逐项对比（见 export_blob.py --check-offsets）。
 *
 * 编译（MSVC，C 模式）：
 *     cl /nologo /W3 dump_offsets.c /Fe:dump_offsets.exe
 *
 * 输出（每行一个，Python 用正则解析）：
 *     offsetof <field> = <n>
 *     offsetof natural.<field> = <n>      （不做 #pragma pack 时的对照）
 *     sizeof model_hdr_packed = <n>
 *     sizeof model_hdr_natural = <n>
 * 任何一项与 cfc_model.h 的字节表不符 -> 打印 FAIL 并退出码 1。
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* ---- 打包形式：与 cfc_model.h 的字节表逐字段一致 ---- */
#pragma pack(push, 1)
typedef struct {
    uint32_t magic;         /* off 0                                */
    uint16_t ver;           /* off 4                                */
    uint16_t flags;         /* off 6                                */
    uint16_t input_dim;     /* off 8                                */
    uint16_t cell_in_dim;   /* off 10                               */
    uint16_t hidden_dim;    /* off 12                               */
    uint16_t backbone_units;/* off 14                               */
    uint16_t n_heads;       /* off 16                               */
    uint16_t reserved;      /* off 18                               */
    uint16_t head_dim[8];   /* off 20   (16 B)                      */
    uint16_t head_act[8];   /* off 36   (16 B)                      */
    uint32_t layer_off[16]; /* off 52   (64 B)                      */
    uint32_t layer_len[16]; /* off 116  (64 B)                      */
    float    input_mean[256];/* off 180 (1024 B)                    */
    float    input_std[256]; /* off 1204 (1024 B)                   */
    float    clip_sigma;    /* off 2228                             */
    float    ts;            /* off 2232                             */
    uint32_t crc32;         /* off 2236                             */
    uint8_t  layer_prec[16];/* off 2240 (16 B)                      */
} model_hdr_packed;
#pragma pack(pop)

/* ---- 自然对齐形式：用来证明"就算不 pack，布局也一样" ---- */
typedef struct {
    uint32_t magic;
    uint16_t ver;
    uint16_t flags;
    uint16_t input_dim;
    uint16_t cell_in_dim;
    uint16_t hidden_dim;
    uint16_t backbone_units;
    uint16_t n_heads;
    uint16_t reserved;
    uint16_t head_dim[8];
    uint16_t head_act[8];
    uint32_t layer_off[16];
    uint32_t layer_len[16];
    float    input_mean[256];
    float    input_std[256];
    float    clip_sigma;
    float    ts;
    uint32_t crc32;
    uint8_t  layer_prec[16];
} model_hdr_natural;

#define EXPECT_HDR_BYTES 2256

typedef struct {
    const char *name;
    size_t      off;
    size_t      expect;
} field_t;

int main(void)
{
    static const field_t fields[] = {
        { "magic",          offsetof(model_hdr_packed, magic),          0 },
        { "ver",            offsetof(model_hdr_packed, ver),            4 },
        { "flags",          offsetof(model_hdr_packed, flags),          6 },
        { "input_dim",      offsetof(model_hdr_packed, input_dim),      8 },
        { "cell_in_dim",    offsetof(model_hdr_packed, cell_in_dim),   10 },
        { "hidden_dim",     offsetof(model_hdr_packed, hidden_dim),    12 },
        { "backbone_units", offsetof(model_hdr_packed, backbone_units),14 },
        { "n_heads",        offsetof(model_hdr_packed, n_heads),       16 },
        { "reserved",       offsetof(model_hdr_packed, reserved),      18 },
        { "head_dim",       offsetof(model_hdr_packed, head_dim),      20 },
        { "head_act",       offsetof(model_hdr_packed, head_act),      36 },
        { "layer_off",      offsetof(model_hdr_packed, layer_off),     52 },
        { "layer_len",      offsetof(model_hdr_packed, layer_len),    116 },
        { "input_mean",     offsetof(model_hdr_packed, input_mean),   180 },
        { "input_std",      offsetof(model_hdr_packed, input_std),   1204 },
        { "clip_sigma",     offsetof(model_hdr_packed, clip_sigma),  2228 },
        { "ts",             offsetof(model_hdr_packed, ts),          2232 },
        { "crc32",          offsetof(model_hdr_packed, crc32),       2236 },
        { "layer_prec",     offsetof(model_hdr_packed, layer_prec),  2240 }
    };
    /* 自然对齐形式的偏移，用来和打包形式逐个比对 */
    static const field_t nat_fields[] = {
        { "magic",          offsetof(model_hdr_natural, magic),          0 },
        { "ver",            offsetof(model_hdr_natural, ver),            4 },
        { "flags",          offsetof(model_hdr_natural, flags),          6 },
        { "input_dim",      offsetof(model_hdr_natural, input_dim),      8 },
        { "cell_in_dim",    offsetof(model_hdr_natural, cell_in_dim),   10 },
        { "hidden_dim",     offsetof(model_hdr_natural, hidden_dim),    12 },
        { "backbone_units", offsetof(model_hdr_natural, backbone_units),14 },
        { "n_heads",        offsetof(model_hdr_natural, n_heads),       16 },
        { "reserved",       offsetof(model_hdr_natural, reserved),      18 },
        { "head_dim",       offsetof(model_hdr_natural, head_dim),      20 },
        { "head_act",       offsetof(model_hdr_natural, head_act),      36 },
        { "layer_off",      offsetof(model_hdr_natural, layer_off),     52 },
        { "layer_len",      offsetof(model_hdr_natural, layer_len),    116 },
        { "input_mean",     offsetof(model_hdr_natural, input_mean),   180 },
        { "input_std",      offsetof(model_hdr_natural, input_std),   1204 },
        { "clip_sigma",     offsetof(model_hdr_natural, clip_sigma),  2228 },
        { "ts",             offsetof(model_hdr_natural, ts),          2232 },
        { "crc32",          offsetof(model_hdr_natural, crc32),       2236 },
        { "layer_prec",     offsetof(model_hdr_natural, layer_prec),  2240 }
    };
    const size_t n_fields = sizeof(fields) / sizeof(fields[0]);
    const size_t sz_packed = sizeof(model_hdr_packed);
    const size_t sz_natural = sizeof(model_hdr_natural);
    size_t i;
    int bad = 0;

    for (i = 0; i < n_fields; i++) {
        printf("offsetof %s = %lu\n", fields[i].name, (unsigned long)fields[i].off);
    }
    for (i = 0; i < n_fields; i++) {
        printf("offsetof natural.%s = %lu\n", nat_fields[i].name,
               (unsigned long)nat_fields[i].off);
    }
    printf("sizeof model_hdr_packed = %lu\n", (unsigned long)sz_packed);
    printf("sizeof model_hdr_natural = %lu\n", (unsigned long)sz_natural);

    printf("expect_hdr_bytes = %d\n", EXPECT_HDR_BYTES);

    if (sz_packed != EXPECT_HDR_BYTES) {
        printf("FAIL sizeof(model_hdr_packed)=%lu != %d\n",
               (unsigned long)sz_packed, EXPECT_HDR_BYTES);
        bad = 1;
    }
    if (sz_natural != EXPECT_HDR_BYTES) {
        printf("FAIL sizeof(model_hdr_natural)=%lu != %d (natural alignment adds "
               "padding => field order differs from cfc_model.h)\n",
               (unsigned long)sz_natural, EXPECT_HDR_BYTES);
        bad = 1;
    }
    for (i = 0; i < n_fields; i++) {
        if (fields[i].off != fields[i].expect) {
            printf("FAIL offsetof(%s)=%lu != %lu\n", fields[i].name,
                   (unsigned long)fields[i].off, (unsigned long)fields[i].expect);
            bad = 1;
        }
        if (nat_fields[i].off != fields[i].off) {
            printf("FAIL natural offsetof(%s)=%lu != packed %lu\n", nat_fields[i].name,
                   (unsigned long)nat_fields[i].off, (unsigned long)fields[i].off);
            bad = 1;
        }
    }
    printf("%s\n", bad ? "FAIL" : "PASS");
    return bad;
}
