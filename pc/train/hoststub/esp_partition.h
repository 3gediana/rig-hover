/*
 * hoststub/esp_partition.h -- 用 RAM 模拟 assets 分区。
 *
 * 语义刻意贴近真 flash：
 *   - 只允许写已擦除区域（write 是 AND，写 0->1 会写出错值，和真 flash 一样）；
 *   - erase 必须扇区对齐（4 KB），否则返回 ESP_ERR_INVALID_ARG；
 *   - 初始内容全 0xFF。
 * 这样 cfc_model.c 的 "擦 -> 写 -> 读回校验" 路径在 PC 上也是有意义的。
 */
#ifndef HOSTSTUB_ESP_PARTITION_H
#define HOSTSTUB_ESP_PARTITION_H

#include <stddef.h>
#include <stdint.h>

typedef int esp_err_t;
#ifndef ESP_OK
#define ESP_OK 0
#endif
#ifndef ESP_ERR_INVALID_ARG
#define ESP_ERR_INVALID_ARG 0x102
#endif
#ifndef ESP_ERR_INVALID_SIZE
#define ESP_ERR_INVALID_SIZE 0x104
#endif

typedef enum {
    ESP_PARTITION_TYPE_APP = 0,
    ESP_PARTITION_TYPE_DATA = 1,
} esp_partition_type_t;

typedef enum {
    ESP_PARTITION_SUBTYPE_ANY = 0xFF,
} esp_partition_subtype_t;

typedef struct {
    esp_partition_type_t    type;
    esp_partition_subtype_t subtype;
    uint32_t                address;
    uint32_t                size;
    char                    label[17];
} esp_partition_t;

/* 返回静态分区描述符（size 由 host_shim.c 决定），找不到时返回 NULL。 */
const esp_partition_t *esp_partition_find_first(esp_partition_type_t type,
                                                esp_partition_subtype_t subtype,
                                                const char *label);
esp_err_t esp_partition_read(const esp_partition_t *part, size_t src_offset,
                             void *dst, size_t size);
esp_err_t esp_partition_write(const esp_partition_t *part, size_t dst_offset,
                              const void *src, size_t size);
esp_err_t esp_partition_erase_range(const esp_partition_t *part, size_t offset,
                                    size_t size);

#endif /* HOSTSTUB_ESP_PARTITION_H */
