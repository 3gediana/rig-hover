/*
 * host_shim.c -- 在 PC 上跑**固件原版** cfc_model.c 所需的最小运行环境。
 *
 * 这里实现的每个符号都是 ESP-IDF 的一个真函数：
 *   heap_caps_malloc / heap_caps_free         -> malloc/free
 *   esp_partition_find_first/read/write/erase -> 一块 1 MB 的 RAM "flash"，
 *                                                写=AND、擦=0xFF、擦除必须 4 KB 对齐
 *   esp_timer_get_time                        -> 每次调用推进 1000 us 的假时钟
 *   vTaskDelay                                -> 推进假时钟
 *   ESP_LOGx                                  -> 打到 stderr
 *
 * 为什么要这么做：test_e2e 要证明的是"明早传上机器人的 blob 能被**固件自己的
 * 加载器**正确解析"，所以宁可给固件源码配一套替身，也不重写一份解析器
 * （重写的解析器即使对，也证明不了固件对）。
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

/* assets 分区大小：必须 > CFC_STORE_REGION_SZ (0x88000)，留够 1 MB */
#define HOST_PART_BYTES (1024u * 1024u)
#define HOST_SECTOR     4096u

static uint8_t *s_flash;
static int64_t  s_now_us = 1000000;

static esp_partition_t s_part = {
    ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, 0, HOST_PART_BYTES, "assets"
};

static int flash_ready(void)
{
    if (s_flash != NULL) {
        return 1;
    }
    s_flash = (uint8_t *)malloc(HOST_PART_BYTES);
    if (s_flash == NULL) {
        return 0;
    }
    memset(s_flash, 0xFF, HOST_PART_BYTES);   /* 擦除态 */
    return 1;
}

/* ---- heap_caps ------------------------------------------------------- */

void *heap_caps_malloc(size_t size, uint32_t caps)
{
    (void)caps;
    if (size == 0) {
        return NULL;
    }
    return malloc(size);
}

void *heap_caps_aligned_alloc(size_t align, size_t size, uint32_t caps)
{
    /* 主机 malloc 已经按 16 字节对齐返回，这里只核对一下，不另开分配器 */
    void *p = heap_caps_malloc(size, caps);
    if (p != NULL && align > 1 && ((uintptr_t)p % align) != 0) {
        free(p);
        return NULL;
    }
    return p;
}

void heap_caps_free(void *ptr)
{
    free(ptr);
}

/* ---- partition ------------------------------------------------------- */

const esp_partition_t *esp_partition_find_first(esp_partition_type_t type,
                                                esp_partition_subtype_t subtype,
                                                const char *label)
{
    if (!flash_ready()) {
        return NULL;
    }
    if (type != ESP_PARTITION_TYPE_DATA) {
        return NULL;
    }
    if (subtype != ESP_PARTITION_SUBTYPE_ANY && subtype != s_part.subtype) {
        return NULL;
    }
    if (label != NULL && strcmp(label, s_part.label) != 0) {
        return NULL;
    }
    return &s_part;
}

esp_err_t esp_partition_read(const esp_partition_t *part, size_t src_offset,
                             void *dst, size_t size)
{
    if (part == NULL || dst == NULL || !flash_ready()) {
        return ESP_ERR_INVALID_ARG;
    }
    if (src_offset > part->size || size > part->size - src_offset) {
        return ESP_ERR_INVALID_ARG;
    }
    memcpy(dst, s_flash + src_offset, size);
    return ESP_OK;
}

esp_err_t esp_partition_write(const esp_partition_t *part, size_t dst_offset,
                              const void *src, size_t size)
{
    size_t i;
    if (part == NULL || src == NULL || !flash_ready()) {
        return ESP_ERR_INVALID_ARG;
    }
    if (dst_offset > part->size || size > part->size - dst_offset) {
        return ESP_ERR_INVALID_ARG;
    }
    /* 真 NOR flash 只能把 1 写成 0：不先擦除就写会得到 AND 的结果 */
    for (i = 0; i < size; i++) {
        s_flash[dst_offset + i] &= ((const uint8_t *)src)[i];
    }
    return ESP_OK;
}

esp_err_t esp_partition_erase_range(const esp_partition_t *part, size_t offset,
                                    size_t size)
{
    if (part == NULL || !flash_ready()) {
        return ESP_ERR_INVALID_ARG;
    }
    if (offset % HOST_SECTOR != 0 || size % HOST_SECTOR != 0) {
        return ESP_ERR_INVALID_ARG;
    }
    if (offset > part->size || size > part->size - offset) {
        return ESP_ERR_INVALID_SIZE;
    }
    memset(s_flash + offset, 0xFF, size);
    return ESP_OK;
}

/* ---- timer / delay --------------------------------------------------- */

int64_t esp_timer_get_time(void)
{
    s_now_us += 1000;       /* 假时钟：保证 cfc_model.c 的等待循环一定收敛 */
    return s_now_us;
}

void vTaskDelay(TickType_t ticks)
{
    s_now_us += (int64_t)ticks * 1000;
}

/* ---- log ------------------------------------------------------------- */

void host_log(const char *level, const char *tag, const char *fmt, ...)
{
    va_list ap;
    fprintf(stderr, "[%s %s] ", level, tag != NULL ? tag : "?");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}
