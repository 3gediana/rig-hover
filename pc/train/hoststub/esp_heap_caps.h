/*
 * hoststub/esp_heap_caps.h -- 在 PC 上编译固件 cfc_model.c 用的最小替身。
 *
 * 目的：让 test_e2e 能**直接编译固件自己的 cfc_model.c**（不是重写一份解析器），
 * 从而验证 "blob 在真机加载器里能被正确解析" 这件事。替身只实现 cfc_model.c
 * 真正用到的符号，行为尽量贴近 ESP-IDF（见 host_shim.c）。
 */
#ifndef HOSTSTUB_ESP_HEAP_CAPS_H
#define HOSTSTUB_ESP_HEAP_CAPS_H

#include <stddef.h>
#include <stdint.h>

#define MALLOC_CAP_SPIRAM   (1u << 10)
#define MALLOC_CAP_INTERNAL (1u << 0)
#define MALLOC_CAP_8BIT     (1u << 2)

void *heap_caps_malloc(size_t size, uint32_t caps);

/* cfc_model.c prefers internal SRAM for small blobs; on the PC both caps map to
 * malloc, and host malloc already returns 16-byte aligned blocks, so the
 * alignment request needs no separate allocator. */
void *heap_caps_aligned_alloc(size_t align, size_t size, uint32_t caps);
void  heap_caps_free(void *ptr);

#endif /* HOSTSTUB_ESP_HEAP_CAPS_H */
