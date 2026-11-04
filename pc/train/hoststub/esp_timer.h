/*
 * hoststub/esp_timer.h -- 微秒级单调时钟的 PC 替身。
 *
 * host_shim.c 里每次读取会推进 1000 us（模拟时间流逝），这样 cfc_model.c
 * 里 "等 200 ms 让读者退出" 的循环在 PC 上一定会终止，而不是空转或死等。
 */
#ifndef HOSTSTUB_ESP_TIMER_H
#define HOSTSTUB_ESP_TIMER_H

#include <stdint.h>

int64_t esp_timer_get_time(void);

#endif /* HOSTSTUB_ESP_TIMER_H */
