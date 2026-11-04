/*
 * hoststub/freertos/FreeRTOS.h -- FreeRTOS 最小替身（cfc_model.c 只用到 vTaskDelay）。
 */
#ifndef HOSTSTUB_FREERTOS_H
#define HOSTSTUB_FREERTOS_H

#include <stdint.h>

typedef uint32_t TickType_t;
typedef int      BaseType_t;

#ifndef pdMS_TO_TICKS
#define pdMS_TO_TICKS(ms) ((TickType_t)(ms))
#endif

#endif /* HOSTSTUB_FREERTOS_H */
