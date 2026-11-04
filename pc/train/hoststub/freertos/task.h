/*
 * hoststub/freertos/task.h -- 只提供 cfc_model.c 用到的 vTaskDelay。
 */
#ifndef HOSTSTUB_FREERTOS_TASK_H
#define HOSTSTUB_FREERTOS_TASK_H

#include "freertos/FreeRTOS.h"

void vTaskDelay(TickType_t ticks);

#endif /* HOSTSTUB_FREERTOS_TASK_H */
