#ifndef ARB_FREERTOS_H
#define ARB_FREERTOS_H
#include <stdint.h>
typedef uint32_t TickType_t;
typedef int      BaseType_t;
typedef struct { int dummy; } portMUX_TYPE;
#define portMUX_INITIALIZER_UNLOCKED { 0 }
#define portENTER_CRITICAL(m)   ((void)(m))
#define portEXIT_CRITICAL(m)    ((void)(m))
#define taskENTER_CRITICAL(m)   ((void)(m))
#define taskEXIT_CRITICAL(m)    ((void)(m))
#define pdMS_TO_TICKS(ms)       ((TickType_t)(ms))
#define pdTRUE  1
#define pdFALSE 0
TickType_t xTaskGetTickCount(void);
void       vTaskDelay(TickType_t ticks);
/* MSVC has no GCC __atomic_* intrinsics. The arbiter uses a single seq-cst
 * fence to order the two published output slots; a single-threaded host test
 * has no second observer, so the fence compiles to nothing here. */
#ifndef __ATOMIC_SEQ_CST
#define __ATOMIC_SEQ_CST 0
#endif
#ifndef __atomic_thread_fence
#define __atomic_thread_fence(order) ((void)(order))
#endif
#endif
