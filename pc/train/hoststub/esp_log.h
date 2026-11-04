/*
 * hoststub/esp_log.h -- ESP_LOGx 的 PC 替身：打到 stderr，格式与 IDF 近似。
 *
 * 变参宏只写 __VA_ARGS__（不写 tag, fmt 拆分），因为固件里每个 ESP_LOGx
 * 调用都至少带一个格式实参，避免依赖 MSVC 对 ##__VA_ARGS__ 的扩展。
 */
#ifndef HOSTSTUB_ESP_LOG_H
#define HOSTSTUB_ESP_LOG_H

void host_log(const char *level, const char *tag, const char *fmt, ...);

#define ESP_LOGE(tag, ...) host_log("E", tag, __VA_ARGS__)
#define ESP_LOGW(tag, ...) host_log("W", tag, __VA_ARGS__)
#define ESP_LOGI(tag, ...) host_log("I", tag, __VA_ARGS__)
#define ESP_LOGD(tag, ...) host_log("D", tag, __VA_ARGS__)

#endif /* HOSTSTUB_ESP_LOG_H */
