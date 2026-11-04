#ifndef _IMU_H_
#define _IMU_H_
#include "robot.h"
// IMU (QMI8658C) 接口
void imu_init();
void imu_read_once();
void imu_deinit();
bool imu_is_initialized();

// 姿态数据
extern float roll;
extern float pitch;
extern float yaw;
extern float accel_x;
extern float accel_y;
extern float accel_z;
extern float pit_gyro;
extern float dq;
// d(yaw)/dt，单位 deg/s。固件里 yaw = yaw - gyro_yaw_comp*dt，所以它是
// -gyro_yaw_comp（已过低通与 1.5 deg/s 死区）。航向环的 D 项用它。
extern float yaw_rate;
#endif
