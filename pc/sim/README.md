# pc/sim —— RIG-Hover 闭环仿真与端到端评估

> 一句话：用**固件里的真实控制参数**搭一个机器人动力学仿真，把教师 / CfC / MLP /
> PD 四种控制器接到同一个闭环里，产出可复现的量化对比。
> 不需要机器人上电，不需要摄像头。

```bat
set PY=D:\smartcar\.venv\Scripts\python.exe
cd D:\robot_work\pc\sim
%PY% eval.py --seeds 10        :: 验收命令：标定 + 5 场景 × 4 控制器 × 10 种子 + 阶跃
%PY% make_report.py            :: 6 张图 + REPORT.md
```

## 文件

| 文件 | 作用 |
|---|---|
| `params.py` | **全部参数 + 出处**（`文件:行号` 或接口字段），估计值单独标 `EST`。`python params.py` 直接打印出处表。 |
| `sim.py` | 机体动力学（轮式倒立摆 ZOH 离散 + 航向双积分 + 舵机二阶）、IMU 模型、`robot_control()` / `control_arbiter.c` 的逐行副本、场景、106 维观测组装、闭环运行器、标定、阶跃实验。 |
| `controllers.py` | 四种控制器的统一接口；CfC 用 numpy 复刻固件前向（`verify_cfc()` 与 torch 逐帧对齐，判据 1e-5，实测 1.1e-06）。 |
| `train_mlp.py` | 训练无记忆基线（同数据/同头/同归一化，唯一差别是没有隐状态）。 |
| `eval.py` | 跑全部组合，输出 `results.json` / `results_table.md` + 终端总表。 |
| `make_report.py` | 6 张图（`figs/`）+ `REPORT.md`。 |

## 三个必须知道的结论（细节见 `REPORT.md`）

1. **源码原样的 `lqr_k=[1300,5,25,0.4]` 让平衡环结构上不可稳定** —— 位置/速度
   两个通道是正反馈，θ 环与位置环的稳定条件互补、临界值与机体参数无关。
   `params.FirmwareParams.gains("firmware"|"sign_flip"|"care")` 是三种口径的开关，
   默认 `sign_flip`（只把 `K0/K1` 取负）。
2. **「只动头」在闭环里做不到**：航向环是纯 P 控双积分，ζ≈0.08，机身一直在小角度
   振荡，把头部的动作淹掉。
3. **CfC 比 MLP / PD 好很多，且在闭环里比教师还好** —— 但后者主要是因为教师的
   `k_fast=3.0` 会把这个欠阻尼航向环推到振荡，**不是** CfC 更聪明的证据。

## 参数纪律

* 能读源码的一律抄源码，出处写在 `params._SRC`（`REPORT.md` §2 就是它的投影）。
* 拿不到真值的（质量、轮距、质心高、舵机带宽、力矩常数）一律标 `EST`，
  并且只影响「多快倒/多快转」，不改变第 1 条的符号结论。
* 两个标定量 `k_torque` / `k_yaw_accel` 由 `sim.calibrate()` 按写明的判据反推，
  每次 `eval.py` 都会重算并打印。
