# 本机并发跑多个 torch 进程会伪造崩溃

**适用**：在这台笔记本上跑任何多进程训练/特征提取。

## 现场

实验 sweep 用 2–4 个 worker 并行时，随机出现**看起来像代码 bug**的失败：

```
AcceleratorError: CUDA error: resource already mapped
RuntimeError: unallocation / bad allocation
OSError [WinError 1455]: 分页文件太小，无法完成操作
OSError [WinError 1114]: 动态链接库(DLL)初始化例程失败（加载 shm.dll / cublas64_12.dll）
torch DataLoader worker 起不来
```

这些错误**不稳定复现、换一次 run 就好了**，最容易误判成"某个模型有 bug"。

## 根因（已确认）

- 这台机器 **commit memory（RAM + 页面文件）会被打满**；多个 python 进程各自
  import torch / CUDA 时会重复映射大块内存。
- `G:\ai\python11` 是虚拟环境的 site-packages，且 `G:` 是 **USB 盘**——
  `torch` 从那里加载 DLL/so 更慢也更容易失败。
- Windows 上 `torch.utils.data.DataLoader(num_workers=10)` 会 spawn 子解释器，
  每个都重新 import torch，直接触发 `WinError 1455`。

## 验证后的做法

1. **一个时刻只跑一个训练/编码进程**。串行变慢，但结果是干净的。
   本项目所有最终数字都是单进程串行跑出来的。
2. 解码用 **ThreadPoolExecutor** 而不是多进程 DataLoader
   （PyAV / cv2 会释放 GIL），并且限制预取（2×）。
3. 训练 DataLoader 用 `num_workers=0` + `preload=True`
   （Windows spawn 会把整个数据集复制进每个 worker）。
4. 大文件别放在 USB 的 `G:` 上做**许多小写**——`G:` 上大量小 npz 的吞吐会掉到 0.25 文件/s。
   顺序写 npz 没问题，**并发下载 + 并发解码**时才是灾难。

## 判据

出现上面那些错误时，**先看并发数和 commit memory，再怀疑代码**。
若把并发降到 1 后错误消失，就按环境问题处理，不要浪费时间调代码。

## 边界

这台机器的具体限制可能变化（页面文件大小、代理、USB 盘）。
结论只保证：**该环境曾因并发 torch 进程耗尽 commit memory 并产生上述假故障**。

## 出处

`D:\robot_work\stream_memory_research\logs\data_video_status.md`（实测记录）、
`reports/中文总结报告.md` §3.3 / §8。
