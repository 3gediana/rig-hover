# cfc_host 交付说明（工作流 C）

> 一句话：`ncps.torch.CfCCell(mode="default")` ↔ numpy ↔ C 三方逐元素对拍全部 PASS，
> 判据 1e-5，实测最大误差 1.9e-06（余量 5x），200 步无累积。
> 但 ncps 源码里有 6 个地方**跟直觉/跟 CONTRACT 写法不一样**，导出时踩到就会静默算错。

## 0. 怎么跑

```bat
:: 真验收（一条命令，自动找 cl.exe / vcvars64.bat，自动编译）
D:\smartcar\.venv\Scripts\python.exe D:\robot_work\pc\cfc_host\test_parity.py

:: 只跑 numpy<->ncps（第一道坎）
D:\smartcar\.venv\Scripts\python.exe D:\robot_work\pc\cfc_host\test_parity_ncps.py
```

当前状态：`41 项 41 PASS / 0 FAIL`，末行打印 `PASS`（退出码 0）。
从任何 cwd 都能跑（脚本内部全用绝对路径）。

## 1. 文件

| 文件 | 作用 |
|---|---|
| `cfc_ref.py` | 通用 numpy 参考实现。`cfc_step_np(params, inp, h)` / `cfc_forward_seq` / `make_params`。支持 input_dim / hidden_dim / backbone_units 可配、无 backbone（`bb_w=None`）、保留门（`params["ret"]`）、`ts`。零框架依赖。 |
| `test_parity_ncps.py` | numpy ↔ ncps 对拍（4 组维度 + 2 个 ts + AutoNCP 掩码）。含 `params_from_cell` / `params_from_masked_cell` / `torch_seq`（可施加保留门）——**导出权重直接抄这两个函数**。 |
| `cfc_engine.h` / `.c` | 纯 C99 引擎，签名严格照 CONTRACT §5.3（`cfc_cell_t` 字段顺序一字未改），另加可选 `cfc_validate()`。只用 `<math.h>` / `<string.h>`，无 VLA、无 malloc、无平台头。 |
| `test_engine.c` | C 侧测试驱动：读 `testdata.bin`、写 `out_c.bin`，另有 `--selftest`（引擎自检）和 `--bench N`（计时）。**不含任何模型解析逻辑。** |
| `test_parity.py` | 一条命令跑完全流程 + 全部比对 + 计时 + PASS/FAIL。 |
| `testdata.bin` / `out_c.bin` | 最近一次（106/64/64，保留门 ret=-3）的输入/输出，格式见 `test_engine.c` 文件头。 |

## 2. 实测数字

| 比对 | 200 步随机输入最大绝对误差 | 最差单例（±10 饱和边界） |
|---|---|---|
| ncps ↔ numpy | 5.81e-07（106/64/64） | 1.25e-06 |
| numpy ↔ C | 7.30e-07 | 1.88e-06 |
| ncps ↔ C | 7.30e-07 | — |

误差累积（106/64/64，200 步）：第 1/10/50/100/200 步分别
`ncps-np: 2.4e-07 3.7e-07 3.3e-07 2.8e-07 3.3e-07`，
`np-C: 3.6e-07 3.9e-07 4.8e-07 3.4e-07 6.3e-07`；
后 10 步/前 10 步 max = 1.27x / 1.17x —— **无累积趋势**，只是 float32 舍入噪声。
C 自链 3 层（AutoNCP）累计也只 2.98e-07。

保留门：`ret=-3`（λ≈0.047）/ `ret=U(-2,4)`（λ≈0.65）/ `ret=+8`（λ≈0.9997）三档
ncps(+torch 端施加门) ↔ numpy ↔ C 全部 < 1e-5。
`λ=0`（`ret=-1e4`）时：C 的 `out_c.bin` 与不带保留门**逐字节相同**（51216 bytes），numpy 同样逐字节相同。
实现手段：`cfc_engine.c` 里 `if (lam != 0.0f) h_new += lam*(h - h_new);`——不执行加法，
连 `h_new == -0.0f` 时 `+0.0f` 会翻转符号位这个病态情形都躲开了。

## 3. C 单步耗时与 ESP32 估计

- PC（本机，100000 步 × 5 次取最优）：**10.8 µs/步**（无保留门 10.7 / 有保留门 10.8，差别可忽略）。
- 该配置 27,264 MAC/步（= `64×(106+64)` backbone + `4×64×64` 四个投影），另有 256 次超越函数
  （64 LeCun-tanh + 128 tanh + 64 exp）。
- **按"ESP32-S3 ≈ 1/4 单核性能"折算 = 43 µs/步。这个数不自洽**：27,264 MAC 在 240 MHz
  单发射 FPU（无 FMA，1 MAC 至少 2 个周期）上不可能低于 ~0.11 ms，实际约 **0.3~0.5 ms/步**；
  再叠加 256 次软件 tanh/exp（每次 150~400 周期）还要 +0.15~0.4 ms。
  建议按 **0.5 ms/步** 做预算（50 Hz 下占 2.5% CPU），与 TECH-PLAN §6.4 的 2.5 ms 相比反而宽松，
  但**瓶颈很可能是超越函数而不是 MAC**——真要再快，先换 tanh/exp 的多项式近似，而不是砍 hidden。
- 交叉检查：按 §6.4 的 20 MFLOPS 口径，54,528 FLOP/步 → 2.7 ms/步。

## 4. ★ ncps 源码里 6 个反直觉点（导出/固件必须知道）

1. **cell 没有 head。** `CfCCell.forward` 返回 `(new_hidden, new_hidden)`，
   CONTRACT §5.2 结尾的 `out = head @ h_new + b_head` **不在 cell 里**——它是
   `ncps.torch.CfC` 这个 RNN 包装器的 `self.fc`（且默认 `nn.Identity()`，只有传
   `proj_size` 才是 Linear）。所以 `cfc_cell_t` 没有 head 字段是对的，head 必须另导出、另算。

2. **LeCun 的常数是 `0.666`，不是 `2/3`。** 源码字面量就是 `1.7159 * tanh(0.666 * x)`。
   用 `2.0/3.0` 会在每个 backbone 输出上引入 ~0.1% 的系统偏差（tanh 的输入差 0.00067·|v|）。
   Python/C 两侧都必须照抄 `0.666`。

3. **`sparsity_mask` 只作用在 ff1 和 ff2，`time_a`/`time_b` 永远稠密。**
   `forward` 里只有 `F.linear(x, W*mask, b)` 这两处带 mask，时间常数投影是
   `self.time_a(x)` / `self.time_b(x)`。**把 mask 也乘到 ta/tb 上，h 会错 ~0.6**（实测）。
   `test_parity.py` 的 [2b] 就是守这一条的。

4. **mask 的朝向**：`CfCCell(sparsity_mask=m)` 的 `m` 形状是 `(in, out)`，内部
   `np.abs(m.T)` 才拿去逐元素乘 `weight(out, in)`。另外 `WiredCfCCell` 给每层拼 mask 时，
   **recurrent 块是 `np.ones`（全 1）**，AutoNCP 只剪感知输入那一段。所以"AutoNCP ⇒ 稀疏"
   对 recurrent 部分不成立。

5. **AutoNCP(units, out).num_layers ≠ 1。** 实测 `AutoNCP(64, 2)` → 3 层，单元数
   `[38, 24, 2]`，即 `WiredCfCCell` 内部是**3 个串联的 CfCCell**，`h` 是 3 段拼接（共 64）。
   一个 `cfc_cell_t` 表示不了它：固件要实例化 3 个 cell、按层切换输入/状态、逐层串联
   （[2c] 已证明这个链法对拍 3.0e-07）。另外 `CfC` 包装器的 readout 只取**最后一层**的输出
   （这里是 2 维），TECH-PLAN §6.1 里 `head: Linear(64→k)` 的 64 维输入要用全 state 拼接或自己切，
   不能想当然。

6. **`ts` 只乘 time_a 那一半**（`t = sigmoid(t_a*ts + t_b)`），不是乘整个 pre-activation；
   且 `CfC` 包装器在 `timespans=None` 时传 `ts=1.0`。另外 `mode="no_gate"` 是
   `ff1 + t*ff2`（不是插值）、`mode="pure"` 是另一套公式——我们只实现 `default`，
   别顺手把别的分支当 default。

补充：`init_weights()` 对所有 2-D 参数做 `xavier_uniform_`（含 time_a/time_b/backbone），
bias 用 `nn.Linear` 默认 `±1/sqrt(fan_in)`。我的 `make_params` 复刻了这个分布。

## 5. 契约（CONTRACT）里我发现的问题，建议 Lead 更新

1. §5.2 的 `out = head @ h_new + b_head` 不属于 cell（见上 4.1）。C 接口不导出 head 是对的，
   但§5.2 把它写进"CfC cell 公式"会让人以为 cell 里有 head。
2. §5.3 的 `use_retention` 与 `ret` 语义没写全：到底是"ret != NULL 就启用"还是"use_retention 是总开关"？
   我的实现是**两者都满足才启用**（保守，避免加载器传了 ret 却忘设开关时静默打开未训练的门），
   并在 `cfc_validate()` 里对两种不一致分别返回 -3（开关开但 ret=NULL）/ -4（ret 非空但开关关）。
   固件加载器请务必调用 `cfc_validate()` 并检查返回值。
3. §5.5 的算力假设"4 个线性变换各 64×64×0.2"与实际不符：`time_a/time_b` 不受 mask 影响，
   恒为稠密；AutoNCP 又是 3 层。建议按 27k MAC（稠密 106/64/64）做预算再乘层数。
4. §5.4 的 1e-5 判据实测余量 5x，够宽松，建议保留（不建议收紧到 1e-6，float32 下会给
   不同编译器/不同 BLAS 留下无谓的 flaky 风险）。

## 6. 最脆弱的一点

**导出侧的 mask 与层结构**（第 3、5 条）：这两处一旦搞错，ncps/numpy/C 三方**依然自洽**、
对拍全绿，但固件行为会静默错掉（错 0.6 量级，足以让机器人行为完全跑偏）。
所以 `test_parity.py` 的 [2b][2c] 不是可选项——每次改导出脚本都要重跑。
