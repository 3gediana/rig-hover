# 接口契约 v1（A 阶段并行开发的共享约定）

> 三个并行工作流（PC感知 / 教师控制器 / CfC引擎）都按这份契约产出，**任何人不得单方面修改这里的约定**。
> 需要改 → 告诉 Lead，由 Lead 更新本文件并通知其他流。

## 0. 环境事实（都已核实，直接用）

| 项 | 值 |
|---|---|
| Python | `D:\smartcar\.venv\Scripts\python.exe`（3.11.9） |
| 关键包 | torch 2.10.0+cu128（CUDA可用）、ultralytics 8.4.21（**支持 YOLO26**）、ncps |
| GPU | RTX 5060 Laptop, 8GB |
| C 编译器 | `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe` |
| WSL | Ubuntu 22.04 存在但**没装 gcc**，不要指望它 |
| 工作根目录 | `D:\robot_work\` |

## 1. 硬性纪律

1. **绝对不要修改 `D:\smartcar` 里的任何文件。** 它只是参考实现 + 借用 venv。只读。
2. 每个工作流只写自己目录下的文件（见 §2 目录划分），**不要碰别人的目录**。
3. 所有产出的 Python 用 `D:\smartcar\.venv\Scripts\python.exe` 跑，不要自己建 venv、不要 pip install（需要新包先问 Lead）。
4. 每个工作流必须自带**可执行的验收测试**，测试要能一条命令跑通并打印 PASS/FAIL。
5. 时间成本优先：**能跑对就别优化**，先通后快。

## 2. 目录划分（写权限边界）

```
D:\robot_work\
├─ TECH-PLAN-v1.md            Lead 维护
├─ CONTRACT.md                本文件，Lead 维护
├─ pc\
│  ├─ perception\             【工作流 A】YOLO + 特征 + 手势基元
│  ├─ teacher\                【工作流 B】教师控制器 + 录制
│  └─ cfc_host\               【工作流 C】CfC 引擎（numpy 参考 + C 实现 + 对拍）
└─ rig_omni\                  【Lead】固件，别人不要碰
```

## 3. 工作流 A（perception）的产出契约

### 3.1 主入口

```python
# pc/perception/run_perception.py
class PerceptionPipeline:
    def __init__(self, model_path="yolo26n-pose.pt", device="cuda:0"):
        ...
    def __call__(self, bgr_frame: np.ndarray) -> PerceptionOut:
        """bgr_frame: HxWx3 uint8, 任意尺寸（内部 letterbox 到 320x320）"""

@dataclass
class PerceptionOut:
    feat_raw:    np.ndarray   # (920,) float32，未压缩的 YOLO 内部特征
    feat_64:     np.ndarray   # (64,)  float32，经 FeatureAdapter 压缩，tanh 后 ∈[-1,1]
    theta_img:   float        # 人相对图像中心的水平角，度（正=右）
    person_w:    float        # bbox 宽 / 图宽 ∈[0,1]
    person_h:    float        # bbox 高 / 图高 ∈[0,1]
    person_cx:   float        # bbox 中心 x / 图宽 ∈[0,1]
    person_conf: float
    seen:        int          # 0/1，本帧是否检测到人
    hand_up_L:   int          # 0/1
    hand_up_R:   int          # 0/1
    wave_sin:    float        # ∈[-1,1]
    wave_cos:    float        # ∈[-1,1]
    wave_amp:    float        # 相对肩宽的摆幅，≥0
    gesture_conf: float
    debug:       dict         # 任意调试信息
```

### 3.2 关键实现要求

- **YOLO 内部特征抽取**：参考 `D:\smartcar\code\lnn\extract_features.py` 的 `build_batch_extractor()`（用 forward hook 抓 P3/P4/P5/Detect.cv3[1]）。**输入是 320×320**（整除 32），不是 640。
- **特征维度不预设**：实际维度以运行时打印为准，写进 `FEATURE_DIM` 常量。如果拼不出恰好 920 维，**就用实际维度**，但必须写清楚每个来源的层名和维度。**绝对不要为了凑数瞎 pad。**
- **`theta_img = (person_cx - 0.5) * FOV_H`**，`FOV_H` 默认 60.0，**做成构造参数**。
- **手势基元必须用肩宽归一化**：
  ```
  shoulder_width = |kp[5] - kp[6]|  （左肩右肩 x 差）
  x_rel = (wrist.x - shoulder_mid.x) / shoulder_width
  y_rel = (wrist.y - shoulder_mid.y) / shoulder_width
  ```
  肩宽 < 5 像素时直接判 `hand_up_*=0`（算不出来就别瞎算）。
- **挥手用滑动累积**，不要用瞬时阈值：
  ```
  手腕横向位置过零 → wave_score += 1
  每帧衰减 wave_score *= exp(-dt / 2.0)
  wave_score > 4 → 触发
  ```
- **FeatureAdapter 必须可训练**：`Linear(D_in, 64)`，训练时和 CfC 一起训。A 阶段先给随机初始化版，接口留好 `load_state_dict`。
- 统计量 `running_mean/running_std` 必须能被固定下来（`freeze_stats()`），供部署用。

### 3.3 验收测试 `test_perception.py`

用**自己合成的一张图**（画一个火柴人）或下载一张人物图，跑通并 `assert`：
- `feat_raw.shape == (FEATURE_DIM,)`，无 NaN
- `feat_64.shape == (64,)`，全在 `[-1,1]`
- `0 <= theta_img <= FOV_H/2` 或负值，总之是有限数
- 空图（全黑）时 `seen == 0` 且不崩溃
- 打印每层特征的来源和维度（层名 → 维度）

## 4. 工作流 B（teacher）的产出契约

### 4.1 主入口

```python
# pc/teacher/teacher.py
class TeacherController:
    def __init__(self, cfg: dict): ...
    def step(self, obs: TeacherObs) -> TeacherAction: ...
    def reset(self): ...

@dataclass
class TeacherObs:
    theta_img: float      # 度
    q_head:    float      # 度，头部当前角
    seen:      int
    person_h:  float
    hand_up_L: int
    hand_up_R: int
    wave_score: float     # 来自 A 的累积量，≥0
    t_now:     float      # 秒，单调时间戳

@dataclass
class TeacherAction:
    head_des: float       # 度，∈[-HEAD_MAX, HEAD_MAX]
    vx_des:   float       # 米/秒
    vyaw_des: float       # 度/秒
    mode:     int         # 0=IDLE 1=TRACK 2=ACK 3=REFRACTORY
    emote:    int         # 0..7
    debug:    dict        # eps_body / 各中间量，必须全给
```

### 4.2 核心公式（**必须严格按这个实现**）

```
img_frame == 0 (head_cam):  eps_body = theta_img + q_head
img_frame == 1 (body_cam):  eps_body = theta_img
```

三档分配律见 `TECH-PLAN-v1.md` §7.2，距离控制见 §7.2b。**所有阈值必须从 YAML 读，不许硬编码。**

### 4.3 验收测试 `test_teacher.py`

构造合成轨迹，`assert`：
- `theta_img=0, q_head=0` → `vyaw_des==0` 且 `head_des==0`
- `theta_img=3, q_head=0`（小误差）→ **`vyaw_des == 0`**（只动头）★核心
- `theta_img=30, q_head=0`（大误差）→ `vyaw_des != 0`
- 挥手序列（wave_score 从 0 涨到 5）→ mode 从 TRACK 变 ACK
- ACK 后再喂 `wave_score=5` → **不进 ACK**（不应期生效）
- 连续 1000 步随机输入 → 输出全部有限、全部在范围内
- 打印一条 200 步的轨迹表，人能看懂

## 5. 工作流 C（cfc_host）的产出契约

### 5.1 目标

把 `ncps` 的 CfC cell 用**纯 numpy** 和**纯 C99** 各实现一遍，三者输出逐元素对拍。

### 5.2 数学口径（`ncps.torch.CfCCell` mode="default"，权威）

```
x   = concat(input, hx)                            # (in_dim + hidden)
z   = LeCun(bb_w @ x + bb_b)                       # (backbone_units)   LeCun(v)=1.7159*tanh(0.666*v)
ff1 = tanh(W_ff1 @ z + b_ff1)                      # (hidden)
ff2 = tanh(W_ff2 @ z + b_ff2)                      # (hidden)
t   = sigmoid((W_ta @ z + b_ta)*ts + (W_tb @ z + b_tb))
h_new = ff1*(1-t) + t*ff2                          # ★注意：没有 h 自环项
out   = head @ h_new + b_head
```

在 `h_new` 之后**加保留门**（我们自己的扩展）：
```
lam   = sigmoid(ret)                               # ret 是 (hidden,) 可学习参数，初值 -3.0
h_out = h_new + lam * (hx - h_new)
```
**硬性要求：当 `lam` 全为 0 时，结果必须与上面不带保留门的公式逐位一致。**

### 5.3 C 接口（固件要用这个签名，所以必须长这样）

```c
// cfc_engine.h
typedef struct {
    int input_dim, hidden_dim, backbone_units;
    int use_retention;
    const float *bb_w, *bb_b;        // backbone_units x (input_dim+hidden_dim), backbone_units
    const float *ff1_w, *ff1_b;      // hidden x backbone_units
    const float *ff2_w, *ff2_b;
    const float *ta_w,  *ta_b;
    const float *tb_w,  *tb_b;
    const float *ret;                // hidden，可为 NULL
    float ts;
} cfc_cell_t;

void cfc_init_state(const cfc_cell_t *c, float *h);            // 全部清零
void cfc_step(const cfc_cell_t *c, const float *in, float *h_io);  // in: input_dim, h_io: hidden，原地更新
```

**要能编过 MSVC 的 C 模式**：`cl /c /O2 /fp:precise cfc_engine.c`。不要用 VLA、不要用 C++ 特性、不要 `#include` 平台相关头。

### 5.4 验收测试 `test_parity.py` + `test_engine.c`

对拍三方：**ncps(真值) ↔ numpy ↔ C**

- 随机权重（xavier 初始化，模拟真实训练后分布）
- 输入：随机 + 边界（全 0、全 1、tanh 饱和值 ±10）
- 连续 200 步递推（检验误差累积）
- 判据：**最大绝对误差 < 1e-5**
- `test_parity.py` 一条命令跑完，自动调 `cl.exe` 编译 `test_engine.c`，跑完打印 PASS/FAIL 和最大误差

### 5.5 算力注意

按 `hidden=64, backbone_units=64, input=106` 估算：backbone `(64+64)*64=8192`，ff1/ff2/ta/tb `4*64*64=16384`，合计 **~25k MAC**。这个规模是目标，别搞大。

## 6. 汇报要求

每个工作流完成后，用 `send_message` 给 Lead 发一条**不超过 400 字**的汇报：
1. 每条验收测试的 PASS/FAIL
2. 实际测出来的数字（维度、误差、耗时）
3. 你觉得**哪里最脆弱**（一句话）
4. 有没有发现契约里的错（有就直说）

**不要把代码贴进汇报，Lead 会自己读文件。**
