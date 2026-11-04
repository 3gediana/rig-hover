# 手势识别：公开数据集训练 + 感知端事件化

把「挥手/停止/认可/指向/握拳」从**手工阈值 + 电平信号**改造成
**公开数据训练的分类器 + 事件化分割**。

一句话结论：同一段连续挥手现在**只产生 1 个 onset**（边沿），
两次挥手（间隔 ≥0.6 s）产生 2 个；触发延迟 837 ms（旧实现 3567 ms 且 90% 漏检）；
静止场景误报 **0 次/8 分钟**。

---

## 1. 为什么要在感知端事件化

连续动作 → 逐帧概率 → 让下游网络自己推断「这一串帧里什么时候发生了一次手势」，
这是把**边界检测**塞给了学习器。而 `ncps` 的 CfC 没有 h 自环项，实测记忆时标
只有 ~2.2 s（`D:\smartcar\code\lnn\reports\cfc_memory_20261030.md`），结构上做不好
这个长时程推断。旧实现里的「2 s 衰减 + 冷却」就是替它补记忆，属于补丁。

所以边界判定放在**感知端**用确定性状态机做，输出 `gesture_onset` 这种**边沿**信号，
下游只学「事件 → 动作」：

```
状态机: IDLE → PENDING(候选积累) → GESTURE(已确认) → RETURN(手落下) → IDLE
```

每帧输出（字段名与验收脚本一致，可直接并入 `PerceptionOut`）：

| 字段 | 含义 |
|---|---|
| `gesture_id` | 0=none 1=wave 2=open_palm 3=thumbs_up 4=pointing 5=fist |
| `gesture_onset` | **边沿脉冲**：进入 GESTURE 那一帧 =1，其余 0 |
| `gesture_phase` | 0=IDLE 1=PENDING 2=GESTURE 3=RETURN |
| `gesture_conf` | 0..1，当前候选/已确认类的平滑概率 |
| `gesture_gap_s` | 距上次 GESTURE 结束的秒数（连续量，上限 `gap_cap_s`） |

三个时间参数的分工（都外置在 `configs/gesture_events.yaml`）：

* `pending_hold`（0.10 s）：**确认窗口**，去抖。
* `release`（0.25 s）：概率掉下去多久算动作结束。
* `min_gap`（0.25 s）：两个 onset 的最小间隔，**只**滤掉紧邻的重复边沿。
  人手再做一个动作要 >400 ms，比它大一个数量级，所以它没有能力「忘记」任何东西。

区别「同一段动作的延续」和「新动作」靠的是 **RETURN 相位**：连续挥手时手不会
真的放下来，进不了 IDLE；两次挥手之间手会落下，RETURN 被 idle 清空，下一次才有
新边沿。如果 RETURN 里动作又回到 enter 阈值之上（手没放下，只是概率抖了一下），
事件**恢复**而不发新 onset——否则「举手不动」会被拆成两次。

---

## 2. 数据

**HaGRID 30k 采样集**（`cj-mills/hagrid-sample-30k-384p`，1.0 GB，CC-BY-SA 4.0）。
选它的理由：18+ 类手势、真人网络摄像头场景、**官方标注带 `user_id`**，
所以「按人划分」是可验证的，不是嘴上说说。

| 项目 | 数量 |
|---|---|
| 图片 | 31,833（全部 20 类） |
| 取用的 7 类（含 idle） | 12,536 张 |
| 其中可用骨架帧（YOLO26n-pose 检到人且肩宽够） | 13,747 帧（含 other 1,224） |
| 受试者 | 5,051 个（`user_id`） |
| train / val / test（**按人**） | 3,296 / 706 / 706 人，三份 user 列表两两不相交 |

类别映射（HaGRID → 本项目）：

| 本项目 | HaGRID 类 | 帧数 |
|---|---|---|
| `idle` | `no_gesture` | 3,094 |
| `open_palm` | `palm` + `stop` + `stop_inverted` | 4,789 |
| `thumbs_up` | `like` | 1,510 |
| `pointing` | `one`（食指单独竖起） | 1,569 |
| `fist` | `fist` | 1,561 |
| **`wave`** | HaGRID **没有**这个类 | 合成（见下） |
| 内部 `other` | `call`/`dislike`/`four`/`mute`/`ok`/`peace` | 1,224 |
| 严格留出（**不进训练**） | `rock`/`three`/`two_up`/… 7 类 | 1,274 |

两个要点：

1. **`no_gesture` 就在官方标注里**（3,098 张，混在各类手势目录中——拍摄时要求
   做手势但实际没做出来的图）。按目录名取会把它们当成手势样本，idle 就剩 0 个。
2. **`wave` 是合成的**：HaGRID 是静态图集，没有挥手。这里用真实 `palm` 姿势当基帧，
   把手腕沿横向正弦轨迹摆动（0.5~2.5 Hz、0.35~0.80 肩宽），肘关节用**2 连杆 IK**
   解出来（保留真实大臂/小臂长度），再叠真实量级的关键点噪声（σ≈0.012 肩宽）。
   基帧的手若在腰际，中位会被抬到肩线以上——真人挥手本来就在这个高度。

---

## 3. 分类器

`src/model.py`：**骨架 1D 空洞卷积（主）+ 手部裁剪小 CNN（辅）**，192,791 参数，
12 帧因果滑窗（@30fps ≈ 400 ms）。

* **骨架分支**：每帧 53 维（17 关键点 × (x_rel, y_rel) + 17 置信度 + 肩宽/图宽 + 有效位），
  肩宽归一化（与 `pc/perception/gesture.py` 同一套坐标约定）；4 层 1D 卷积（膨胀 1/2/4）
  看窗内运动。挥手和「举着不动」在单帧上姿势相同，差别只在运动，这一层负责区分。
* **外观分支**：按「举得更高的那只手腕」裁 32×32 灰度（边长 0.40 肩宽，中心向手的方向
  外推 0.22 肩宽），两层小卷积。

**为什么必须有外观分支**（不是偷懒，是消融证实的）：COCO-17 没有手指关键点，
`fist`/`open_palm`/`pointing`/`thumbs_up` 在骨架上是同一个姿势。纯骨架消融
（`models/gesture_classifier_skeleton.pt`）在测试集上 `pointing` F1 = **0.000**、
`fist` F1 = **0.033**；加外观分支后分别是 0.574 / 0.846。详见
[`reports/classifier.md`](reports/classifier.md)。

### 测试集结果（706 个留出用户，8000 个窗口）

| 类别 | precision | recall | F1 | 纯骨架 F1 |
|---|---|---|---|---|
| idle | 0.871 | 0.897 | **0.884** | 0.908 |
| wave | 0.957 | 0.983 | **0.970** | 0.971 |
| open_palm | 0.728 | 0.945 | **0.822** | 0.405 |
| thumbs_up | 0.725 | 0.745 | **0.735** | 0.746 |
| pointing | 0.901 | 0.421 | **0.574** | 0.000 |
| fist | 0.940 | 0.769 | **0.846** | 0.033 |

宏平均 F1 **0.805**，准确率 **0.854**。

---

## 4. 标定（任务 C）

`src/calibrate_events.py` 用带**已知动作起点**的长序列标定，`src/sweep_events.py`
把「延迟 ↔ 误报」的取舍曲线打出来。标定集含 3 分钟纯静止 + 2 分钟未见手势
（只用 20 s 静止去标定会选出「延迟最低但误报 3 次/分钟」的参数——第一版就是这样翻的车）。

同一批序列上：

| 指标 | 标定前（`wave_score > 4` + 2 s 衰减） | 标定后（事件机） |
|---|---|---|
| 挥手触发延迟 | 3567 ms（且只有 1/10 条能触发） | **837 ms** |
| 挥手漏检率 | 90% | **0%** |
| 连续挥手 onset 数 | 电平：一直亮着 | **恒为 1** |
| 两次挥手（间隔 1 s） | 无法表达 | **2 个 onset** |
| 静止误报 | 0 次 | **0 次/8 分钟** |
| 举手不动 | 无法表达（只认挥手） | **1 个 onset** |

旧实现为什么不行的实测（`src/diag_old_algo.py`）：`score ≈ 过零率 × 2 s`，
1 Hz 慢挥的稳态刚好是 **4.0**，而阈值就是 `>4`——永远贴边运行，实测峰值 2.17。
另外 4 s 内活动手被切换了 **44~59 次**（按 `|x_rel|` 选，关键点一抖就换手），
每次切换都会清空过零状态。

详细数字与参数表见 [`reports/events.md`](reports/events.md)。

---

## 5. 单帧耗时（RTX 5060 Laptop，imgsz=320）

| 环节 | 中位 | p90 |
|---|---|---|
| 关键点提取（YOLO26n-pose, batch=1） | 7.7 ms | 12.6 ms |
| 关键点提取（batch=8） | 1.6 ms/帧 | 2.2 ms/帧 |
| 分类：骨架特征 + 手部裁剪 | 0.07 ms | 0.08 ms |
| 分类：12 帧滑窗网络（GPU / CPU） | 1.26 / 1.49 ms | 1.73 / 1.62 ms |
| **分类端到端（像素 → 概率）** | **1.57 ms** | 1.61 ms |

分类只占单帧总耗时（~9 ms）的 17%，30 Hz 预算内绰绰有余。

---

## 6. 怎么跑

```bash
# 0) 数据（1.0 GB；HF 单连接会被限速，脚本用 8 线程分块下）
python src/download_hagrid.py
python src/prepare_hagrid.py            # 解压 + 生成 data/manifest.csv

# 1) 骨架缓存（15,036 张图，~2.5 分钟）
python src/extract_skeletons.py --batch 16

# 2) 按人划分 + 生成窗口（~3 分钟）
python src/build_dataset.py

# 3) 训练（融合 / 纯骨架消融）
python src/train_classifier.py --epochs 18
python src/train_classifier.py --skeleton-only --epochs 18

# 4) 评测与标定
python src/eval_classifier.py --device cuda:0 --tag fused
python src/calibrate_events.py            # 网格搜索 + 写回 YAML
python src/sweep_events.py                # 延迟↔误报 取舍曲线
python src/scan_false_alarms.py --minutes 8

# 5) 验收（一条命令，打印 PASS/FAIL）
python test_gesture.py

# 6) 报告
python src/make_report.py
```

所有命令都用 `D:\smartcar\.venv\Scripts\python.exe`。

---

## 7. 集成到现有感知管线

```python
from gesture_classifier import GestureClassifier
from gesture_events import GestureEventMachine

clf = GestureClassifier.load(device="cuda:0")     # models/gesture_classifier.pt
evt = GestureEventMachine("configs/gesture_events.yaml")

# 每帧：run_perception 已经算好 kp_xy / kp_conf（原图像素）
probs = clf.push_kp(kp_xy, kp_conf, frame_bgr)    # 无人时用 clf.push_invalid()
out = evt.update(probs, dt)                       # dt 用真实帧间隔
# out.gesture_id / gesture_onset / gesture_phase / gesture_conf / gesture_gap_s
```

`PerceptionOut` 只需加这 5 个字段（或先塞进 `debug`），教师控制器的
`wave_score > 4` + 冷却逻辑可以整段删掉，改成看 `gesture_onset`。

---

## 8. 局限（读数字前必看）

1. **`wave` 没有真人数据验证**。训练/测试样本都是「真实 palm 姿势 + IK 合成摆动」，
   所以 0.97 的 F1 证明的是「模型能从这种运动分布里认出挥手」，**不等于**真人挥手
   已验证。真机验证要靠 TECH-PLAN §9.1 场景 7 的实录。这是最可能在真机上翻车的一条。
2. **`pointing` 只有 0.574**。COCO-17 没有手指关键点，32×32 裁剪里手只有 ~20 px，
   食指是否伸出几乎是亚像素信息。真要用 pointing，得换手部 21 关键点模型
   （MediaPipe Hands / YOLO hand-pose），本管线做不到。
3. **开放集误报**：7 类完全没进训练的手势，被认成「有手势」的比例 27.5%，
   实测 **1.12 次/分钟**误触发。静止场景是 0 次/8 分钟，但「人比了个 rock」会误触发。
4. **手部裁剪依赖「举得更高的那只手腕」**。双手都在画面里且都抬起时，规则可能在两手
   之间跳，裁剪内容随之跳变。训练时用同一规则 + 抖动覆盖了一部分，但没验证过。
5. 分类器的 `idle` 误判主要在 `idle ⇄ thumbs_up`（thumbs-up 在胸口高度做，
   和「手放在身前」在 32×32 上很像）。`thumbs_up` precision 0.725 是这条的直接体现。
6. 距离/姿态极端时（肩宽 < 5 px、侧身、遮挡）关键点不可信，管线靠 `valid` 位退回 idle；
   这些帧没有单独评测。

---

## 9. 文件清单

```
gesture_events.py           事件机（重点交付物）；参数全部外置
gesture_classifier.py       部署用滑窗分类器（含 push_kp / push_invalid）
test_gesture.py             验收：一条命令，PASS/FAIL
configs/gesture_events.yaml 事件机全部参数（标定后写回）
src/gesture_common.py       训练与部署共用的骨架特征定义（单一真源）
src/download_hagrid.py      并行分块下载（curl/schannel 对 HF CDN 握手失败）
src/prepare_hagrid.py       解压 + 官方标注 → manifest
src/extract_skeletons.py    YOLO26n-pose → 关键点 + 手部裁剪缓存
src/synth.py                IK 合成动态序列（真实姿势 + 合成运动）
src/episodes.py             带真值起点的长序列
src/build_dataset.py        按人划分 + 生成训练窗口
src/model.py                骨架 1D-CNN + 手部外观分支
src/train_classifier.py     训练（含纯骨架消融）
src/eval_classifier.py      测试集指标 + 开放集误报
src/calibrate_events.py     事件参数标定（前后对比）
src/sweep_events.py         延迟↔误报 取舍曲线
src/scan_false_alarms.py    误报率长扫描（次/分钟）
src/bench_latency.py        单帧耗时（关键点/分类分开）
src/make_report.py          从 JSON 产物生成 reports/*.md
reports/classifier.md       每类 F1、混淆矩阵、开放集
reports/events.md           标定前后、逐序列延迟、参数表、耗时
reports/*.json              原始指标（数字都从这里来）
```
