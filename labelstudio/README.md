# RIG-Hover 标注工作台（Label Studio）

复用 `D:\smartcar\labelstudio` 的安装（同一套 label-studio 包），但**数据目录和端口独立**：
智能车那个实例跑 3333，这个跑 **3334**，两边数据库、项目、用户完全隔离。
共用一个安装只是省一次 pip install，不共享任何数据。

## 一、启动

双击 `start_labelstudio.bat`，或在 `D:\robot_work\labelstudio` 下跑：

```powershell
$env:DJANGO_TELEMETRY="0"
$env:LABEL_STUDIO_DATA_DIR="D:\robot_work\labelstudio\data"
& "D:\smartcar\labelstudio\venv\Scripts\label-studio.exe" start --port 3334 --data-dir "D:\robot_work\labelstudio\data"
```

打开 **http://localhost:3334**

> 当前已经在跑（PID 见 `Get-NetTCPConnection -LocalPort 3334`）。bat 有单实例保护，
> 重复启动只会提示"已在运行"。

## 二、账号

| | |
|---|---|
| 邮箱 | `robot@rig.local` |
| 用户名 | `robot` |
| 密码 | `Rig2026!` |
| API Token | 存在 `.ls_token`（脚本自动读，不用手填） |

项目：**RIG-Hover 场景v2 face/fist/hand**（id=1）

### 只标三类

| 标签 | 快捷键 | 框什么 |
|---|---|---|
| `face` | 1 | 只框脸，不带脖子 |
| `fist` | 2 | 握紧的那一团 |
| `hand` | 3 | 张开或放松的手 |

**为什么必须有 `hand`**：少了它，模型无从知道"这是手但不是拳"，会把所有手都判成拳。

**场景 v2 用不到的手势不标**（wave / pointing / thumbs_up / open_palm）——那是场景 v1
的 ACK 机制，v2 没有 ACK 这个状态。别去标，标了也不会被用。

240×240 很小：**按住 Ctrl 滚轮放大**后再框。

## 三、工作流（照搬智能车那套）

```
新帧 → ls_preannotate.py → 分流
                            ├─ 模型高置信度 → 直接写 YOLO 标注进 data/labeled_inbox/（不用人）
                            └─ 低置信度/零框 → 导入 LS，你在 Data Manager 按分数捞出来人工标
```

### 1. 预标注（模型先猜一遍）

```powershell
# 第一轮必须加 --no-auto：YuNet 的框在桌面域没有经过人工复核，
# 自动入库等于把未验证的标注喂给训练。等人工标过一批、确认框可信后再去掉。
& G:\ai\python11\python.exe ls_preannotate.py --dir D:\robot_work\data\inbox --no-auto
```

注意：**必须用 `G:\ai\python11\python.exe`**（带 torch/ultralytics），
`labelstudio\venv` 里没有这些，跑不了感知链。

分流阈值：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--conf-low` | 0.35 | 低于此分的框不写 |
| `--auto-accept` | 0.70 | 最低框分 ≥ 此值 → 自动写标注；否则上传 LS |
| `--fist-box` | 56 | 拳头合成框边长（像素） |

**拳头框是怎么来的**：现有链没有拳头检测器，只有"是不是拳"的分类概率和腕部坐标。
所以框是**从腕部合成**的——位置对、大小是猜的（默认 56 px），标注时人要改大小。
脸框是 YuNet 直接给的，位置和大小都是真的。

### 2. 在 Label Studio 里只标低置信度的

Data Manager → **Filters** → **Prediction score** → `less` → 填阈值（比如 0.7）
→ 列出来的就是模型没把握的那些。

### 3. 导出成 YOLO 训练集

```powershell
& G:\ai\python11\python.exe ls_export.py --out D:\robot_work\data\dataset --holdout label_none
```

- 文件名前缀就是 session id（`label_fist_00003.jpg` → `label_fist`）
- **按段切分，不按帧**：同一段相邻帧高度相关，按帧随机切等于把验证集漏进训练集
- `--holdout` 指定的段整段进 val；不指定就每 5 段取 1 段
- 预测（模型猜的框）**不算标注**，必须人提交过才算

## 四、当前状态

已导入 **400 帧**（从 `evidence/live/` 抽的，每 3 帧取 1，相邻帧几乎相同全标是浪费）：

| 来源 | 帧数 |
|---|---|
| `label_fist` | 126 |
| `label_none` | 126 |
| `calib_1` | 148 |

预测全部写入成功（400/400），其中 61 帧模型零框 —— 这些正是最需要人标的。
已标注 **0** 帧。

## 五、文件

| 文件 | 作用 |
|---|---|
| `start_labelstudio.bat` | 启动服务 |
| `ls_admin.py` | 建号 / 建项目（**直接连数据库，服务停不停都能跑**） |
| `ls_preannotate.py` | 模型预标注 + 分流 |
| `ls_export.py` | 导出 YOLO 数据集 |
| `reports/preannotate_*.md` | 每轮预标注报告 |
| `data/dataset/data.yaml` | 类别表（`face=0 fist=1 hand=2`），训练共用 |

## 六、踩过的坑

1. **`Authorization: Token xxx` 全部 401**：Label Studio 1.23 默认关闭传统 Token 认证
   （`org.jwt.legacy_api_tokens_enabled`）。`ls_admin.py` 建号时已经打开，
   但换数据目录重建组织时要重新跑一次 `ls_admin.py user ...`。
2. **模型类不能直接 import**：`from label_studio.jwt_auth.models import JWTSettings`
   会报 "doesn't declare an explicit app_label"，必须用
   `django.apps.apps.get_model("jwt_auth", "JWTSettings")`。
3. **必须 `requests.Session().trust_env = False`**：本机有 HTTP_PROXY，
   不开这个连 127.0.0.1 都会被代理吞掉。
4. **`label-studio version` 退出码是 1**：它只是打印版本号，不是真失败。
5. **预标注会移动源文件**到 `data/done/round_<ts>/`。所以**别直接对
   `evidence/live/` 跑**——那是我留的证据。先复制到 `data/inbox/`。
