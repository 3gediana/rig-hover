# -*- coding: utf-8 -*-
"""vlm_label.py -- 用视觉语言模型自动标注图片（本地权重 / 国内 API 两种后端）。

为什么有这个东西
================

真机数据集里**唯一需要人手标的**是手部手势类别（fist / palm / ...），因为
那 64px 手图是外观分类器的输入，而 COCO-17 没有手指关键点。本文档配套的
`DATASET-AND-ANNOTATION.md` 讨论"到底要不要人标"。结论是：

* CfC 的行为标签由教师状态机逐帧生成 —— **不需要标**；
* 人person 框 / 关键点由 yolo26n-pose 预训练权重直接给 —— **不需要标**；
* 敲击事件用录制时按空格打标记 —— **不需要事后标**；
* 只剩手图类别这一项。它可以用两种零人工方式解决：
  (a) 用 HaGRID 训出来的分类器自标 + 只人工复核低置信度那部分；
  (b) 用 VLM 打第二意见。

本文件实现 (b)，并且刻意做成"backend 无关"：本地 Qwen3-VL 与阿里百炼 /
智谱的 OpenAI 兼容接口走同一套代码，换后端只改参数。

三种用法
========

1) 分类（最常用）::

    python vlm_label.py --backend local --classify \
        --images D:\\g\\crops --classes fist,palm,none --out crops.json

2) 打框::

    python vlm_label.py --backend qwen --detect \
        --images D:\\g\\frames --classes person,hand --out det.json

3) 当库用（bench 脚本就是这么用的）::

    from vlm_label import make_backend
    be = make_backend("local", model="G:/ms/Qwen/Qwen3-VL-4B-Instruct")
    be.classify(pil_img, ["fist", "palm", "none"])

后端
====

``local``   transformers + 本地权重（Qwen3-VL 全系 / Qwen2.5-VL）。8 GB 卡上
            4B 走 4bit 量化；量化失败会自动退回 fp16 + device_map=auto（慢但能跑）。
``qwen``    阿里百炼 OpenAI 兼容端点，默认 ``qwen3-vl-plus``。
``glm``     智谱 BigModel OpenAI 兼容端点，默认 ``glm-4.6v-flash``（当前免费）。
``mock``    不加载任何模型，按文件名里的关键字返回答案 —— 只用于跑通流程和单测。

坐标约定
========

所有对外暴露的框都是 **归一化 [x1,y1,x2,y2]，0–1，左上原点**。各家的原始格式
（Qwen 的 0–1000、Florence 的 0–999 bin、Gemini 的 [ymin,xmin,ymax,xmax]）
一律在这里转换，调用方不需要知道差异。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys

__all__ = [
    "VLMBackend", "LocalQwenBackend", "OpenAICompatBackend", "MockBackend",
    "make_backend", "parse_boxes", "parse_label", "to_data_url",
    "DETECT_PROMPT", "CLASSIFY_PROMPT",
]

#: 打框用的默认提示词。要求 JSON 输出，因为自由文本各家格式不一样，
#: 而 JSON 可以只写一个解析器。
DETECT_PROMPT = (
    "Detect every instance of these categories in the image: {classes}.\n"
    "Answer with a JSON array only, no prose, no markdown fence. Each element:\n"
    '{"label": "<one of {classes}>", "bbox_2d": [x1, y1, x2, y2]}\n'
    "Coordinates are integers normalised to 0-1000 with the origin at the "
    "top-left of the image. Output [] when nothing matches."
)

#: 分类用的默认提示词。**必须**允许 "unsure"：不限定时 VLM 对看不清的目标也会
#: 给一个自信的答案，于是校准指标全失真。
CLASSIFY_PROMPT = (
    "Look at the image and decide which single category it shows: {classes}.\n"
    "Answer with exactly one category name and nothing else. "
    "If the image is too small, blurry or ambiguous to decide, answer unsure."
)


# ----------------------------------------------------------------------
# 后端
# ----------------------------------------------------------------------
class VLMBackend:
    """所有后端的共同接口。子类只实现 :meth:`chat`。"""

    name = "base"

    def chat(self, images, prompt: str, max_new_tokens: int = 128) -> str:
        """把若干张 PIL 图和一句提示词发给模型，返回纯文本。"""
        raise NotImplementedError

    # -- 建立在上面的两个便利方法 --------------------------------------
    def detect(self, image, classes, prompt: str = "") -> list:
        """返回 ``[(label, [x1,y1,x2,y2] 归一化), ...]``。"""
        text = self.chat([image], (prompt or DETECT_PROMPT).format(
            classes=", ".join(classes)))
        return parse_boxes(text, classes)

    def classify(self, image, classes, prompt: str = "") -> str:
        """返回类别名；模型说 "unsure" 就返回 ``"unsure"``。"""
        text = self.chat([image], (prompt or CLASSIFY_PROMPT).format(
            classes=", ".join(classes)), max_new_tokens=16)
        return parse_label(text, classes)


class MockBackend(VLMBackend):
    """离线后端：按文件名猜答案。用来验证调用链，不代表任何模型性能。

    设 ``be.MOCK_TRUTH = "fist"`` 可以让它回显指定的真值 —— 用于验证
    "渲染 → 提问 → 解析 → 统计"整条链路，跑出来应当接近 100%。
    """

    name = "mock"

    def chat(self, images, prompt, max_new_tokens=128) -> str:
        truth = getattr(self, "MOCK_TRUTH", None)
        if truth:
            return str(truth)
        name = getattr(images[0], "filename", "") or ""
        for word in re.findall(r"[A-Za-z_]+", name):
            if word.lower() in prompt.lower():
                return word
        return "unsure"


class LocalQwenBackend(VLMBackend):
    """本地 transformers 权重（Qwen3-VL / Qwen2.5-VL）。

    Args:
        model: 本地目录或 HF/ModelScope 名字。
        device: ``"cuda"`` / ``"cpu"``。
        four_bit: 8 GB 卡上 4B 模型必须开；加载失败自动退回 fp16 + offload。
    """

    name = "local"

    def __init__(self, model, device="cuda", four_bit=True, verbose=True):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:  # transformers < 4.46
            from transformers import AutoModelForVision2Seq as AutoVLM
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        kw = {"trust_remote_code": True, "device_map": device}
        if four_bit:
            try:
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True)
                kw["dtype"] = torch.bfloat16
                self.model = AutoVLM.from_pretrained(model, **kw)
            except Exception as exc:  # noqa: BLE001 - 量化失败要能退回去
                if verbose:
                    print("[vlm] 4bit 加载失败，退回 fp16 + device_map=auto：%s"
                          % str(exc)[:200], file=sys.stderr)
                kw.pop("quantization_config", None)
                kw["dtype"] = torch.float16
                self.model = AutoVLM.from_pretrained(model, **kw)
        else:
            kw["dtype"] = torch.float16
            self.model = AutoVLM.from_pretrained(model, **kw)
        self.model.eval()
        if verbose:
            n = sum(p.numel() for p in self.model.parameters()) / 1e9
            print("[vlm] %s  %.2f B params  device=%s"
                  % (os.path.basename(str(model)), n, device), file=sys.stderr)

    def chat(self, images, prompt, max_new_tokens=128) -> str:
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
        dev = next(self.model.parameters()).device
        inputs = {k: (v.to(dev) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen, skip_special_tokens=True).strip()


class OpenAICompatBackend(VLMBackend):
    """任何 OpenAI 兼容的 ``/chat/completions``，图片走 data URL。

    国内可选：
        qwen -> https://dashscope.aliyuncs.com/compatible-mode/v1
        glm  -> https://open.bigmodel.cn/api/paas/v4
        sf   -> https://api.siliconflow.cn/v1
    """

    name = "api"

    def __init__(self, base_url, model, api_key=None, env_key="",
                 max_side=1024, verbose=True):
        from openai import OpenAI
        key = api_key or os.environ.get(env_key) or ""
        if not key:
            raise SystemExit(
                "没有 API key：设 %s 或传 --api-key。"
                "（zhipu 的 glm-4.6v-flash 目前免费，注册就有额度。）" % env_key)
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model = model
        self.max_side = max_side
        if verbose:
            print("[vlm] api %s  model=%s" % (base_url, model), file=sys.stderr)

    def chat(self, images, prompt, max_new_tokens=256) -> str:
        content = [{"type": "text", "text": prompt}]
        for im in images:
            content.append({"type": "image_url",
                            "image_url": {"url": to_data_url(im, self.max_side)}})
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_new_tokens, temperature=0.0)
        return (r.choices[0].message.content or "").strip()


BACKENDS = {
    "local": (LocalQwenBackend, {}),
    "qwen": (OpenAICompatBackend, dict(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen3-vl-plus", env_key="DASHSCOPE_API_KEY")),
    "glm": (OpenAICompatBackend, dict(
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model="glm-4.6v-flash", env_key="ZHIPUAI_API_KEY")),
    "sf": (OpenAICompatBackend, dict(
        base_url="https://api.siliconflow.cn/v1",
        model="Qwen/Qwen3-VL-8B-Instruct", env_key="SILICONFLOW_API_KEY")),
    "mock": (MockBackend, {}),
}


def make_backend(kind: str, **kw) -> VLMBackend:
    """按名字建后端；``local`` 需要 ``model=``，API 后端需要对应的环境变量。"""
    if kind not in BACKENDS:
        raise SystemExit("未知 backend %r，可选：%s"
                         % (kind, ", ".join(sorted(BACKENDS))))
    cls, defaults = BACKENDS[kind]
    cfg = dict(defaults)
    cfg.update({k: v for k, v in kw.items() if v is not None})
    if kind == "local":
        return cls(cfg.pop("model"))
    return cls(**cfg)


# ----------------------------------------------------------------------
# 解析
# ----------------------------------------------------------------------
def to_data_url(image, max_side=1024) -> str:
    """PIL 图 → data URL；顺手把长边压到 ``max_side``，省 token。"""
    im = image.convert("RGB")
    if max_side and max(im.size) > max_side:
        s = max_side / float(max(im.size))
        im = im.resize((max(1, int(im.size[0] * s)),
                        max(1, int(im.size[1] * s))), 3)  # 3 = BICUBIC
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


_JSON_ARR = re.compile(r"\[.*\]", re.S)
_JSON_OBJ = re.compile(r"\{.*\}", re.S)


def parse_boxes(text: str, classes=()) -> list:
    """从模型输出里抠出框。

    容忍三种脏输出：纯 JSON、```json 围栏、句子中间夹一个 JSON 数组。
    坐标按 0–1000 归一化理解；若最大坐标 ≤1 则认为是已经归一化到 0–1。
    """
    if not text:
        return []
    m = _JSON_ARR.search(text) or _JSON_OBJ.search(text)
    if not m:
        return []
    blob = m.group(0)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        # 常见毛病：尾逗号、单引号
        fixed = re.sub(r",\s*([\]}])", r"\1", blob).replace("'", '"')
        try:
            data = json.loads(fixed)
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict):
        data = [data]
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        box = (item.get("bbox_2d") or item.get("bbox") or item.get("box")
               or item.get("xyxy"))
        if not box or len(box) != 4:
            continue
        try:
            v = [float(t) for t in box]
        except (TypeError, ValueError):
            continue
        if max(v) > 1.5:
            v = [t / 1000.0 for t in v]
        v = [min(max(t, 0.0), 1.0) for t in v]
        if v[2] <= v[0] or v[3] <= v[1]:
            continue
        label = str(item.get("label") or item.get("name") or "").strip().lower()
        if classes and label not in [c.lower() for c in classes]:
            # 模型自造了类别名：只有唯一的候选时才接受，否则丢掉
            if len(classes) != 1:
                continue
            label = classes[0].lower()
        out.append((label, v))
    return out


def parse_label(text: str, classes) -> str:
    """把模型输出收敛到一个类别名（或 ``unsure``）。"""
    if not text:
        return "unsure"
    low = text.strip().lower()
    low = re.sub(r"[^a-z_\s,]", " ", low)
    toks = [t for t in re.split(r"[\s,]+", low) if t]
    names = [c.lower() for c in classes]
    for t in toks:
        if t in names:
            return t
    if "unsure" in low or "cannot" in low or "unclear" in low:
        return "unsure"
    return "unsure"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _iter_images(path):
    from PIL import Image
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.lower().endswith(exts):
                files.append(os.path.join(path, name))
    else:
        files = [path]
    for f in files:
        im = Image.open(f).convert("RGB")
        im.filename = os.path.basename(f)
        yield os.path.basename(f), im


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="用 VLM 自动标注（本地权重或国内 API）")
    ap.add_argument("--backend", default="mock",
                    choices=sorted(BACKENDS), help="本地 / API / mock")
    ap.add_argument("--model", default="", help="local 后端的权重路径")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--images", required=True, help="图片目录或单张图")
    ap.add_argument("--out", default="", help="输出 JSON 路径")
    ap.add_argument("--limit", type=int, default=0)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--classify", action="store_true", help="整图分类")
    g.add_argument("--detect", action="store_true", help="出框")
    ap.add_argument("--classes", default="fist,palm,none",
                    help="逗号分隔的类别名")
    ap.add_argument("--prompt", default="", help="覆盖默认提示词")
    ap.add_argument("--max-side", type=int, default=1024)
    a = ap.parse_args(argv)

    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    kw = {"max_side": a.max_side} if a.backend != "local" else {}
    if a.backend == "local":
        if not a.model:
            raise SystemExit("--backend local 需要 --model <权重目录>")
        be = make_backend("local", model=a.model)
    else:
        be = make_backend(a.backend,
                          api_key=a.api_key or None,
                          model=a.model or None, **kw)

    rows = []
    for i, (name, im) in enumerate(_iter_images(a.images)):
        if a.limit and i >= a.limit:
            break
        try:
            if a.classify:
                pred = be.classify(im, classes, a.prompt)
                rows.append({"image": name, "label": pred})
            else:
                boxes = be.detect(im, classes, a.prompt)
                rows.append({"image": name,
                             "boxes": [{"label": lb, "bbox_xyxy_norm": bb}
                                       for lb, bb in boxes]})
        except Exception as exc:  # noqa: BLE001 - 单张失败不该中断整批
            rows.append({"image": name, "error": str(exc)[:200]})
        if (i + 1) % 20 == 0:
            print("  %d done" % (i + 1), file=sys.stderr)

    text = json.dumps(rows, ensure_ascii=False, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print("wrote %s  (%d rows)" % (a.out, len(rows)))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
