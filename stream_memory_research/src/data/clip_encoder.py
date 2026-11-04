"""Frozen CLIP visual encoder -> (feat, tok) feature cache.

This is *the* encoder for every dataset in the project (DESIGN.md §1.1: the
visual encoder is frozen and shared so that all models compare on identical
features).  Never change the model id or the pooling without bumping
``ENCODER_TAG`` -- different versions must not be mixed in one manifest.

Spec (identical for every dataset)
----------------------------------
* ``transformers.CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")``
  ``.eval().half().cuda()`` under ``torch.no_grad()``.
* Preprocess exactly like ``CLIPImageProcessor``: RGB, resize shortest edge to
  224 (bicubic), center crop 224x224, CLIP mean/std normalise.
* ``last_hidden_state`` is [B,197,768]; token 0 = CLS -> ``feat`` [B,768].
* Patch tokens [B,1:,768] -> (14,14,768) -> ``adaptive_avg_pool2d`` (2,2)
  -> flatten -> ``tok`` [B,4,768].
* ROI: crop the GT box with 10% margin, resize to 224x224 (plain stretch),
  same encoder -> [B,768].

All features are returned float16 (CPU numpy) because 800k frames x 768 dims
is ~1.2 GB in f16 and 2.5 GB in f32, and the models consume f32 anyway after
a cast inside the dataset.

The preprocessing runs on the GPU: frames arrive as uint8 HWC numpy, get
stacked to a uint8 tensor, moved once, then divided/scaled in fp16.  That is
2-3x faster than doing per-image float work on the CPU and it means the CPU
only ever does cv2.resize.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

MODEL_NAME = "openai/clip-vit-base-patch16"
ENCODER_TAG = "clip-vit-base-p16-224-f16-2x2tok-v1"   # bump on any change
FEAT_DIM = 768
N_TOK = 4
TOK_DIM = 768
INPUT_SIZE = 224
ROI_MARGIN = 0.10

# CLIP normalisation constants (must match CLIPImageProcessor)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


# --------------------------------------------------------------------------- #
# CPU-side geometry (cv2) -- fast, batchable, no PIL
# --------------------------------------------------------------------------- #
def resize_shortest_center_crop(img: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """RGB uint8 HWC -> exact CLIP geometry: shortest edge -> size, center crop.

    Uses INTER_CUBIC which is what CLIPImageProcessor uses (PIL BICUBIC);
    cv2's cubic is within a few grey levels of PIL's, verified in
    scripts/check_encoder.py against the real CLIPImageProcessor.
    """
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    scale = size / min(h, w)
    nh, nw = max(size, int(round(h * scale))), max(size, int(round(w * scale)))
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    y0 = (nh - size) // 2
    x0 = (nw - size) // 2
    out = out[y0:y0 + size, x0:x0 + size]
    if out.shape[0] != size or out.shape[1] != size:      # paranoia
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(out)


def crop_box_with_margin(img: np.ndarray, box: Sequence[float],
                         margin: float = ROI_MARGIN,
                         out_size: int = INPUT_SIZE) -> np.ndarray:
    """Crop an (x, y, w, h) pixel box with a relative margin, stretch to out_size.

    ``box`` may be outside the image; it is clamped.  A degenerate box yields a
    heavily upscaled tiny crop -- the caller decides validity, this function
    never fails.
    """
    h, w = img.shape[:2]
    x, y, bw, bh = [float(v) for v in box[:4]]
    mx, my = bw * margin, bh * margin
    x0 = int(np.floor(x - mx))
    y0 = int(np.floor(y - my))
    x1 = int(np.ceil(x + bw + mx))
    y1 = int(np.ceil(y + bh + my))
    # give degenerate boxes at least a few pixels of context
    if x1 - x0 < 2:
        x0, x1 = x0 - 1, x1 + 1
    if y1 - y0 < 2:
        y0, y1 = y0 - 1, y1 + 1
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w, x1))
    y1 = max(y0 + 1, min(h, y1))
    patch = img[y0:y1, x0:x1]
    out = cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(out)


# --------------------------------------------------------------------------- #
# GPU encoder
# --------------------------------------------------------------------------- #
class CLIPEncoder:
    """Frozen CLIP ViT-B/16 image encoder with OOM auto-backoff."""

    def __init__(self, device: str = "cuda", batch_size: int = 128,
                 min_batch: int = 8, verbose: bool = True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size)
        self.min_batch = int(min_batch)
        self.verbose = verbose
        t0 = time.time()
        from transformers import CLIPVisionModel
        self.model = CLIPVisionModel.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model.to(self.device).to(self.dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)
        # normalisation tensors on device, NCHW
        self._mean = torch.tensor(CLIP_MEAN, dtype=self.dtype,
                                  device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(CLIP_STD, dtype=self.dtype,
                                 device=self.device).view(1, 3, 1, 1)
        self.n_calls = 0
        self.n_images = 0
        if verbose:
            print(f"[clip] loaded {MODEL_NAME} on {self.device} "
                  f"({self.dtype}) in {time.time() - t0:.1f}s")

    # -- properties ---------------------------------------------------------
    @property
    def output_spec(self) -> Dict[str, object]:
        return {"name": MODEL_NAME, "tag": ENCODER_TAG, "input": INPUT_SIZE,
                "pool": "2x2", "n_tok": N_TOK, "tok_dim": TOK_DIM,
                "feat_dim": FEAT_DIM, "dtype": "float16",
                "aggregation": "CLS token; patch tokens avgpool 14x14 -> 2x2",
                "roi_margin": ROI_MARGIN, "preprocess": "clip_image_processor"}

    # -- core ---------------------------------------------------------------
    def _to_input(self, imgs_u8: np.ndarray) -> torch.Tensor:
        """[B,H,W,3] uint8 RGB -> [B,3,224,224] normalised half tensor."""
        assert imgs_u8.dtype == np.uint8 and imgs_u8.ndim == 4, imgs_u8.shape
        B = imgs_u8.shape[0]
        x = torch.from_numpy(np.ascontiguousarray(imgs_u8))
        if x.shape[1] != INPUT_SIZE or x.shape[2] != INPUT_SIZE:
            # resize path only used if a caller passes raw images
            x = torch.nn.functional.interpolate(
                x.permute(0, 3, 1, 2).float(), size=(INPUT_SIZE, INPUT_SIZE),
                mode="bilinear", align_corners=False).round().clamp(0, 255).byte()
        else:
            x = x.permute(0, 3, 1, 2)
        x = x.to(self.device, non_blocking=True).to(self.dtype).div_(255.0)
        return (x - self._mean) / self._std

    @torch.no_grad()
    def _forward(self, pixel: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        out = self.model(pixel_values=pixel).last_hidden_state   # [B,197,768]
        B = out.shape[0]
        feat = out[:, 0]                                          # [B,768]
        patches = out[:, 1:]                                      # [B,196,768]
        side = int(round(patches.shape[1] ** 0.5))                # 14
        p = patches.reshape(B, side, side, -1).permute(0, 3, 1, 2)  # B,C,14,14
        p = torch.nn.functional.adaptive_avg_pool2d(p.float(), (2, 2))  # B,C,2,2
        from math import sqrt
        n_tok_side = int(round(sqrt(N_TOK)))
        if n_tok_side * n_tok_side != N_TOK:
            raise ValueError(f"N_TOK={N_TOK} is not a perfect square")
        tok = p.reshape(B, -1, N_TOK).permute(0, 2, 1).contiguous()     # B,4,768
        return (feat.to(torch.float16).cpu().numpy(),
                tok.to(torch.float16).cpu().numpy())

    def encode(self, imgs_u8: List[np.ndarray] | np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode a list of RGB uint8 HWC images.

        Returns ``feat`` [N,768] float16 and ``tok`` [N,4,768] float16.
        Input images are resized/cropped to CLIP geometry here if they are not
        already 224x224.
        """
        if isinstance(imgs_u8, np.ndarray) and imgs_u8.ndim == 4:
            arr = imgs_u8
        else:
            arr = np.stack([np.asarray(im, dtype=np.uint8) for im in imgs_u8], 0)
        N = arr.shape[0]
        if N == 0:
            return (np.zeros((0, FEAT_DIM), np.float16),
                    np.zeros((0, N_TOK, TOK_DIM), np.float16))
        feats, toks = [], []
        i = 0
        while i < N:
            bs = min(self.batch_size, N - i)
            try:
                batch = arr[i:i + bs]
                x = self._to_input(batch)
                f, t = self._forward(x)
                feats.append(f)
                toks.append(t)
                self.n_images += bs
                self.n_calls += 1
                i += bs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.batch_size <= self.min_batch:
                    raise
                self.batch_size = max(self.min_batch, self.batch_size // 2)
                if self.verbose:
                    print(f"[clip] OOM -> batch_size={self.batch_size}")
        return np.concatenate(feats, 0), np.concatenate(toks, 0)

    # -- convenience --------------------------------------------------------
    def encode_frames(self, frames: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Encode full frames: applies CLIP geometry (resize shortest + crop)."""
        prepped = [resize_shortest_center_crop(f) for f in frames]
        return self.encode(prepped)

    def encode_rois(self, frames: List[np.ndarray],
                    boxes: List[Sequence[float]]) -> np.ndarray:
        """Encode ROI crops.  ``boxes`` are pixel (x, y, w, h) in ``frames``."""
        prepped = [crop_box_with_margin(f, b) for f, b in zip(frames, boxes)]
        f, _ = self.encode(prepped)
        return f

    @torch.no_grad()
    def warmup(self, n: int = 2) -> None:
        dummy = np.zeros((n, INPUT_SIZE, INPUT_SIZE, 3), np.uint8)
        self.encode(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()


def gpu_free_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _total = torch.cuda.mem_get_info()
    return free / 1e9
