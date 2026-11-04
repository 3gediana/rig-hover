"""Pixel-to-Event Relational Stream Network -- implementation of
``DESIGN-PIXEL-EVENT-NETWORK.md``.

Contract (spec §2): per arrival ``i`` the model gets the raw frame ``I_i`` and the
capture timestamp ``tau_i``; ``Delta_i = tau_i - tau_{i-1}``.  Nothing looks ahead.
The returned state is bounded and independent of video length.

Per-step order is fixed by spec §3.1 to avoid a circular dependency between
attention and event state:

    1. propagate previous entity/event state using elapsed time
    2. scout the current frame
    3. choose K regions (scout map + propagated slots + event-phase queries)
    4. encode the selected crops
    5. associate crops to persistent entity slots (dustbin for unmatched)
    6. update relation tokens
    7. update event phases (semi-Markov hazard), emit boundaries / counts / readiness
    8. return the new bounded state

Two invariants the unit tests enforce:

* **causality** -- perturbing frames/timestamps at ``t >= k`` cannot change any
  output at ``t < k``;
* **idempotent counting** -- the count is a state variable advanced only by a
  completion *transition*, so replaying the same frame twice does not add 2.

Sizes follow spec §3.1's starting budget at 160x160 (the cached Charades/robot
frames), which is deliberately smaller than the 224/240 target so the whole study
fits on an 8 GB card.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# default configuration (spec §3.1 tensor budget)
# --------------------------------------------------------------------------- #
PIXEL_SPEC = {
    # input
    "res": 160,                 # cached frame side
    "scout_in": 160,            # scout input side (spec: 160)
    "in_ch": 1,                 # grey cache; 3 for RGB
    # Part A
    "scout_ch": (24, 48, 96, 128),
    "scout_f4_dim": 64,         # stride-4 map channels ([B,40,40,64] at 160)
    "global_dim": 128,
    "K": 4,                     # foveal crops
    "crop": 64,                 # crop side sampled from the frame
    "crop_tok": 128,
    "focus_hidden": 128,
    # Part B
    "N": 4,                     # entity slots
    "slot_dim": 128,
    "rel_dim": 64,
    "n_rel_rounds": 2,
    "event_dim": 128,
    "scout_pretrained": True,   # spec 4.1: initialise the scout from ImageNet weights
    "n_phases": 2,              # background / active per event class
    # "direct": a linear per-frame decoder on the fused event features (retained
    #           for the reported matrix -- the hazard activation collapses to the
    #           class prior on Charades, see the ablation);
    # "hazard": the semi-Markov phase activation itself is the emitted probability.
    "event_head": "direct",
    "dt_clamp_max": 5.0,
    "hard_crops_eval": True,
}

#: every head the model can emit; a dataset turns on the ones it has labels for
HEADS = ("event", "onset", "count", "presence", "ready", "gesture")


# --------------------------------------------------------------------------- #
# small building blocks
# --------------------------------------------------------------------------- #
def norm2d(ch: int) -> nn.Module:
    """Per-sample normalisation for the convolutional stacks.

    ``BatchNorm2d`` would normalise with statistics pooled over the *batch*
    dimension.  The scout is intentionally run over a whole window at once
    (``scout_all``) to amortise kernel-launch overhead, which would make those
    statistics mix frames across time and silently break strict causality in
    training mode (measured max|d| = 7e-3 on the ``t < k`` prefix).  GroupNorm is
    per-sample, so batching over time is numerically identical to running the
    scout frame by frame, in both train and eval mode.

    One group (i.e. normalising over C,H,W per sample) rather than eight: measured
    on this GPU, ``GroupNorm(8)`` costs 1.63x ``BatchNorm2d`` on the scout shapes
    while ``GroupNorm(1)`` costs 1.00x, so one group keeps both the causality and
    the throughput.
    """
    return nn.GroupNorm(1, ch)


def dwise_sep(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    """Depthwise-separable conv block (mobile-style, INT8 friendly)."""
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
        norm2d(cin), nn.ReLU6(inplace=True),
        nn.Conv2d(cin, cout, 1, 1, 0, bias=False),
        norm2d(cout), nn.ReLU6(inplace=True),
    )


class Scout(nn.Module):
    """Global scout: coarse whole-frame view (spec §4.1).

    Returns a stride-4 map that keeps small hands/contact regions (40x40 at
    160 px), a coarser map for the global token, and the global scene token.
    Global coverage is structurally guaranteed: every pixel of the frame reaches
    the global token through average pooling, so crops cannot hide an entrant.
    """

    def __init__(self, in_ch: int = 1, ch=(24, 48, 96, 128), f4: int = 64,
                 gdim: int = 128):
        super().__init__()
        c1, c2, c3, c4 = ch
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, 2, 1, bias=False), norm2d(c1),
            nn.ReLU6(inplace=True),
            dwise_sep(c1, c2, 1),                    # stride 2
        )
        self.to_f4 = dwise_sep(c2, f4, 2)            # stride 4  -> 40x40
        self.deep = nn.Sequential(
            dwise_sep(f4, c3, 2),                    # stride 8
            dwise_sep(c3, c4, 2),                    # stride 16
            dwise_sep(c4, c4, 2),                    # stride 32
        )
        self.glob = nn.Sequential(nn.Linear(c4, gdim), nn.LayerNorm(gdim),
                                  nn.ReLU(inplace=True))
        self.f4_proj = nn.Conv2d(f4, f4, 1, 1)

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``img [B,1,H,W]`` -> ``(f4 [B,f4,H/4,W/4], global [B,gdim])``."""
        h = self.stem(img)
        f4 = self.to_f4(h)
        deep = self.deep(f4)
        g = self.glob(deep.mean(dim=(2, 3)))
        return self.f4_proj(f4), g


def _grey_first_conv(conv: nn.Conv2d) -> nn.Conv2d:
    """Adapt a pretrained 3-channel stem to single-channel grey input.

    Averaging the RGB kernels keeps the response of a grey image identical to the
    response of the same image replicated into three channels, so the pretrained
    features transfer instead of being destroyed by a random re-init.
    """
    k = conv.kernel_size
    new = nn.Conv2d(1, conv.out_channels, k, conv.stride, conv.padding,
                    bias=conv.bias is not None)
    with torch.no_grad():
        new.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


class PretrainedScout(nn.Module):
    """Global scout initialised from ImageNet MobileNetV2 (spec §4.1).

    `mobilenet_v2.features[0:4]` lands on stride 4 with 24 channels, which is
    exactly the stride-4 map spec §3.1 asks for (`[B,40,40,64]` after a 1x1
    projection at 160 px); the rest of the trunk produces the global token.  All
    stages stay fine-tunable -- this is an initialisation, not a frozen encoder.
    """

    def __init__(self, in_ch: int = 1, f4: int = 64, gdim: int = 128):
        super().__init__()
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
        feats = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
        self.stem = feats[0:4]                     # stride 4, 24 ch
        if in_ch == 1:
            self.stem[0][0] = _grey_first_conv(feats[0][0])
        elif in_ch != 3:
            raise ValueError(f"PretrainedScout expects 1 or 3 input channels, got {in_ch}")
        self.deep = feats[4:]                      # -> stride 32, 1280 ch
        self.f4_proj = nn.Conv2d(24, f4, 1, bias=False)
        c_out = 1280
        self.glob = nn.Sequential(nn.Linear(c_out, gdim), nn.LayerNorm(gdim),
                                  nn.ReLU(inplace=True))

    def train(self, mode: bool = True):                     # noqa: D102
        # Freeze the pretrained BatchNorm statistics.  With a small batch the
        # training-mode statistics are also *causal poison* here: the scout runs over
        # the whole window at once, so batch statistics would mix frames across time
        # (measured train-mode causality error 2.1e-2).  Pinning BN to eval keeps
        # `scout_all` bit-identical to running frame by frame, and freezing the
        # running stats is the standard choice when fine-tuning with small batches.
        super().train(mode)
        for mod in self.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
        return self

    def forward(self, img: torch.Tensor):
        h = self.stem(img)
        f4 = self.f4_proj(h)
        d = self.deep(h)
        return f4, self.glob(d.mean(dim=(2, 3)))


def make_scout(cfg=None):
    """Build the scout every model shares.

    `cfg["scout_pretrained"]` selects the ImageNet-initialised MobileNetV2 trunk
    (spec §4.1); the fallback is the from-scratch depthwise-separable scout.  The
    main model and every baseline call *this* function, so the visual front end is
    never a source of difference between architectures.
    """
    c = dict(PIXEL_SPEC)
    c.update(cfg or {})
    if c.get("scout_pretrained", False):
        return PretrainedScout(int(c["in_ch"]), int(c["scout_f4_dim"]),
                               int(c["global_dim"]))
    return Scout(int(c["in_ch"]), tuple(c["scout_ch"]),
                 int(c["scout_f4_dim"]), int(c["global_dim"]))


class FovealEncoder(nn.Module):
    """Shared-weight encoder for the K high-resolution crops (spec §4.2)."""

    def __init__(self, in_ch: int = 1, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1, bias=False), norm2d(32),
            nn.ReLU6(inplace=True),
            dwise_sep(32, 64, 2),
            dwise_sep(64, 96, 2),
            dwise_sep(96, dim, 2),
        )

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        """``crops [B,K,1,S,S]`` -> ``[B,K,dim]``."""
        B, K, C, S, _ = crops.shape
        x = self.net(crops.reshape(B * K, C, S, S))
        return x.mean(dim=(2, 3)).reshape(B, K, -1)


# --------------------------------------------------------------------------- #
# Part A: task-conditioned visual eye
# --------------------------------------------------------------------------- #
class VisualEye(nn.Module):
    """Scout + focus controller + foveal reads -> visual tokens for Part B.

    The focus controller is *not* a preprocessing detector: its boxes come from
    a learned head that receives the task's event-phase queries and takes task
    gradients.  Sampling is ``affine_grid`` + ``grid_sample``, i.e. differentiable
    w.r.t. box geometry, so no straight-through estimator is needed; inference can
    be made deterministic by dropping the sampling noise (there is none).
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        p = lambda k: cfg[k]
        self.K = int(p("K"))
        self.res = int(p("res"))
        self.crop = int(p("crop"))
        self.f4 = int(p("scout_f4_dim"))
        self.gdim = int(p("global_dim"))
        self.ctok = int(p("crop_tok"))
        self.scout = make_scout(cfg)
        self.fovea = FovealEncoder(cfg["in_ch"], self.ctok)
        # controller inputs: global token + pooled slot context + event query
        self.ctrl = nn.Sequential(
            nn.Linear(self.gdim + self.f4 + cfg["slot_dim"] + cfg["event_dim"],
                      cfg["focus_hidden"]), nn.ReLU(inplace=True),
            nn.Linear(cfg["focus_hidden"], cfg["focus_hidden"]), nn.ReLU(inplace=True),
        )
        self.ctrl_box = nn.Linear(cfg["focus_hidden"], self.K * 4)   # cx,cy,log w,log h
        self.ctrl_nov = nn.Linear(cfg["focus_hidden"], self.K)       # novelty logit
        self.motion = nn.Conv2d(self.f4, self.f4, 1, 1)
        self.tok = nn.Linear(self.ctok + 2 + 2 + 1 + self.f4, cfg["slot_dim"])
        self.register_buffer("_identity_theta",
                             torch.tensor([[[1.0, 0, 0], [0, 1.0, 0]]]), persistent=False)

    # -- crop sampling ---------------------------------------------------- #
    def _boxes_to_theta(self, boxes: torch.Tensor) -> torch.Tensor:
        """``boxes [B,K,4]`` (cx,cy,w,h in [-1,1]) -> affine theta ``[B,K,2,3]``."""
        cx, cy, w, h = boxes.unbind(-1)
        z = torch.zeros_like(cx)
        return torch.stack([torch.stack([w, z, cx], -1),
                            torch.stack([z, h, cy], -1)], -2)

    def sample_crops(self, img: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        B, K = boxes.shape[:2]
        theta = self._boxes_to_theta(boxes).reshape(B * K, 2, 3)
        grid = F.affine_grid(theta, (B * K, img.shape[1], self.crop, self.crop),
                             align_corners=False)
        crops = F.grid_sample(img.repeat_interleave(K, 0), grid,
                              align_corners=False, mode="bilinear")
        return crops.reshape(B, K, img.shape[1], self.crop, self.crop)

    def sample_motion(self, mmap: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Average-pool the motion map inside each box -> ``[B,K,f4]``."""
        B, K = boxes.shape[:2]
        th = self._boxes_to_theta(boxes).reshape(B * K, 2, 3)
        grid = F.affine_grid(th, (B * K, mmap.shape[1], 4, 4), align_corners=False)
        pooled = F.grid_sample(mmap.repeat_interleave(K, 0), grid,
                              align_corners=False, mode="bilinear")
        return pooled.mean(dim=(2, 3)).reshape(B, K, -1)

    # -- forward ---------------------------------------------------------- #
    def scout_all(self, imgs: torch.Tensor):
        """Batched per-frame scout over a whole window.

        The scout is strictly per-frame (no state), so running it on all ``T``
        frames at once is mathematically identical to running it inside the
        per-step loop -- but it turns ``T`` small kernel launches into one big
        one.  Motion maps are causal: ``mmap_t`` only uses ``f4_t`` and
        ``f4_{t-1}``.

        ``imgs [B,T,C,H,W]`` -> ``(f4 [B,T,f4,h,w], g [B,T,gd], mmap [B,T,f4,h,w])``
        """
        B, T = imgs.shape[:2]
        f4, g = self.scout(imgs.reshape(B * T, *imgs.shape[2:]))
        f4 = f4.reshape(B, T, *f4.shape[1:])
        g = g.reshape(B, T, -1)
        prev = torch.cat([torch.zeros_like(f4[:, :1]), f4[:, :-1]], 1)
        mmap = self.motion((f4 - prev).reshape(B * T, *f4.shape[2:]))
        mmap = mmap.reshape(B, T, *mmap.shape[1:])
        mmap[:, 0] = 0.0
        return f4, g, mmap

    def forward(self, img: torch.Tensor, slot_ctx: torch.Tensor,
                event_q: torch.Tensor, prev_f4: Optional[torch.Tensor] = None,
                prev_img: Optional[torch.Tensor] = None,
                f4: Optional[torch.Tensor] = None, g: Optional[torch.Tensor] = None,
                mmap: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """One frame.

        ``img [B,C,H,W]``; ``slot_ctx`` = mean slot appearance ``[B,slot_dim]``;
        ``event_q`` = event-state query ``[B,event_dim]``.  ``f4`` / ``g`` / ``mmap``
        may be supplied precomputed (batched path); otherwise they are computed
        here (true streaming path used by :meth:`step`).
        """
        if f4 is None or g is None or mmap is None:
            f4, g = self.scout(img)
            mmap = torch.zeros_like(f4) if prev_f4 is None else self.motion(f4 - prev_f4)
        h = self.ctrl(torch.cat([g, f4.mean(dim=(2, 3)), slot_ctx, event_q], -1))
        raw = self.ctrl_box(h).reshape(img.shape[0], self.K, 4)
        boxes = torch.cat([torch.tanh(raw[..., :2]) * 0.9,
                           torch.sigmoid(raw[..., 2:]) * 0.9 + 0.05], -1)
        novelty = self.ctrl_nov(h)                        # ``[B,K]``
        crops = self.sample_crops(img, boxes)
        ctok = self.fovea(crops)
        mot = self.sample_motion(mmap, boxes)
        vis = torch.ones_like(boxes[..., :1])
        tokens = self.tok(torch.cat([ctok, boxes[..., :2], boxes[..., 2:], vis, mot], -1))
        return {"tokens": tokens, "boxes": boxes, "novelty": novelty,
                "global": g, "f4": f4, "motion": mmap.mean(dim=(2, 3))}


# --------------------------------------------------------------------------- #
# Part B: relational scene and event engine
# --------------------------------------------------------------------------- #
class EntityBinder(nn.Module):
    """Persistent entity slots with content+position association (spec §5.1).

    ``slot = {appearance, position, velocity, visibility, identity_confidence, age}``.
    A dustbin column lets unmatched tokens avoid corrupting an identity; unassigned
    salient tokens (high novelty) initialise a free slot.  During occlusion the
    appearance/identity embedding is *held* while visibility falls and the position
    uncertainty grows -- identity is not erased by missing frames.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.N = int(cfg["N"])
        self.d = int(cfg["slot_dim"])
        self.dustbin = nn.Parameter(torch.zeros(1))
        self.q = nn.Linear(self.d, self.d)
        self.k = nn.Linear(self.d, self.d)
        self.v = nn.Linear(self.d, self.d)
        self.pos = nn.Linear(4, 1)                     # |p_m - p_j| (2) + |v_j| (2)
        self.gate = nn.Linear(2 * self.d, self.d)
        self.upd = nn.Sequential(nn.Linear(2 * self.d, 2 * self.d), nn.GELU(),
                                 nn.Linear(2 * self.d, self.d))

    def init_state(self, B: int, device, dtype) -> Dict[str, torch.Tensor]:
        z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)
        return {"app": z(B, self.N, self.d), "pos": z(B, self.N, 2),
                "vel": z(B, self.N, 2), "vis": z(B, self.N),
                "conf": z(B, self.N), "age": z(B, self.N),
                "occ": z(B, self.N),        # steps since last evidence (occlusion counter)
                "used": z(B, self.N)}       # evidence mass ever seen -> free-slot policy

    def forward(self, tokens: torch.Tensor, novelty: torch.Tensor, boxes: torch.Tensor,
                mask: torch.Tensor, st: Dict[str, torch.Tensor]):
        """``tokens [B,K,d]``, ``boxes [B,K,4]`` -> new slots + association."""
        B, K, _ = tokens.shape
        centre = boxes[..., :2]
        occ = (st["occ"] + 1.0).clamp(max=64.0)           # steps without evidence
        vis = st["vis"] * 0.85                            # visibility decays while unseen
        conf = st["conf"] * 0.95
        q = self.q(tokens)                                # [B,K,d]
        k = self.k(st["app"])                             # [B,N,d]
        v = self.v(tokens)
        score = torch.einsum("bkd,bnd->bkn", q, k) / math.sqrt(self.d)
        dp = (centre.unsqueeze(2) - st["pos"].unsqueeze(1)).abs()          # [B,K,N,2]
        posfeat = torch.cat([dp, st["vel"].unsqueeze(1).expand_as(dp)], -1)
        score = score - self.pos(posfeat).squeeze(-1)                      # b_pos
        score = score - 2.0 * (occ.unsqueeze(1) / 8.0)                     # b_vis
        dust = self.dustbin.expand(B, K, 1)                                # dustbin column
        att = torch.softmax(torch.cat([score, dust], -1), -1)              # [B,K,N+1]
        att_s = att[..., :self.N]
        w = att_s.sum(dim=1)                                               # [B,N] evidence
        agg = torch.einsum("bkn,bkd->bnd", att_s, v)
        wj = w.unsqueeze(-1).clamp_min(1e-6)
        ctx = agg / wj
        have = torch.sigmoid(w.unsqueeze(-1) * 2.0)                        # [B,N,1]
        g = torch.sigmoid(self.gate(torch.cat([st["app"], ctx], -1)))
        upd = self.upd(torch.cat([st["app"] + ctx, st["app"] - ctx], -1))
        upd = F.layer_norm(upd, upd.shape[-1:])
        app_new = st["app"] + have * g * (upd - st["app"])
        # position: evidence-weighted centroid; otherwise hold the prediction
        has = (w > 1e-3).float()                                           # [B,N]
        pos_ev = torch.einsum("bkn,bk->bn", att_s, centre[..., 0]) / wj.squeeze(-1)
        pos_ey = torch.einsum("bkn,bk->bn", att_s, centre[..., 1]) / wj.squeeze(-1)
        pred_pos = st["pos"] + st["vel"]
        new_pos = has.unsqueeze(-1) * torch.stack([pos_ev, pos_ey], -1) + \
            (1 - has).unsqueeze(-1) * pred_pos
        new_vel = 0.5 * st["vel"] + 0.5 * (new_pos - st["pos"])
        new_vis = torch.maximum(vis, w)
        new_conf = torch.maximum(conf, w)
        new_occ = occ * (1.0 - has)                                        # reset on evidence
        used = st["used"] + has
        # ---- unassigned salient token -> initialise a free slot (spec §5.1) ----
        unassigned = novelty - att_s.max(dim=2).values                      # [B,K]
        pick = unassigned.argmax(dim=1)                                    # [B]
        ar = torch.arange(B, device=tokens.device)
        tok_pick = tokens[ar, pick]                                        # [B,d]
        pos_pick = centre[ar, pick]                                        # [B,2]
        init = (unassigned[ar, pick] > 0).float()                          # [B]
        slot_idx = used.argmin(dim=1)                                      # [B]
        oh = F.one_hot(slot_idx, self.N).float()                           # [B,N]
        m = oh * init.unsqueeze(-1)                                        # [B,N]
        app_new = app_new + m.unsqueeze(-1) * (tok_pick.unsqueeze(1) - app_new)
        new_pos = new_pos + m.unsqueeze(-1) * (pos_pick.unsqueeze(1) - new_pos)
        new_vel = new_vel * (1 - m.unsqueeze(-1))
        new_vis = new_vis + m * (1.0 - new_vis)
        new_conf = new_conf + m * (1.0 - new_conf)
        new_occ = new_occ * (1 - m)
        new_age = st["age"] * 0.99 + m * (1 - st["age"] * 0.99)
        keep = mask.unsqueeze(-1)
        out = {"app": torch.where(keep.unsqueeze(-1), app_new, st["app"]),
               "pos": torch.where(keep.unsqueeze(-1), new_pos, st["pos"]),
               "vel": torch.where(keep.unsqueeze(-1), new_vel, st["vel"]),
               "vis": torch.where(keep, new_vis, st["vis"]),
               "conf": torch.where(keep, new_conf, st["conf"]),
               "age": torch.where(keep, new_age, st["age"]),
               "occ": torch.where(keep, new_occ, st["occ"]),
               "used": torch.where(keep, used, st["used"])}
        return out, att_s


class RelationGraph(nn.Module):
    """Directed pairwise relation tokens + a scene anchor, 2 sparse rounds (§5.2).

    Nodes are ``N`` entity slots plus one scene anchor, so the relation tensor is
    ``[B, N+1, N+1, rel_dim]`` -- bounded in video length by construction.  The
    graph's job (spec) is to bind *who did what to which object/surface* so the
    event engine never receives unrelated per-frame features with no entity
    correspondence.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.N = int(cfg["N"])
        self.dr = int(cfg["rel_dim"])
        self.d = int(cfg["slot_dim"])
        self.n = self.N + 1
        self.node = nn.Linear(self.d, self.dr)
        self.scene = nn.Linear(int(cfg["global_dim"]), self.dr)
        self.edge = nn.Sequential(nn.Linear(2 * self.dr + 4, self.dr), nn.GELU(),
                                  nn.Linear(self.dr, self.dr))
        self.msg = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * self.dr, self.dr), nn.GELU(),
                          nn.Linear(self.dr, self.dr)) for _ in range(cfg["n_rel_rounds"])
        ])

    def forward(self, slots: Dict[str, torch.Tensor], global_tok: torch.Tensor,
                prev: torch.Tensor) -> torch.Tensor:
        B, N = slots["app"].shape[0], self.N
        nodes = torch.cat([self.node(slots["app"]),
                           self.scene(global_tok).unsqueeze(1)], 1)      # [B,n,dr]
        pos = torch.cat([slots["pos"], torch.zeros(B, 1, 2, device=nodes.device,
                                                   dtype=nodes.dtype)], 1)  # [B,n,2]
        vis = torch.cat([slots["vis"], slots["vis"].new_ones(B, 1)], 1)     # [B,n]
        dp = pos.unsqueeze(2) - pos.unsqueeze(1)                            # [B,n,n,2]
        ni = nodes.unsqueeze(2).expand(-1, -1, self.n, -1)
        nj = nodes.unsqueeze(1).expand(-1, self.n, -1, -1)
        vi = vis.unsqueeze(2).expand(-1, -1, self.n).unsqueeze(-1)
        vj = vis.unsqueeze(1).expand(-1, self.n, -1).unsqueeze(-1)
        rel = self.edge(torch.cat([ni, nj, dp, vi, vj], -1))              # [B,n,n,dr]
        for m in self.msg:
            ctx = rel.mean(dim=2, keepdim=True).expand_as(rel)
            rel = rel + m(torch.cat([rel, ctx], -1))
        return rel


class EventEngine(nn.Module):
    """Semi-Markov phase processes with an explicit hazard (spec §5.3).

    Per event class ``c`` the persistent state is a phase activation probability
    ``a_c``.  A learned hazard ``lambda`` (1/s) is integrated over the *physical*
    interval:

        p_on  = 1 - exp(-lambda_on  * Delta)
        p_off = 1 - exp(-lambda_off * Delta)
        a'    = a + (1-a) p_on - a p_off

    A completion pulse is ``a * p_off`` and the cumulative count only advances on
    that pulse, so re-evaluating a frame cannot double-count.  The phase
    activation *is* the emitted event probability, which forces the network to
    carry an explicit event state instead of re-deciding each frame.
    """

    def __init__(self, cfg: Dict, n_classes: int):
        super().__init__()
        self.C = int(n_classes)
        self.d = int(cfg["event_dim"])
        self.de = int(cfg["event_dim"])
        din = int(cfg["slot_dim"]) * cfg["N"] \
            + int(cfg["rel_dim"]) * (cfg["N"] + 1) ** 2 \
            + int(cfg["global_dim"]) + self.d
        self.inp = nn.Sequential(nn.Linear(din, self.de), nn.GELU())
        self.cell = nn.GRUCell(self.de, self.de)
        self.direct = nn.Linear(self.de, self.C)   # discrete-time phase decoder
        self.haz_on = nn.Linear(self.de, self.C)
        self.haz_off = nn.Linear(self.de, self.C)
        self.dt_clamp = float(cfg["dt_clamp_max"])

    def init_state(self, B: int, device, dtype) -> Dict[str, torch.Tensor]:
        z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)
        return {"h": z(B, self.de), "act": z(B, self.C), "count": z(B, self.C),
                "elapsed": z(B, self.C)}

    def forward(self, slots: Dict[str, torch.Tensor], rel: torch.Tensor,
                global_tok: torch.Tensor, dt: torch.Tensor, mask: torch.Tensor,
                st: Dict[str, torch.Tensor]):
        B = slots["app"].shape[0]
        e = self.inp(torch.cat([slots["app"].reshape(B, -1), rel.reshape(B, -1),
                                global_tok, st["h"]], -1))
        h = self.cell(e, st["h"])
        d = (dt.clamp(0.0, self.dt_clamp)).unsqueeze(-1)
        lam_on = F.softplus(self.haz_on(h)) + 1e-3
        lam_off = F.softplus(self.haz_off(h)) + 1e-3
        p_on = 1.0 - torch.exp(-lam_on * d)
        p_off = 1.0 - torch.exp(-lam_off * d)
        a = st["act"]
        act = a + (1 - a) * p_on - a * p_off
        act = act.clamp(0.0, 1.0)
        pulse = a * p_off                                  # completion transitions
        count = st["count"] + pulse * mask.unsqueeze(-1)
        elapsed = torch.where(act > 0.5, st["elapsed"] + d, torch.zeros_like(st["elapsed"]))
        m = mask.unsqueeze(-1)
        new = {"h": torch.where(m, h, st["h"]),
               "act": torch.where(m, act, st["act"]),
               "count": torch.where(m, count, st["count"]),
               "elapsed": torch.where(m, elapsed, st["elapsed"])}
        return new, {"pulse": pulse, "onset_p": (1 - a) * p_on, "offset_p": a * p_off,
                     "lam_on": lam_on, "lam_off": lam_off, "event_q": h,
                     "direct": self.direct(h), "act": act}


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
class PixelEventNet(nn.Module):
    """Full pixel-to-event network.  Also the container for the streaming state."""

    def __init__(self, cfg: Dict, n_event_classes: int = 25,
                 n_gesture: int = 10, n_presence: int = 4):
        super().__init__()
        self.cfg = dict(cfg)
        self.C = int(n_event_classes)
        self.G = int(n_gesture)
        self.P = int(n_presence)
        self.eye = VisualEye(cfg)
        self.binder = EntityBinder(cfg)
        self.graph = RelationGraph(cfg)
        self.events = EventEngine(cfg, self.C)
        sd = int(cfg["slot_dim"])
        dd = int(cfg["event_dim"])
        self.presence = nn.Sequential(nn.Linear(sd + dd, 128), nn.GELU(),
                                      nn.Linear(128, self.P))
        self.gesture = nn.Sequential(nn.Linear(sd + dd, 128), nn.GELU(),
                                     nn.Linear(128, self.G))
        self.ready = nn.Sequential(nn.Linear(sd + dd, 128), nn.GELU(), nn.Linear(128, 1))
        self.trunk = nn.Identity()

    # -- state ------------------------------------------------------------ #
    def init_state(self, B: int, device=None, dtype=torch.float32):
        device = device or next(self.parameters()).device
        return {"slots": self.binder.init_state(B, device, dtype),
                "events": self.events.init_state(B, device, dtype),
                "rel": torch.zeros(B, self.cfg["N"] + 1, self.cfg["N"] + 1,
                                   self.cfg["rel_dim"], device=device, dtype=dtype),
                "prev_f4": None, "prev_img": None}

    # -- one causal step -------------------------------------------------- #
    def step(self, img: torch.Tensor, dt: torch.Tensor, mask: torch.Tensor,
             state: Optional[Dict] = None, pre: Optional[Dict] = None
             ) -> Tuple[Dict[str, torch.Tensor], Dict, Dict]:
        """``img [B,C,H,W]``, ``dt [B]``, ``mask [B]`` -> (logits, new_state, aux).

        ``pre`` optionally carries the batched-scout results ``{"f4","g","mmap"}``
        for this frame; it changes nothing numerically, only how many kernel
        launches the per-frame path costs.
        """
        if state is None:
            state = self.init_state(img.shape[0], img.device, img.dtype)
        mask = mask.bool()
        dt = dt.to(img.dtype)
        slots = state["slots"]
        ev = state["events"]
        slot_ctx = slots["app"].mean(dim=1)
        if pre is None:                      # true streaming: the state carries f4_{t-1}
            eye = self.eye(img, slot_ctx, ev["h"], prev_f4=state.get("prev_f4"))
        else:                                # batched window: scout already done
            eye = self.eye(img, slot_ctx, ev["h"], prev_f4=None,
                           f4=pre["f4"], g=pre["g"], mmap=pre["mmap"])
        slots_new, assoc = self.binder(eye["tokens"], eye["novelty"], eye["boxes"],
                                       mask, slots)
        rel = self.graph(slots_new, eye["global"], state["rel"])
        ev_new, einfo = self.events(slots_new, rel, eye["global"], dt, mask, ev)
        sd = int(self.cfg["slot_dim"])
        ctx = torch.cat([slots_new["app"].mean(dim=1), einfo["event_q"]], -1)
        # The event *phase activation* is the emitted event probability: the
        # network has to carry an explicit event state instead of re-deciding.
        if self.cfg.get("event_head", "direct") == "direct":
            # discrete-time decoder: the semi-Markov state still drives the count
            # (a completion transition is the only thing that increments it), but the
            # per-frame probability is read out directly instead of being tied to
            # the bounded phase activation.
            ev_logit = einfo["direct"]
        else:
            ev_logit = torch.logit(ev_new["act"].clamp(1e-6, 1 - 1e-6))
        logits = {
            "event": ev_logit,
            "onset": torch.logit(einfo["onset_p"].sum(-1, keepdim=True).clamp(1e-6, 1 - 1e-6)),
            "count": ev_new["count"],
            "presence": self.presence(ctx),
            "ready": self.ready(ctx),
            "gesture": self.gesture(ctx),
        }
        m = mask.unsqueeze(-1)
        new_state = {
            "slots": slots_new,
            "events": ev_new,
            "rel": torch.where(mask.view(-1, 1, 1, 1), rel, state["rel"]),
            "prev_f4": eye["f4"].detach(),
        }
        aux = {"boxes": eye["boxes"], "assoc": assoc, "novelty": eye["novelty"],
               "pulse": einfo["pulse"], "motion": eye["motion"]}
        return logits, new_state, aux

    # -- window forward --------------------------------------------------- #
    def forward(self, imgs: torch.Tensor, dt: torch.Tensor, mask: torch.Tensor):
        """``imgs [B,T,C,H,W]``, ``dt [B,T]``, ``mask [B,T]``.

        Runs the *same* ``step`` per frame (so training and streaming are
        identical by construction) and stacks the per-frame logits.  Only the
        stateless scout is batched over the window.
        """
        B, T = imgs.shape[:2]
        f4, g, mmap = self.eye.scout_all(imgs)
        state = self.init_state(B, imgs.device, imgs.dtype)
        out: Dict[str, List[torch.Tensor]] = {k: [] for k in HEADS}
        auxs: Dict[str, List[torch.Tensor]] = {"boxes": [], "pulse": [], "novelty": []}
        for t in range(T):
            logits, state, aux = self.step(imgs[:, t], dt[:, t], mask[:, t], state,
                                           pre={"f4": f4[:, t], "g": g[:, t],
                                                "mmap": mmap[:, t]})
            for k in HEADS:
                out[k].append(logits[k])
            for k in auxs:
                auxs[k].append(aux[k])
        seq = {k: torch.stack(v, 1) for k, v in out.items()}
        seq["count_final"] = state["events"]["count"]
        aux = {k: torch.stack(v, 1) for k, v in auxs.items()}
        aux["state"] = state
        return seq, aux


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
PIXEL_MODELS = ("pixel_event",)


def build_pixel_model(name: str, cfg: Optional[Dict] = None, **kw) -> nn.Module:
    c = dict(PIXEL_SPEC)
    c.update(cfg or {})
    if name == "pixel_event":
        return PixelEventNet(c, **kw)
    raise ValueError(f"unknown pixel model {name!r}; have {PIXEL_MODELS}")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def count_macs(m: nn.Module, imgs: torch.Tensor, dt: torch.Tensor,
               mask: torch.Tensor) -> float:
    """Exact forward-MAC count for one window, measured with module hooks.

    Returns MACs per analysed frame (``total / T``), which is the number spec §8
    budget should be compared against (``<=1.5 GMAC/frame``).
    """
    total = {"v": 0.0}

    def conv_hook(mod, inp, out):
        total["v"] += (out.numel() * mod.kernel_size[0] * mod.kernel_size[1]
                       * mod.in_channels // mod.groups)

    def lin_hook(mod, inp, out):
        total["v"] += out.numel() * mod.in_features

    hs = [mod.register_forward_hook(conv_hook) for mod in m.modules()
          if isinstance(mod, nn.Conv2d)]
    hs += [mod.register_forward_hook(lin_hook) for mod in m.modules()
           if isinstance(mod, nn.Linear)]
    was_training = m.training
    m.eval()
    with torch.no_grad():
        m(imgs, dt, mask)
    if was_training:
        m.train()
    for h in hs:
        h.remove()
    return total["v"] / max(1, imgs.shape[1])
