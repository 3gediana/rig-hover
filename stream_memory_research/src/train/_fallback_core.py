"""Tiny internal core used ONLY for pipeline smoke tests.

`factory.build_core` routes the model name ``"dummy"`` here (and only that
name).  It exists so the training / evaluation pipeline can be built and
verified end-to-end before ``src.models.build_core`` (model-zoo) lands.

It is a minimal causal GRU with the *same contract and the same aux tensors* the
real DSM cores must expose, so the aux objectives, robustness suite, probes and
streaming path are all exercised rather than stubbed.

This is NOT a baseline; it must never be reported as one.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.common.contract import TemporalCore


class DummyCore(TemporalCore):
    """Causal GRU + fake content-addressed read/write gates.

    aux keys (shapes for a [B,T] batch):
        read        [B,T,mem_dim]      memory readout
        hidden      [B,T,hidden]       slow state
        write_gate  [B,T]              write gate in (0,1)
        route       [B,T,n_keys,n_slots]  row-stochastic routing over slots
        slot_read   [B,T,n_slots,mem_dim] per-slot readout
    """

    name = "dummy"

    def __init__(self, in_dim: int, hidden: int = 64, mem_dim: int = 16,
                 n_slots: int = 4, n_keys: int = 2, use_dt_feature: bool = True,
                 **_unused):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.mem_dim = int(mem_dim)
        self.n_slots = int(n_slots)
        self.n_keys = int(n_keys)
        self.use_dt_feature = bool(use_dt_feature)
        self.out_dim = int(hidden)

        d_in = self.in_dim + (1 if self.use_dt_feature else 0)
        self.enc = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.cell = nn.GRUCell(hidden, hidden)
        self.q = nn.Linear(hidden, self.mem_dim)
        self.slot_keys = nn.Parameter(torch.randn(self.n_slots, self.mem_dim) * 0.1)
        self.slot_val = nn.Linear(self.mem_dim, self.mem_dim)
        self.gate = nn.Linear(hidden + 1, 1)
        self.route = nn.Linear(hidden, self.n_keys * self.n_slots)
        self.out = nn.Linear(hidden + self.mem_dim, hidden)

    # ------------------------------------------------------------------ #
    def _step(self, x_t, dt_t, h, read):
        B = x_t.shape[0]
        ld = torch.log1p(dt_t.clamp_min(0.0)).unsqueeze(-1)
        inp = torch.cat([x_t, ld], -1) if self.use_dt_feature else x_t
        e = self.enc(inp)
        h_new = self.cell(e, h)
        q = self.q(h_new)
        logits = q @ self.slot_keys.t() / max(self.mem_dim ** 0.5, 1e-6)
        w = torch.softmax(logits, -1)                       # [B,N]
        vals = self.slot_val(self.slot_keys)                # [N,mem]
        read_new = w @ vals                                 # [B,mem]
        g = torch.sigmoid(self.gate(torch.cat([h_new, ld], -1))).squeeze(-1)
        route = torch.softmax(self.route(h_new).view(B, self.n_keys, self.n_slots), -1)
        o = self.out(torch.cat([h_new, read_new], -1))
        slot_read = w.unsqueeze(-1) * vals.unsqueeze(0)     # [B,N,mem]
        aux = {"read": read_new, "hidden": h_new, "write_gate": g,
               "route": route, "slot_read": slot_read}
        return o, aux, h_new, read_new

    # ------------------------------------------------------------------ #
    def forward(self, x, dt, mask, tok=None, tok_valid=None,
                roi=None, roi_valid=None, roi_meta=None):
        B, T, _ = x.shape
        dev, dtp = x.device, x.dtype
        h = torch.zeros(B, self.hidden, device=dev, dtype=dtp)
        read = torch.zeros(B, self.mem_dim, device=dev, dtype=dtp)
        outs = []
        keys = ("read", "hidden", "write_gate", "route", "slot_read")
        buf = {k: [] for k in keys}
        for t in range(T):
            o, aux, h_new, read_new = self._step(x[:, t], dt[:, t], h, read)
            m = mask[:, t]
            mb = m.view(B, 1)
            h = torch.where(mb, h_new, h)
            read = torch.where(mb, read_new, read)
            o = torch.where(mb, o, torch.zeros_like(o))
            for k in keys:
                v = aux[k]
                shape = [B] + [1] * (v.dim() - 1)
                buf[k].append(v * m.view(*shape).to(v.dtype))
            outs.append(o)
        out = torch.stack(outs, 1)
        aux = {k: torch.stack(v, 1) for k, v in buf.items()}
        return out, aux

    # ------------------------------------------------------------------ #
    def init_state(self, B: int, device):
        return {"h": torch.zeros(B, self.hidden, device=device),
                "read": torch.zeros(B, self.mem_dim, device=device)}

    def step(self, x_t, dt_t, mask_t=None, state=None,
             tok_t=None, roi_t=None, roi_valid_t=None, roi_meta_t=None):
        """Single-frame streaming step: (out_t, aux_t, new_state)."""
        if state is None:
            state = self.init_state(x_t.shape[0], x_t.device)
        if mask_t is None:
            mask_t = torch.ones(x_t.shape[0], dtype=torch.bool, device=x_t.device)
        o, aux, h_new, read_new = self._step(x_t, dt_t, state["h"], state["read"])
        mb = mask_t.view(-1, 1)
        new_state = {"h": torch.where(mb, h_new, state["h"]),
                     "read": torch.where(mb, read_new, state["read"])}
        o = torch.where(mb, o, torch.zeros_like(o))
        return o, aux, new_state
