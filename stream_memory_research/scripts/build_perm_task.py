"""Build `synth_recall_multi_perm`: multi-object recall with UNORDERED object tokens.

Why this task exists
--------------------
Every other multi-object task we have gives the model tokens in a *stable canonical
order* (object 0 is always row 0, ...).  That is not what a real detector produces:
detection rows have no stable identity across frames, so any model that reads
position `k` as "object k" is exploiting an artefact.

Here the 16 token rows are randomly permuted **independently at every timestep**,
while the labels stay in canonical slot order.  Consequences:

* A model that consumes the flattened token vector sees a different permutation of
  the same multiset every frame -> its input statistics are corrupted.
* A model that consumes tokens as a *set* and addresses memory by CONTENT
  (`use_tok=True` in dsm/dsm_slot) is permutation-invariant and should be unaffected.
* The last-information-bearing step for each object is untouched, so a perfect
  content-addressed memory oracle still scores 1.0 (verified against the base task's
  hand-written oracle by construction: we only permute the observation order).

This is the honest version of "multi-target, low cross-talk, content-addressed
memory": the observation really is an unordered detection set.

Output: data_manifests/synth_recall_multi_perm.json
        G:/stream_memory_data/synth/synth_recall_multi_perm/<seq>.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\robot_work\stream_memory_research")
sys.path.insert(0, str(ROOT))

from src.common import contract  # noqa: E402
from src.common.contract import Manifest, SeqRecord  # noqa: E402

SRC_NAME = "synth_recall_multi"
DST_NAME = "synth_recall_multi_perm"
OUT = contract.SYNTH / DST_NAME


def main():
    man = Manifest.load(contract.MANIFESTS / f"{SRC_NAME}.json")
    OUT.mkdir(parents=True, exist_ok=True)
    seqs = []
    rng_master = np.random.default_rng(20261104)
    n_perm = 0
    for r in man.seqs:
        with np.load(r.path, allow_pickle=False) as z:
            arrays = {k: np.asarray(z[k]) for k in z.files}
        tok = arrays["tok"]                      # [T, K, Dt]
        T, K, Dt = tok.shape
        rng = np.random.default_rng(abs(hash(r.seq_id)) % (2 ** 63))
        out_tok = np.empty_like(tok)
        for t in range(T):
            perm = rng.permutation(K)
            out_tok[t] = tok[t][perm]
            n_perm += 1
        arrays["tok"] = out_tok
        dst = OUT / Path(r.path).name
        np.savez_compressed(dst, **arrays)
        seqs.append(SeqRecord(seq_id=r.seq_id + "_perm", source_id=r.source_id,
                              split=r.split, T=r.T, path=str(dst),
                              labels=list(r.labels), meta=dict(r.meta)))
    extra = dict(man.extra)
    extra.update({
        "note": ("multi-object recall with UNORDERED object tokens: the 16 token rows "
                 "are randomly permuted independently at every timestep. Labels stay in "
                 "canonical slot order, so the task requires CONTENT-ADDRESSED memory; "
                 "a model that reads token position as identity cannot solve it."),
        "derived_from": SRC_NAME,
        "construction": "tok[t] <- tok[t][rng.permutation(K)] for every t; all else unchanged",
        "base_dataset": SRC_NAME,
        "main_metric": man.extra.get("main_metric", "cos"),
        "eval_only": False,
    })
    new = Manifest(dataset=DST_NAME, task=man.task, feature_dim=man.feature_dim,
                   tok_dim=man.tok_dim, roi_dim=man.roi_dim, n_roi=man.n_roi,
                   encoder=man.encoder, label_specs=dict(man.label_specs),
                   extra=extra, seqs=seqs)
    p = new.save(contract.MANIFESTS / f"{DST_NAME}.json")
    new.check_split_disjoint()
    print("wrote", p, new.counts(), "permuted steps:", n_perm)
    z = np.load(seqs[0].path)
    zs = np.load(man.seqs[0].path)
    print("same label?", np.allclose(z["lab_state"], zs["lab_state"]))
    print("tok row-multiset identical at t=0?",
          np.allclose(np.sort(z["tok"][0].sum(-1)), np.sort(zs["tok"][0].sum(-1))))
    print("tok order differs at t=0?", not np.allclose(z["tok"][0], zs["tok"][0]))


if __name__ == "__main__":
    main()
