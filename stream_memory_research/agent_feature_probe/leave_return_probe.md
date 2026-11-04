# One-clip robot feature probe

**Exploratory only:** one 0.5 m, front-facing, night session; same recording is split by time. Labels are coarse manual visibility states from contact sheets, not frame annotations collected independently.

Frames: 192; train 0-95; test 96-191. Classes: 0=absent, 1=visible, 2=transition/partial.

| Features | Accuracy | Macro-F1 | Test support (0/1/2) |
|---|---:|---:|---|
| raw_16x16_current | 0.781 | 0.703 | 21/53/22 |
| cls_current | 0.635 | 0.670 | 21/53/22 |
| patch_current | 0.812 | 0.807 | 21/53/22 |
| cls_plus_delta | 0.677 | 0.702 | 21/53/22 |
| cls_causal_mean5 | 0.604 | 0.618 | 21/53/22 |
| cls_mean5_plus_delta | 0.604 | 0.622 | 21/53/22 |

This probe only checks whether the selected feature sequence contains a signal for coarse presence in this same clip. It cannot establish leave/return generalization, identity continuity across people/scenes, interaction readiness, fist recognition, knock phases/count, or target location. The test shares recording/background/person with training; it is not a valid deployment estimate.

No knock/wave count scores are produced because the recordings lack reliable event-level boundaries/count annotations.
