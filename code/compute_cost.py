"""Approximate training cost of auxiliary pre-training relative to the fine-tuning epochs it saves.

Cost is expressed in image-epochs at 640 x 640 px; one training image at 1,280 x 1,280 px counts as
four units because it contains four times as many pixels. Validation passes are ignored.

Inputs : T_convergence.csv written by analysis_revision.py
Outputs: T_compute_cost.csv in the same folder
Usage  : python compute_cost.py TABLE_DIR
"""
import sys
from pathlib import Path

import pandas as pd

TAB = Path(sys.argv[1])
VAL_FRACTION = 0.15          # share of each auxiliary subset held out for validation
AUX_EPOCHS = 30              # auxiliary pre-training epochs at 640 px
PHONEBEE_TRAIN_FRAMES = 222  # two folds of 111 frames are used for training in each fold
UNITS_PER_1280_IMAGE = 4     # (1280 / 640) ** 2

conv = pd.read_csv(TAB / "T_convergence.csv")
rows = []
for run, d in conv.groupby("run"):
    control = float(d.loc[d.config == "coco", "epochs_to_080_mean_curve"].iloc[0])
    for _, r in d[d.config != "coco"].iterrows():
        n = int("".join(ch for ch in r.config if ch.isdigit()))
        pretrain = round(n * (1 - VAL_FRACTION)) * AUX_EPOCHS
        saved_epochs = control - float(r.epochs_to_080_mean_curve)
        saved_units = saved_epochs * PHONEBEE_TRAIN_FRAMES * UNITS_PER_1280_IMAGE
        rows.append(dict(run=run, config=r.config, pretrain_units=pretrain,
                         control_epochs_to_080=control, epochs_to_080=float(r.epochs_to_080_mean_curve),
                         epochs_saved=saved_epochs, saved_units=saved_units,
                         target_datasets_to_break_even=(pretrain / saved_units) if saved_units > 0 else float("nan")))
pd.DataFrame(rows).to_csv(TAB / "T_compute_cost.csv", index=False)
print(pd.DataFrame(rows).round(2).to_string(index=False))
