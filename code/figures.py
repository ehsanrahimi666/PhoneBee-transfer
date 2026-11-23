"""Figures for the PhoneBee transfer study.

Inputs : TAB   tables written by analysis_revision.py
         ORIG  outputs of the original Kaggle runs
         REV   folder 'revision' produced by revision_experiments_kaggle.py
Outputs: PNG (300 dpi) and TIFF (600 dpi) figures in FIG
Usage  : python figures_revision.py TAB ORIG REV FIG
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

TAB, ORIG, REV, FIG = (Path(a) for a in sys.argv[1:5])
FIG.mkdir(parents=True, exist_ok=True)

C_GBIF, C_CT, C_PB, C_GT, C_CTRL = "#D9772B", "#2A7F7F", "#3F88C5", "#8C8C8C", "#4D4D4D"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.7, "xtick.major.width": 0.7,
                     "ytick.major.width": 0.7, "legend.frameon": False, "legend.fontsize": 7.5,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
W = 6.3
CONFIGS = ["coco", "gbif500", "gbif2000", "gbif5000", "ct500", "ct2000", "ct5000"]
NS = [500, 2000, 5000]


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(FIG / f"{name}.tif", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def title(ax, letter, text):
    ax.set_title(f"{letter}   {text}", loc="left", fontsize=9.5, fontweight="bold")


def tab(name):
    return pd.read_csv(TAB / f"{name}.csv")


# ------------------------------------------------------------------ Figure 2: study design
def fig_design():
    fig, ax = plt.subplots(figsize=(W, 3.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 72)
    ax.axis("off")

    def box(x, y, w, h, head, lines, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.4", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h - 1.6, head, ha="center", va="top", fontsize=7.4, fontweight="bold")
        ax.text(x + w / 2, y + h - 5.6, "\n".join(lines), ha="center", va="top", fontsize=6.5, linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#555555"))

    for y, t in ((71.5, "Image sources"), (46.5, "Training"), (22.5, "Evaluation")):
        ax.text(0.5, y, t, fontsize=8.3, fontweight="bold", color="#333333", va="top")
    box(1, 50, 30, 15, "PhoneBee smartphone video", ["333 frames, 775 annotated bees", "median bee 20 px (1.36% of frame)", "3 contiguous folds of 111 frames"], "#DDEAF6", C_PB)
    box(35, 50, 30, 15, "GBIF occurrence photographs", [r"98,731 $\it{Apis\ mellifera}$ images", "nested subsets N = 500, 2,000, 5,000", "Grounding DINO pseudo-labels"], "#FBE9DA", C_GBIF)
    box(69, 50, 30, 15, "Insect camera-trap benchmark", ["6,619 images, expert labels", "nested subsets N = 500, 2,000, 5,000", "held-out test set: 1,300 images"], "#DCEEEE", C_CT)
    box(1, 25, 40, 17, "Fine-tuning on PhoneBee folds", ["7 initialisations × 3 folds", "60 epochs at 1,280 px, final weights", "no early stopping; a second training run", "estimates run-to-run variation"], "#F2F2F2", "#777777")
    box(57, 25, 40, 17, "Auxiliary pre-training", ["COCO weights, 30 epochs at 640 px", "6 models: 2 sources × 3 sizes", "COCO-only control: no auxiliary stage"], "#F2F2F2", "#777777")
    box(1, 1, 30, 18, "O1–O2  Fine-tuned detection", ["mAP@0.5 and recall per fold", "effects of source and volume", "epochs to reach mAP@0.5 = 0.80"], "#EEF5EC", "#5B8C51")
    box(35, 1, 30, 18, "O3  Direct transfer", ["7 models × 3 target domains", "inference at 640 and 1,280 px", "confidence, size and location", "of predictions on PhoneBee"], "#EEF5EC", "#5B8C51")
    box(69, 1, 30, 18, "O4  Object-scale manipulation", ["sliced inference on PhoneBee", "magnification 1–24×", "bees brought to each model's", "training scale; AP@0.5, 95% CI"], "#EEF5EC", "#5B8C51")
    arrow(16, 49.3, 16, 42.8)
    arrow(50, 49.3, 70, 42.8)
    arrow(84, 49.3, 82, 42.8)
    arrow(56.4, 33.5, 41.6, 33.5)
    arrow(16, 24.3, 16, 19.8)
    arrow(68, 24.3, 52, 19.8)
    arrow(84, 24.3, 84, 19.8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, "Fig02_study_design")


# ------------------------------------------------------------------ Figure 3: source-domain competence
def fig_within():
    t = tab("T1_within_domain")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), constrained_layout=True)
    for ax, (src, col, test, lab) in zip(axes, (
            ("camera-trap", C_CT, "camera-trap held-out test (1300 images)", "Camera-trap test set"),
            ("GBIF", C_GBIF, "GBIF reference set disjoint from all subsets (300 images)", "GBIF reference set"))):
        d = t[(t.source == src) & (t.imgsz == 640) & (t.test_set == test)].sort_values("N")
        ax.plot(d.N, d.mAP50, "-o", color=col, lw=1.6, ms=5, label="mAP@0.5")
        ax.plot(d.N, d.mAP50_95, "--s", color=col, lw=1.2, ms=4, alpha=0.75, label="mAP@0.5:0.95")
        for x, y in zip(d.N, d.mAP50):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
        ax.set_xscale("log")
        ax.set_xticks(NS, ["500", "2,000", "5,000"])
        ax.minorticks_off()
        ax.set_ylim(0, 1.05)
        ax.set_xlim(380, 6600)
        ax.set_xlabel("Pre-training images (N)")
        ax.grid(alpha=0.25, lw=0.5)
        title(ax, "a" if src == "camera-trap" else "b", lab)
    axes[0].set_ylabel("Score at 640 px")
    axes[0].legend(loc="lower right")
    save(fig, "Fig03_source_domain")


# ------------------------------------------------------------------ Figure 4: fine-tuned performance
def fig_finetune():
    fm = tab("S1_fold_metrics")
    fm = fm[fm.protocol == "fixed 60 epochs, final weights"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.85), constrained_layout=True)
    for ax, col, lab, letter in ((axes[0], "mAP50", "mAP@0.5", "a"), (axes[1], "recall", "Recall", "b")):
        ctrl = fm[fm.config == "coco"][col]
        m, s = ctrl.mean(), ctrl.std(ddof=1)
        ax.axhspan(m - s, m + s, color="#E6E6E6", lw=0)
        ax.axhline(m, color=C_CTRL, ls="--", lw=1.1, label=f"COCO-only control ({m:.3f})")
        for src, c, mk, name in (("gbif", C_GBIF, "o", "GBIF photographs"), ("ct", C_CT, "s", "Camera-trap images")):
            mean = [fm[fm.config == f"{src}{n}"][col].mean() for n in NS]
            sd = [fm[fm.config == f"{src}{n}"][col].std(ddof=1) for n in NS]
            xs = np.array(NS) * (0.94 if src == "gbif" else 1.06)
            ax.errorbar(xs, mean, yerr=sd, fmt=f"-{mk}", color=c, ms=5, lw=1.5, capsize=2.5, label=name)
            for n, x in zip(NS, xs):
                ax.scatter([x] * 3, fm[fm.config == f"{src}{n}"][col], s=8, color=c, alpha=0.45, lw=0, zorder=3)
        ax.set_xscale("log")
        ax.set_xticks(NS, ["500", "2,000", "5,000"])
        ax.minorticks_off()
        ax.set_xlim(380, 6600)
        ax.set_xlabel("Pre-training images (N)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25, lw=0.5)
        title(ax, letter, f"{lab} on held-out folds")
    axes[0].set_ylim(0.80, 0.95)
    axes[1].set_ylim(0.78, 0.92)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=7.2)
    save(fig, "Fig04_finetune")


# ------------------------------------------------------------------ Figure 5: contrasts and variation
def fig_contrasts():
    con = tab("T3_contrasts")
    var = json.loads((TAB / "T3b_variation.json").read_text())
    names = [("camera_trap_minus_GBIF", "Camera-trap − GBIF\n(matched N, 9 pairs)"),
             ("N5000_minus_N500", "N 5,000 − N 500\n(within source, 6 pairs)"),
             ("auxiliary_minus_control", "Auxiliary − control\n(18 pairs)")]
    fig = plt.figure(figsize=(W, 2.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 0.92, 0.9])
    for gi, (est, letter, ttl) in enumerate((("fixed 60 epochs, final weights (primary)", "a", "60 epochs"),)):
        ax = fig.add_subplot(gs[0, gi])
        d = con[con.estimate == est]
        for i, (k, lab) in enumerate(names):
            r = d[d.contrast == k].iloc[0]
            ax.scatter([r.fold0, r.fold1, r.fold2], [i] * 3, s=22, color=["#7A7A7A", "#A6A6A6", "#C9C9C9"], ec="k", lw=0.3, zorder=3)
            ax.scatter(r.mean_delta, i, marker="|", s=260, color="k", lw=1.6, zorder=4)
            ax.text(0.062, i, f"{r.mean_delta:+.3f}\n{int(r.folds_positive)}/3 folds", va="center", fontsize=6.4)
        ax.axvline(0, color="k", lw=0.7)
        ax.axvspan(-var["run_to_run_epoch30_sd_per_run"], var["run_to_run_epoch30_sd_per_run"], color="#E8E8E8", lw=0, zorder=0)
        ax.set_yticks(range(3), [n[1] for n in names], fontsize=6.8)
        ax.invert_yaxis()
        ax.set_xlim(-0.04, 0.095)
        ax.set_xlabel("Δ mAP@0.5")
        title(ax, letter, ttl)
    ax = fig.add_subplot(gs[0, 1])
    for i, (k, lab) in enumerate(names):
        for j, (est, c) in enumerate((("after 30 epochs, training run 1", "#9E9E9E"), ("after 30 epochs, training run 2", "#3D3D3D"))):
            r = con[(con.estimate == est) & (con.contrast == k)].iloc[0]
            y = i + (-0.14 if j == 0 else 0.14)
            ax.scatter([r.fold0, r.fold1, r.fold2], [y] * 3, s=16, color=c if j else "white", ec=c, lw=0.9, zorder=3)
            ax.scatter(r.mean_delta, y, marker="|", s=120, color=c, lw=1.5, zorder=4)
    ax.axvline(0, color="k", lw=0.7)
    ax.axvspan(-var["run_to_run_epoch30_sd_per_run"], var["run_to_run_epoch30_sd_per_run"], color="#E8E8E8", lw=0, zorder=0)
    ax.set_yticks(range(3), ["", "", ""])
    ax.invert_yaxis()
    ax.set_xlim(-0.04, 0.085)
    ax.set_xlabel("Δ mAP@0.5")
    ax.scatter([], [], s=16, color="white", ec="#9E9E9E", lw=0.9, label="run 1")
    ax.scatter([], [], s=16, color="#3D3D3D", label="run 2")
    ax.legend(loc="upper right", fontsize=6.6, handletextpad=0.2, borderaxespad=0.2)
    title(ax, "b", "30 epochs")
    ax = fig.add_subplot(gs[0, 2])
    vals = [("Fold means", var["sd_fold_means"], "#7A7A7A"), ("Configuration\nmeans", var["sd_config_means"], C_PB),
            ("Configuration\n× fold residual", var["residual_sd"], "#A0A0A0"), ("Run-to-run\n(per run)", var["run_to_run_epoch30_sd_per_run"], "#C0504D")]
    ax.barh(range(4), [v[1] for v in vals], color=[v[2] for v in vals], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v[1] + 0.0006, i, f"{v[1]:.3f}", va="center", fontsize=7)
    ax.set_yticks(range(4), [v[0] for v in vals], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.036)
    ax.set_xlabel("SD of mAP@0.5")
    title(ax, "c", "Variation")
    save(fig, "Fig05_contrasts_variation")


# ------------------------------------------------------------------ Figure 6: convergence
def fig_convergence():
    cv = tab("S_curves_run2")
    conv = tab("T_convergence")
    conv = conv[conv.run.str.startswith("training run 2")]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.35), sharey=True, constrained_layout=True)
    ctrl = cv[cv.config == "coco"].groupby("epoch").mAP50.agg(["mean", "std"])
    e80 = conv.set_index("config").epochs_to_080_mean_curve
    for ax, n, letter in zip(axes, NS, "abc"):
        ax.plot(ctrl.index, ctrl["mean"], color=C_GT, lw=1.3, label="COCO-only control")
        for src, c, name in (("gbif", C_GBIF, "GBIF"), ("ct", C_CT, "Camera-trap")):
            g = cv[cv.config == f"{src}{n}"].groupby("epoch").mAP50.agg(["mean", "std"])
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=c, alpha=0.15, lw=0)
            ax.plot(g.index, g["mean"], color=c, lw=1.5, label=name)
        ax.axhline(0.80, color="k", ls=":", lw=0.8)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 61)
        ax.set_xlabel("Fine-tuning epoch")
        ax.grid(alpha=0.25, lw=0.5)
        ax.text(59, 0.06, f"epochs to 0.80\ncontrol {int(e80['coco'])}\nGBIF {int(e80[f'gbif{n}'])}\ncamera-trap {int(e80[f'ct{n}'])}",
                ha="right", va="bottom", fontsize=6.6, color="#333333")
        title(ax, letter, f"N = {n:,}")
    axes[0].set_ylabel("mAP@0.5 on held-out fold")
    axes[0].legend(loc="center right", fontsize=6.6, bbox_to_anchor=(1.0, 0.52))
    save(fig, "Fig06_convergence")


# ------------------------------------------------------------------ Figure 7: transfer matrix
def fig_transfer():
    t = tab("T4_transfer_matrix")
    labels = ["GBIF-500", "GBIF-2000", "GBIF-5000", "CT-500", "CT-2000", "CT-5000", "PhoneBee\n(fold models)"]
    cols = ["PhoneBee", "CameraTrap", "GBIF"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.2), constrained_layout=True)
    for ax, sz, letter in zip(axes, (640, 1280), "ab"):
        M = t[[f"{c}@{sz}" for c in cols]].values
        im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ind = (i < 3 and j == 2) or (3 <= i < 6 and j == 1) or (i == 6 and j == 0)
                v = M[i, j]
                ax.text(j, i, f"{v:.3f}{'*' if ind else ''}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else "black")
        ax.set_xticks(range(3), ["PhoneBee", "Camera-trap", "GBIF"])
        ax.set_yticks(range(7), labels if letter == "a" else [""] * 7, fontsize=7)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        title(ax, letter, f"Inference at {sz:,} px")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("mAP@0.5")
    save(fig, "Fig07_transfer_matrix")


# ------------------------------------------------------------------ Figure 8: confidence
def fig_confidence():
    z = {"GBIF-5000": np.load(ORIG / "gap_fill_results__1_/conf_gbif5000.npz")["max_conf"],
         "Camera-trap-5000": np.load(ORIG / "gap_fill_results__1_/conf_ct5000.npz")["max_conf"],
         "PhoneBee fold models": np.load(REV / "conf_infold.npz")["max_conf"]}
    cols = {"GBIF-5000": C_GBIF, "Camera-trap-5000": C_CT, "PhoneBee fold models": C_PB}
    ft = tab("T2_finetune_summary").set_index("config")
    tm = tab("T4_transfer_matrix")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), constrained_layout=True, gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    for k, v in z.items():
        ax.hist(v, bins=bins, color=cols[k], alpha=0.7, label=f"{k} (median {np.median(v):.3f})")
    ax.axvline(0.25, color="#B03A2E", ls="--", lw=1)
    ax.text(0.265, ax.get_ylim()[1] * 0.42, "operating\nthreshold\n0.25", color="#B03A2E", fontsize=6.6, va="top")
    ax.set_xlabel("Highest detection confidence per frame")
    ax.set_ylabel("Frames")
    ax.legend(loc="upper right", fontsize=6.3, bbox_to_anchor=(1.0, 1.0))
    title(ax, "a", "Maximum confidence per frame (n = 333)")
    ax = axes[1]
    frac = [np.mean(z[k] >= 0.25) * 100 for k in z]
    mapv = [tm.loc[2, "PhoneBee@1280"] * 100, tm.loc[5, "PhoneBee@1280"] * 100, ft.loc["coco", "mAP50_mean"] * 100]
    x = np.arange(3)
    ax.bar(x - 0.19, frac, 0.36, color="#8C8C8C", label="frames with a detection ≥ 0.25 (%)")
    ax.bar(x + 0.19, mapv, 0.36, color=C_PB, label="mAP@0.5 × 100")
    for i in range(3):
        ax.text(x[i] - 0.19, frac[i] + 2, f"{frac[i]:.0f}", ha="center", fontsize=7)
        ax.text(x[i] + 0.19, mapv[i] + 2, f"{mapv[i]:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x, ["GBIF-5000", "Camera-trap\n-5000", "PhoneBee\nfold models"], fontsize=7)
    ax.set_ylim(0, 138)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Percent")
    ax.legend(loc="upper left", fontsize=6.3, borderaxespad=0.1)
    title(ax, "b", "Confident frames and accuracy")
    save(fig, "Fig08_confidence")


# ------------------------------------------------------------------ Figure 9: size and location of predictions
def sides(path, conf=0.25):
    z = np.load(path)
    out = []
    for k in z.files:
        if k.endswith("_b"):
            b = z[k].astype(float)
            s = z[k[:-2] + "_s"].astype(float)
            b = b[s >= conf]
            out.append(np.sqrt((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])) / np.sqrt(1920 * 1080))
    return np.concatenate(out)


def fig_scale_location():
    gt = np.load(ORIG / "box_size_distributions.npz")["gt_side"] * 100
    ct = sides(REV / "sweep/ct5000@1280_z1.npz") * 100
    pb = sides(REV / "sweep/infold@1280_z1.npz") * 100
    loc = json.loads((TAB / "T_ct_detection_locations.json").read_text())["ct5000@1280_z1"]
    ex = json.loads((REV / "examples/example_predictions.json").read_text())
    fr = [k for k in ex if k.startswith("frame_037436")][0]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.65), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    bins = np.logspace(np.log10(0.5), np.log10(50), 70)
    ax.hist(gt, bins=bins, color=C_GT, alpha=0.75, label=f"bees ({len(gt)})")
    ax.hist(pb, bins=bins, color=C_PB, alpha=0.6, label=f"fold models ({len(pb)})")
    ax.hist(ct, bins=bins, color=C_CT, alpha=0.75, label=f"CT-5000 ({len(ct)})")
    ax.axvline(5.02, color=C_CT, ls="--", lw=1)
    ax.axvline(32.31, color=C_GBIF, ls="--", lw=1)
    ax.text(5.4, ax.get_ylim()[1] * 0.55, "camera-trap\ntraining scale", color=C_CT, fontsize=6.3, va="top")
    ax.text(30.0, ax.get_ylim()[1] * 0.97, "GBIF\ntraining\nscale", color=C_GBIF, fontsize=6.3, ha="right", va="top")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_xlabel("Box side (% of frame side)")
    ax.set_ylabel("Boxes")
    ax.legend(loc="upper left", fontsize=6.2, borderaxespad=0.1)
    title(ax, "a", "Box size on PhoneBee")
    ax = axes[1]
    im = Image.open(REV / f"examples/{fr[:12]}.jpg")
    ax.imshow(im)
    legend_lines = []
    for letter, ((cx, cy), n) in zip("ABC", zip(loc["centres"], loc["counts"])):
        ax.add_patch(plt.Circle((cx, cy), 42, fill=False, ec="#FFD400", lw=1.8))
        ax.text(cx, cy - 52, letter, color="black", fontsize=7, ha="center", va="bottom", fontweight="bold",
                bbox=dict(fc="#FFD400", ec="none", pad=0.8, alpha=0.95))
        legend_lines.append(f"{letter}: {n}")
    ax.text(30, 30, "detections at\n" + "\n".join(legend_lines), fontsize=6.6, va="top", ha="left",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    for x1, y1, x2, y2 in ex[fr]["gt"]:
        ax.add_patch(Rectangle((x1 - 8, y1 - 8), x2 - x1 + 16, y2 - y1 + 16, fill=False, ec="white", lw=0.9))
    ax.axis("off")
    title(ax, "b", f"Where {loc['n_detections']} confident detections fall")
    save(fig, "Fig09_size_location")


# ------------------------------------------------------------------ Figure 10: example frames
def fig_examples():
    ex = json.loads((REV / "examples/example_predictions.json").read_text())
    keys = [k for p in ("frame_015328", "frame_016854", "frame_001543") for k in ex if k.startswith(p)]
    fig, axes = plt.subplots(3, 3, figsize=(W, 3.7), constrained_layout=True)
    for i, k in enumerate(keys):
        im = Image.open(REV / f"examples/{k[:12]}.jpg")
        for j, (tag, c, name) in enumerate((("gbif5000", C_GBIF, "GBIF-5000"), ("ct5000", C_CT, "Camera-trap-5000"),
                                            ("infold", C_PB, "PhoneBee fold model"))):
            ax = axes[i, j]
            ax.imshow(im)
            for x1, y1, x2, y2 in ex[k]["gt"]:
                ax.add_patch(Rectangle((x1 - 10, y1 - 10), x2 - x1 + 20, y2 - y1 + 20, fill=False, ec="white", lw=0.8, ls=":"))
            for x1, y1, x2, y2 in ex[k][tag]["boxes"]:
                ax.add_patch(Rectangle((x1 - 4, y1 - 4), x2 - x1 + 8, y2 - y1 + 8, fill=False, ec=c, lw=1.4))
            n = len(ex[k][tag]["boxes"])
            ax.text(25, 60, f"{n} detection{'s' if n != 1 else ''}", color="white", fontsize=6.3, va="top",
                    bbox=dict(fc=c, ec="none", alpha=0.85, pad=1.2))
            ax.axis("off")
            if i == 0:
                ax.set_title(name, color=c, fontsize=8.5, loc="center")
    save(fig, "Fig10_example_frames")



# ------------------------------------------------------------------ Figure 1: exemplar images of the three corpora
def fig_exemplars():
    man = json.loads((ORIG / "exemplars/manifest.json").read_text())["exemplars"]
    gb = json.loads((ORIG / "exemplars/gbif_ccby/gbif_exemplars.json").read_text())
    rows = [("GBIF occurrence\nphotographs\nmedian side 32.3%", C_GBIF, 1.25,
             [(ORIG / "exemplars/gbif_ccby" / g["file"], g["boxes_xywhn"]) for g in gb]),
            ("Insect camera trap\nmedian side 5.02%", C_CT, 2.4,
             [(ORIG / "exemplars/images" / e["file"], e["boxes_xywhn"]) for e in man["camera_trap"][:3]]),
            ("PhoneBee\nsmartphone video\nmedian side 1.36%", C_PB, 3.0,
             [(ORIG / "exemplars/images" / e["file"], e["boxes_xywhn"]) for e in man["phonebee"][:3]])]
    fig = plt.figure(figsize=(W, 4.7))
    gs = fig.add_gridspec(3, 4, left=0.085, right=0.995, top=0.94, bottom=0.01, wspace=0.05, hspace=0.08,
                          width_ratios=[1, 1, 1, 0.62])
    for r, (label, colour, zoom, items) in enumerate(rows):
        for c, (path, boxes) in enumerate(items):
            im = Image.open(path).convert("RGB")
            w_img, h_img = im.size
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(im)
            pad = 0.006 * max(w_img, h_img)
            for x, y, w, h in boxes:
                ax.add_patch(Rectangle(((x - w / 2) * w_img - pad, (y - h / 2) * h_img - pad), w * w_img + 2 * pad,
                                       h * h_img + 2 * pad, fill=False, ec=colour, lw=1.3))
            ax.axis("off")
        path, boxes = items[0]
        im = Image.open(path).convert("RGB")
        w_img, h_img = im.size
        x, y, w, h = boxes[0]
        side = max(w * w_img, h * h_img) * zoom
        cx, cy = x * w_img, y * h_img
        x0, y0 = max(0, cx - side / 2), max(0, cy - side / 2)
        crop = im.crop((int(x0), int(y0), int(min(w_img, x0 + side)), int(min(h_img, y0 + side))))
        ax = fig.add_subplot(gs[r, 3])
        ax.imshow(crop)
        ax.add_patch(Rectangle(((x - w / 2) * w_img - x0, (y - h / 2) * h_img - y0), w * w_img, h * h_img,
                               fill=False, ec="white", lw=0.9, ls=":"))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(colour)
            sp.set_linewidth(2.0)
        if r == 0:
            ax.set_title("focal bee,\nmagnified", fontsize=7, color="#444444")
        fig.text(0.012, 0.94 - (r + 0.5) * 0.31, label, rotation=90, ha="left", va="center", fontsize=7.6,
                 fontweight="bold", color=colour, multialignment="center")
    save(fig, "Fig01_exemplars")


# ------------------------------------------------------------------ Figure 12: object scale and annotation density
def fig_corpora():
    d = np.load(ORIG / "paper_assets_slim/dataset_distributions.npz")
    g = np.load(ORIG / "gap_fill_results/gbif_distributions.npz")
    sets = (("PhoneBee", C_PB, np.sqrt(d["flower333__rel_area"]), d["flower333__per_img"]),
            ("camera-trap", C_CT, np.sqrt(d["camera_trap__rel_area"]), d["camera_trap__per_img"]),
            ("GBIF", C_GBIF, np.sqrt(g["GBIF__rel_area"]), g["GBIF__per_img"]))
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1]), constrained_layout=True)
    ax = axes[0]
    bins = np.logspace(np.log10(0.3), np.log10(100), 60)
    for name, colour, side, _ in sets:
        ax.hist(side * 100, bins=bins, color=colour, alpha=0.6,
                label=f"{name} (n = {len(side)}, median {np.median(side) * 100:.2f}%)")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("Box side (% of image side)")
    ax.set_ylabel("Boxes")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.legend(frameon=False, fontsize=6.4, loc="upper left")
    title(ax, "a", "Object scale")
    ax = axes[1]
    means = [float(np.mean(pi)) for _, _, _, pi in sets]
    ax.bar(range(3), means, color=[c for _, c, _, _ in sets], width=0.6)
    for i, (m, (_, _, _, pi)) in enumerate(zip(means, sets)):
        ax.text(i, m + 0.04, f"{m:.2f}\n({len(pi)} images)", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels([n for n, _, _, _ in sets])
    ax.set_ylabel("Annotated objects per image (mean)")
    ax.set_ylim(0, max(means) * 1.3)
    title(ax, "b", "Annotation density")
    save(fig, "Fig12_corpora")

# ------------------------------------------------------------------ Figure 11: random crops
def fig_crops():
    fig = plt.figure(figsize=(W, 5.0))
    blocks = (("ct5000", C_CT, 0.935, 0.515, "a   Camera-trap-5000: random confident detections (confidence \u2265 0.5)"),
              ("gt", C_GT, 0.445, 0.025, "b   Random annotated bees"))
    for tag, colour, top, bottom, title in blocks:
        gs = fig.add_gridspec(2, 6, left=0.01, right=0.99, top=top, bottom=bottom, wspace=0.06, hspace=0.06)
        for r in range(2):
            for c in range(6):
                ax = fig.add_subplot(gs[r, c])
                ax.imshow(Image.open(REV / f"examples/crop_{tag}_{r * 6 + c:02d}.jpg").convert("RGB"))
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_color(colour)
                    sp.set_linewidth(1.6)
        fig.text(0.01, top + 0.012, title, fontsize=8.5, fontweight="bold", ha="left", va="bottom")
    save(fig, "Fig11_random_crops")


# ------------------------------------------------------------------ Figure 12: object-scale manipulation
def fig_scale_manipulation():
    sw = tab("S_scale_sweep")
    ex = json.loads((REV / "examples/example_predictions.json").read_text())
    fr = [k for k in ex if k.startswith("frame_037436")][0]
    fig = plt.figure(figsize=(W, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    ax = fig.add_subplot(gs[0, 0])
    im = Image.open(REV / f"examples/{fr[:12]}.jpg")
    ax.imshow(im)
    z, ov = 3.7, 0.25
    tw, th = 1920 / z, 1080 / z

    def starts(L, t):
        s = list(np.arange(0.0, L - t + 1e-6, t * (1 - ov)))
        if L - t - s[-1] > 0.5:
            s.append(L - t)
        return s
    for y in starts(1080, th):
        for x in starts(1920, tw):
            ax.add_patch(Rectangle((x, y), tw, th, fill=False, ec="white", lw=0.5, alpha=0.8))
    tiles = [(x, y) for y in starts(1080, th) for x in starts(1920, tw)]
    count = [sum(1 for x1, y1, x2, y2 in ex[fr]["gt"] if x1 >= x and y1 >= y and x2 <= x + tw and y2 <= y + th) for x, y in tiles]
    hx, hy = tiles[int(np.argmax(count))]
    ax.add_patch(Rectangle((hx, hy), tw, th, fill=False, ec="#FFD400", lw=1.6))
    ax.axis("off")
    title(ax, "a", "Sliced inference at 3.7×")
    ax2 = fig.add_subplot(gs[0, 1])
    tile = im.crop((int(hx), int(hy), int(hx + tw), int(hy + th)))
    ax2.imshow(tile)
    for x1, y1, x2, y2 in ex[fr]["gt"]:
        if x1 >= hx and y1 >= hy and x2 <= hx + tw and y2 <= hy + th:
            ax2.add_patch(Rectangle((x1 - hx - 4, y1 - hy - 4), x2 - x1 + 8, y2 - y1 + 8, fill=False, ec="white", lw=0.9, ls=":"))
    ax2.axis("off")
    title(ax2, "b", "One input tile (bees outlined)")
    ax3 = fig.add_subplot(gs[1, :])
    series = [("infold@1280", C_PB, "-o", "PhoneBee fold models (1,280 px)"),
              ("ct5000@640", C_CT, "-s", "Camera-trap-5000 (640 px)"),
              ("ct5000@1280", C_CT, "--s", "Camera-trap-5000 (1,280 px)"),
              ("gbif5000@640", C_GBIF, "-^", "GBIF-5000 (640 px)"),
              ("gbif5000@640_subset", C_GBIF, ":^", "GBIF-5000 (640 px, 40-frame subset)")]
    for lab, c, st_, name in series:
        d = sw[sw.series == lab].sort_values("zoom")
        x = d.bee_side_rel_tile * 100
        ax3.plot(x, d.AP50, st_, color=c, ms=4.5, lw=1.4, label=name, mfc="white" if "subset" in lab or "1280" in lab and "ct" in lab else c)
        ax3.fill_between(x, d.AP50_lo, d.AP50_hi, color=c, alpha=0.12, lw=0)
    for xv, c, t in ((1.358, C_PB, "PhoneBee\ntraining scale"), (5.02, C_CT, "camera-trap\ntraining scale"), (32.31, C_GBIF, "GBIF\ntraining scale")):
        ax3.axvline(xv, color=c, ls=":", lw=1)
        ax3.text(xv * (0.95 if xv > 30 else 1.05), 0.62 if xv < 3 else (0.40 if xv < 30 else 0.30), t, color=c, fontsize=6.5, va="top",
                 ha="right" if xv > 30 else "left")
    ax3.set_xscale("log")
    ax3.set_xticks([1, 2, 5, 10, 20, 35], ["1", "2", "5", "10", "20", "35"])
    ax3.minorticks_off()
    ax3.set_xlim(1.2, 40)
    ax3.set_ylim(-0.02, 0.95)
    ax3.set_xlabel("Apparent bee size: median bee side as % of the input tile (PhoneBee frames magnified 1–24×)")
    ax3.set_ylabel("AP@0.5 on PhoneBee")
    ax3.grid(alpha=0.25, lw=0.5)
    ax3.legend(loc="upper right", fontsize=6.5, ncol=1, bbox_to_anchor=(1.0, 1.0))
    title(ax3, "c", "Detection versus apparent bee size")
    save(fig, "Fig14_scale_manipulation")


def fig_scale_small_n():
    sw = tab("S_scale_sweep")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.35), sharey=True, constrained_layout=True)
    for ax, src, c, letter in ((axes[0], "ct", C_CT, "a"), (axes[1], "gbif", C_GBIF, "b")):
        for n, mk, ls in ((500, "o", ":"), (2000, "s", "--"), (5000, "^", "-")):
            d = sw[sw.series == f"{src}{n}@640"].sort_values("zoom")
            ax.plot(d.bee_side_rel_tile * 100, d.AP50, ls, marker=mk, color=c, ms=4, lw=1.2, label=f"N = {n:,}")
            ax.fill_between(d.bee_side_rel_tile * 100, d.AP50_lo, d.AP50_hi, color=c, alpha=0.1, lw=0)
        ax.axvline(5.02 if src == "ct" else 32.31, color=c, ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 14], ["1", "2", "5", "10", "14"])
        ax.minorticks_off()
        ax.set_xlim(1.1, 15)
        ax.set_xlabel("Median bee side as % of input tile")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(loc="upper left", fontsize=6.8)
        title(ax, letter, "Camera-trap models (640 px)" if src == "ct" else "GBIF models (640 px)")
    axes[0].set_ylabel("AP@0.5 on PhoneBee")
    save(fig, "FigS2_scale_manipulation_by_N")


def fig_curves_both_runs():
    D = ORIG
    fig, axes = plt.subplots(2, 4, figsize=(W, 3.6), sharey=True, sharex=True, constrained_layout=True)
    cv = tab("S_curves_run2")
    names = {"coco": "COCO-only control", "gbif500": "GBIF-500", "gbif2000": "GBIF-2000", "gbif5000": "GBIF-5000",
             "ct500": "CT-500", "ct2000": "CT-2000", "ct5000": "CT-5000"}
    shades = ["#1b1b1b", "#6b6b6b", "#b0b0b0"]
    for ax, cfg in zip(axes.ravel(), CONFIGS):
        for k in range(3):
            d = cv[(cv.config == cfg) & (cv.fold == k)]
            ax.plot(d.epoch, d.mAP50, color=shades[k], lw=1.0, label=f"fold {k + 1}, run 2")
            if cfg == "coco":
                p = D / f"paper_assets_slim/baseline_runs/fold{k}/results.csv"
            elif cfg.startswith("ct"):
                p = D / f"ct_study_results/ct_study/ft_runs/{cfg}_fold{k}/results.csv"
            else:
                p = sorted((D / f"{cfg}_result").rglob(f"*fold{k}*/results.csv"))[0]
            o = pd.read_csv(p)
            o.columns = [c.strip() for c in o.columns]
            ax.plot(o["epoch"], o["metrics/mAP50(B)"], color=shades[k], lw=0.9, ls="--", alpha=0.9, label=f"fold {k + 1}, run 1")
        ax.axhline(0.8, color="k", ls=":", lw=0.6)
        ax.set_title(names[cfg], fontsize=8, loc="left")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2, lw=0.4)
    axes[1, 3].axis("off")
    h, l = axes[0, 0].get_legend_handles_labels()
    axes[1, 3].legend(h, l, loc="center", fontsize=6.8)
    for ax in axes[1, :3]:
        ax.set_xlabel("Epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("mAP@0.5")
    save(fig, "FigS1_learning_curves")


# ------------------------------------------------------------------ Figure 14: appearance distance
def fig_clip():
    E = dict(np.load(ORIG / "gap_fill_results__1_/clip_embeddings_fixed.npz"))
    E = {k: v / np.linalg.norm(v, axis=1, keepdims=True) for k, v in E.items()}
    X = np.concatenate([E["flower333"], E["camera_trap"], E["GBIF"]])
    lab = np.array(["PhoneBee"] * len(E["flower333"]) + ["Camera-trap"] * len(E["camera_trap"]) + ["GBIF"] * len(E["GBIF"]))
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T
    ev = (S ** 2 / (S ** 2).sum())[:2] * 100
    fd = tab("T_clip_frechet")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.65), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    for name, c in (("GBIF", C_GBIF), ("Camera-trap", C_CT), ("PhoneBee", C_PB)):
        m = lab == name
        ax.scatter(P[m, 0], P[m, 1], s=6, alpha=0.55, color=c, lw=0, label=f"{name} (n = {m.sum()})")
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.legend(loc="best", fontsize=6.8, markerscale=2)
    title(ax, "a", "CLIP image embeddings")
    ax = axes[1]
    rows = [("PhoneBee ↔ camera-trap", "flower333|camera_trap"), ("Camera-trap ↔ GBIF", "camera_trap|GBIF"),
            ("PhoneBee ↔ GBIF", "flower333|GBIF"), ("within PhoneBee", "flower333|flower333 split-half"),
            ("within camera-trap", "camera_trap|camera_trap split-half"), ("within GBIF", "GBIF|GBIF split-half")]
    for i, (name, key) in enumerate(rows):
        r = fd[fd.pair == key].iloc[0]
        col = "#4D4D4D" if i < 3 else "#B3B3B3"
        ax.barh(i, r.frechet, color=col, height=0.6)
        ax.errorbar(r.frechet, i, xerr=[[r.frechet - r.ci_lo], [r.ci_hi - r.frechet]], color="k", lw=0.8, capsize=2)
        ax.text(r.ci_hi + 0.012, i, f"{r.frechet:.3f}", va="center", fontsize=7)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.66)
    ax.set_xlabel("Fréchet distance (95% bootstrap CI)")
    title(ax, "b", "Distance between corpora")
    save(fig, "Fig13_clip_distance")


if __name__ == "__main__":
    fig_design()
    fig_within()
    fig_finetune()
    fig_contrasts()
    fig_convergence()
    fig_transfer()
    fig_confidence()
    fig_scale_location()
    fig_examples()
    fig_crops()
    fig_exemplars()
    fig_corpora()
    fig_scale_manipulation()
    fig_scale_small_n()
    fig_curves_both_runs()
    fig_clip()
    print("figures written to", FIG)
