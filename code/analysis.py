"""Tables for the PhoneBee transfer study from the saved run outputs.

Inputs : ORIG  folder with the outputs of the auxiliary pre-training, fine-tuning,
               transfer, confidence, scale and CLIP analyses (Kaggle runs 1-6)
         REV   folder 'revision' produced by revision_experiments_kaggle.py
Outputs: CSV and JSON tables in OUT
Usage  : python analysis_revision.py ORIG REV OUT
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ORIG, REV, OUT = (Path(a) for a in sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
CONFIGS = ["coco", "gbif500", "gbif2000", "gbif5000", "ct500", "ct2000", "ct5000"]
FOLDS = (0, 1, 2)
GM_PB = np.sqrt(1920 * 1080)          # geometric-mean side of a PhoneBee frame (px)
rng = np.random.default_rng(42)


def jload(p):
    return json.loads(Path(p).read_text())


def save(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False)
    return df


st = jload(REV / "state.json")
ct_state = jload(ORIG / "ct_study_results/state.json")
gap = jload(ORIG / "gap_fill_results__1_/gap_state.json")
assets = jload(ORIG / "paper_assets_slim/assets_state.json")

# ------------------------------------------------------------------ 1. within-domain competence
rows = []
for n in (500, 2000, 5000):
    for sz in (640, 1280):
        r = gap["transfer"][f"ct{n}->camera_trap@{sz}"]
        rows.append(dict(source="camera-trap", N=n, imgsz=sz, test_set="camera-trap held-out test (1300 images)", **{k: r[k] for k in ("precision", "recall", "f1", "mAP50", "mAP50_95")}))
    for sz in (640, 1280):
        r = st["transfer"][f"gbif{n}->GBIFdisjoint@{sz}"]
        f1 = 2 * r["precision"] * r["recall"] / (r["precision"] + r["recall"])
        rows.append(dict(source="GBIF", N=n, imgsz=sz, test_set="GBIF reference set disjoint from all subsets (300 images)",
                         precision=r["precision"], recall=r["recall"], f1=f1, mAP50=r["mAP50"], mAP50_95=r["mAP50_95"]))
        o = gap["transfer"][f"gbif{n}->GBIF@{sz}"]
        rows.append(dict(source="GBIF", N=n, imgsz=sz, test_set="original GBIF reference set (overlaps the subsets)",
                         precision=o["precision"], recall=o["recall"], f1=o["f1"], mAP50=o["mAP50"], mAP50_95=o["mAP50_95"]))
save(pd.DataFrame(rows), "T1_within_domain")
(OUT / "gbif_reference_audit.json").write_text(json.dumps(dict(st["phase0"], disjoint_set=dict(
    {k: v for k, v in st["gbif_labels"].items() if k != "pairs"})), indent=2))

# ------------------------------------------------------------------ 2. fine-tuning
refit = []
for k, v in st["refit"].items():
    cfg, fold = k.rsplit("_fold", 1)
    refit.append(dict(config=cfg, fold=int(fold), protocol="fixed 60 epochs, final weights",
                      **v["final"], mAP50_at640=v["final_at640"]["mAP50"]))
refit = pd.DataFrame(refit)
orig = []
for k in FOLDS:
    orig.append(dict(config="coco", fold=k, **assets["baseline"][str(k)]))
for n in (500, 2000, 5000):
    g = jload(ORIG / f"gbif{n}_result/C1_gbif{n}_result.json")["folds"]
    for k in FOLDS:
        orig.append(dict(config=f"gbif{n}", fold=k, **g[k]))
        orig.append(dict(config=f"ct{n}", fold=k, **ct_state["flower_ft"][f"ct{n}"][str(k)]))
orig = pd.DataFrame(orig)
orig["protocol"] = "early stopping and best epoch on the held-out fold"
save(pd.concat([refit, orig], ignore_index=True), "S1_fold_metrics")


def matrix(df, col="mAP50"):
    return df.pivot(index="config", columns="fold", values=col).reindex(CONFIGS)


M = matrix(refit)
summ = []
for c in CONFIGS:
    d = refit[refit.config == c]
    delta = (M.loc[c] - M.loc["coco"]).values
    summ.append(dict(config=c, mAP50_mean=d.mAP50.mean(), mAP50_sd=d.mAP50.std(ddof=1),
                     recall_mean=d.recall.mean(), recall_sd=d.recall.std(ddof=1),
                     precision_mean=d.precision.mean(), mAP50_95_mean=d.mAP50_95.mean(),
                     mAP50_at640_mean=d.mAP50_at640.mean(),
                     delta_vs_control=delta.mean() if c != "coco" else np.nan,
                     delta_min=delta.min() if c != "coco" else np.nan,
                     delta_max=delta.max() if c != "coco" else np.nan,
                     folds_above_control=int((delta > 0).sum()) if c != "coco" else np.nan))
save(pd.DataFrame(summ), "T2_finetune_summary")


# ------------------------------------------------------------------ 3. contrasts
def epoch_table(root, pattern_fn):
    out = {}
    for c in CONFIGS:
        for k in FOLDS:
            d = pd.read_csv(pattern_fn(root, c, k))
            d.columns = [x.strip() for x in d.columns]
            out[(c, k)] = d
    return out


def orig_curve_path(root, c, k):
    if c == "coco":
        return root / f"paper_assets_slim/baseline_runs/fold{k}/results.csv"
    if c.startswith("ct"):
        return root / f"ct_study_results/ct_study/ft_runs/{c}_fold{k}/results.csv"
    hits = sorted(glob.glob(str(root / f"{c}_result" / "**" / f"*fold{k}*" / "results.csv"), recursive=True))
    return Path(hits[0])


cur_new = epoch_table(REV, lambda r, c, k: r / "ft_runs" / f"{c}_fold{k}" / "results.csv")
cur_old = epoch_table(ORIG, orig_curve_path)


def at_epoch(curves, e):
    return pd.DataFrame({c: [curves[(c, k)]["metrics/mAP50(B)"].iloc[e - 1] for k in FOLDS] for c in CONFIGS}).T


def contrasts(Mx, label):
    aux = Mx.drop("coco") - Mx.loc["coco"]
    sets = dict(
        camera_trap_minus_GBIF=pd.DataFrame([Mx.loc[f"ct{n}"] - Mx.loc[f"gbif{n}"] for n in (500, 2000, 5000)]),
        N5000_minus_N500=pd.DataFrame([Mx.loc[f"{s}5000"] - Mx.loc[f"{s}500"] for s in ("gbif", "ct")]),
        auxiliary_minus_control=aux,
        camera_trap_minus_control=aux.loc[["ct500", "ct2000", "ct5000"]],
        GBIF_minus_control=aux.loc[["gbif500", "gbif2000", "gbif5000"]])
    rows = []
    for name, d in sets.items():
        fm = d.mean(axis=0)
        rows.append(dict(estimate=label, contrast=name, n_pairs=int(d.size), mean_delta=float(d.values.mean()),
                         fold0=float(fm[0]), fold1=float(fm[1]), fold2=float(fm[2]),
                         folds_positive=int((fm > 0).sum()), pairs_positive=int((d.values > 0).sum())))
    return rows


M30_new, M30_old = at_epoch(cur_new, 30), at_epoch(cur_old, 30)
Morig = matrix(orig)
con = (contrasts(M, "fixed 60 epochs, final weights (primary)")
       + contrasts(M30_new, "after 30 epochs, training run 2")
       + contrasts(M30_old, "after 30 epochs, training run 1")
       + contrasts(Morig, "best epoch selected on held-out fold, training run 1"))
save(pd.DataFrame(con), "T3_contrasts")

fold_est = []
for label, mat in (("fixed 60 epochs, final weights (primary)", M), ("after 30 epochs, training run 2", M30_new),
                   ("after 30 epochs, training run 1", M30_old),
                   ("best epoch selected on held-out fold, training run 1", Morig)):
    for c in CONFIGS:
        for k in FOLDS:
            fold_est.append(dict(estimate=label, config=c, fold=int(k), mAP50=float(mat.loc[c, k])))
save(pd.DataFrame(fold_est), "S1b_fold_mAP_estimates")

diff30 = (M30_new - M30_old).values.ravel()
grand = M.values.mean()
resid = M.values - M.mean(axis=1).values[:, None] - M.mean(axis=0).values[None, :] + grand
var = dict(
    sd_fold_means=float(M.mean(axis=0).std(ddof=1)),
    sd_config_means=float(M.mean(axis=1).std(ddof=1)),
    residual_sd=float(np.sqrt((resid ** 2).sum() / ((M.shape[0] - 1) * (M.shape[1] - 1)))),
    fold_means={int(k): float(v) for k, v in M.mean(axis=0).items()},
    fold0_highest_in_all_configs=bool((M.idxmax(axis=1) == 0).all()),
    configs_with_fold_order_0_1_2=int(sum(list(M.loc[c].sort_values(ascending=False).index) == [0, 1, 2] for c in CONFIGS)),
    run_to_run_epoch30_mean_diff=float(diff30.mean()),
    run_to_run_epoch30_sd_diff=float(diff30.std(ddof=1)),
    run_to_run_epoch30_sd_per_run=float(diff30.std(ddof=1) / np.sqrt(2)),
    run_to_run_epoch30_median_abs=float(np.median(np.abs(diff30))),
    run_to_run_epoch30_max_abs=float(np.abs(diff30).max()))
(OUT / "T3b_variation.json").write_text(json.dumps(var, indent=2))

# ------------------------------------------------------------------ 4. convergence
conv = []
for label, curves in (("training run 2 (fixed 60 epochs)", cur_new), ("training run 1", cur_old)):
    for c in CONFIGS:
        per = []
        for k in FOLDS:
            m = curves[(c, k)]["metrics/mAP50(B)"].values
            hit = np.where(m >= 0.80)[0]
            per.append(int(hit[0]) + 1 if len(hit) else np.nan)
        L = min(len(curves[(c, k)]) for k in FOLDS)
        mc = np.mean([curves[(c, k)]["metrics/mAP50(B)"].values[:L] for k in FOLDS], axis=0)
        hit = np.where(mc >= 0.80)[0]
        conv.append(dict(run=label, config=c, epochs_to_080_mean_curve=int(hit[0]) + 1 if len(hit) else np.nan,
                         fold0=per[0], fold1=per[1], fold2=per[2], epochs_completed_min=L))
save(pd.DataFrame(conv), "T_convergence")
curves_out = []
for c in CONFIGS:
    for k in FOLDS:
        d = cur_new[(c, k)]
        curves_out += [dict(config=c, fold=k, epoch=int(e), mAP50=float(m)) for e, m in zip(d["epoch"], d["metrics/mAP50(B)"])]
save(pd.DataFrame(curves_out), "S_curves_run2")
curves_old = []
for c in CONFIGS:
    for k in FOLDS:
        d = cur_old[(c, k)]
        curves_old += [dict(config=c, fold=k, epoch=int(e), mAP50=float(m)) for e, m in zip(d["epoch"], d["metrics/mAP50(B)"])]
save(pd.DataFrame(curves_old), "S_curves_run1")

# ------------------------------------------------------------------ 5. transfer matrix
tm = []
for tag in ("gbif500", "gbif2000", "gbif5000", "ct500", "ct2000", "ct5000"):
    row = dict(model=tag)
    for sz in (640, 1280):
        row[f"PhoneBee@{sz}"] = gap["transfer"][f"{tag}->flower333@{sz}"]["mAP50"]
        row[f"CameraTrap@{sz}"] = gap["transfer"][f"{tag}->camera_trap@{sz}"]["mAP50"]
        row[f"GBIF@{sz}"] = st["transfer"][f"{tag}->GBIFdisjoint@{sz}"]["mAP50"]
    tm.append(row)
row = dict(model="PhoneBee (COCO-only fold models)")
for sz in (640, 1280):
    coco = refit[refit.config == "coco"]
    row[f"PhoneBee@{sz}"] = float(coco["mAP50" if sz == 1280 else "mAP50_at640"].mean())
    row[f"CameraTrap@{sz}"] = float(np.mean([st["transfer"][f"coco_fold{k}->CTtest@{sz}"]["mAP50"] for k in FOLDS]))
    row[f"GBIF@{sz}"] = float(np.mean([st["transfer"][f"coco_fold{k}->GBIFdisjoint@{sz}"]["mAP50"] for k in FOLDS]))
tm.append(row)
save(pd.DataFrame(tm), "T4_transfer_matrix")

# ------------------------------------------------------------------ 6. confidence
conf = []
for tag, path in (("GBIF-5000", ORIG / "gap_fill_results__1_/conf_gbif5000.npz"),
                  ("camera-trap-5000", ORIG / "gap_fill_results__1_/conf_ct5000.npz"),
                  ("PhoneBee fold models (held-out folds)", REV / "conf_infold.npz")):
    z = np.load(path)
    mc, nd = z["max_conf"], z["n_dets"]
    conf.append(dict(model=tag, n_frames=len(mc), median_max_conf=float(np.median(mc)), max_max_conf=float(mc.max()),
                     frac_frames_ge_025=float(np.mean(mc >= 0.25)), frac_frames_ge_050=float(np.mean(mc >= 0.5)),
                     mean_candidates_per_frame=float(nd.mean())))
save(pd.DataFrame(conf), "T_confidence")


# ------------------------------------------------------------------ 7. predicted versus true object size
def sides_from_npz(path, conf=0.25):
    z = np.load(path)
    s, b = [], []
    for k in z.files:
        if k.endswith("_b"):
            bb = z[k].astype(float)
            ss = z[k[:-2] + "_s"].astype(float)
            keep = ss >= conf
            s.append(np.sqrt((bb[keep, 2] - bb[keep, 0]) * (bb[keep, 3] - bb[keep, 1])) / GM_PB)
    return np.concatenate(s)


bs = np.load(ORIG / "box_size_distributions.npz")
gt = bs["gt_side"]
ct = sides_from_npz(REV / "sweep/ct5000@1280_z1.npz")
inf = sides_from_npz(REV / "sweep/infold@1280_z1.npz")
gmed, lo, hi = float(np.median(gt)), float(np.percentile(gt, 5)), float(np.percentile(gt, 95))
sz_rows = [dict(model="annotated bees", n=len(gt), median_side=gmed, p5=lo, p95=hi, min=float(gt.min()), max=float(gt.max()))]
for tag, s, train in (("camera-trap-5000 (1280 px)", ct, 0.0502), ("PhoneBee fold models (1280 px)", inf, gmed)):
    sz_rows.append(dict(model=tag, n=len(s), median_side=float(np.median(s)), p5=float(np.percentile(s, 5)),
                        p95=float(np.percentile(s, 95)), min=float(s.min()), max=float(s.max()),
                        ratio_to_truth=float(np.median(s) / gmed), area_ratio=float((np.median(s) / gmed) ** 2),
                        ratio_to_training_scale=float(np.median(s) / train),
                        frac_within_truth_p5_p95=float(np.mean((s >= lo) & (s <= hi))),
                        frac_within_1p5x_truth_median=float(np.mean((s > gmed / 1.5) & (s < gmed * 1.5)))))
save(pd.DataFrame(sz_rows), "T5_object_size")
pix = dict(px_native_median=gmed * GM_PB, px_native_p10=float(np.percentile(gt, 10)) * GM_PB,
           px_native_p90=float(np.percentile(gt, 90)) * GM_PB, px_at_1280=gmed * np.sqrt(1280 * 720),
           px_at_640=gmed * np.sqrt(640 * 360), gt_q25=float(np.percentile(gt, 25)), gt_q75=float(np.percentile(gt, 75)),
           iou_threshold_scale_ratio=float(np.sqrt(2)))
(OUT / "phonebee_pixels.json").write_text(json.dumps(pix, indent=2))


# locations of confident camera-trap detections
def cluster(path, conf=0.25, radius=40.0):
    z = np.load(path)
    pts = []
    for k in z.files:
        if k.endswith("_b"):
            f = int(k[1:-2])
            bb, ss = z[k].astype(float), z[k[:-2] + "_s"].astype(float)
            for b, s in zip(bb, ss):
                if s >= conf:
                    pts.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, s, f))
    P = np.array(pts)
    centres, counts = [], []
    for x, y, s, f in P[np.argsort(-P[:, 2])]:
        for i, (cx, cy) in enumerate(centres):
            if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
                counts[i] += 1
                break
        else:
            centres.append((x, y))
            counts.append(1)
    order = np.argsort(counts)[::-1]
    return dict(n_detections=int(len(P)), n_frames=int(len(set(P[:, 3].astype(int)))), n_locations=len(counts),
                counts=[int(counts[i]) for i in order], centres=[[float(c) for c in centres[i]] for i in order])


loc = {lab: cluster(REV / f"sweep/{lab}.npz") for lab in ("ct5000@1280_z1", "ct5000@640_z1", "ct5000@640_z3.7")}
(OUT / "T_ct_detection_locations.json").write_text(json.dumps(loc, indent=2))

# ------------------------------------------------------------------ 8. object-scale manipulation
sw = []
for k, v in st["sweep"].items():
    lab, z = k.split("|z")
    sw.append(dict(series=lab, zoom=float(z), frames=v["n_frames"], tiles_per_frame=v["tiles_per_frame"],
                   bee_side_rel_tile=v["bee_rel_side_in_tile"], bee_px_input=v["bee_px_in_input"],
                   AP50=v["ap50"], AP50_lo=v["ap50_lo"], AP50_hi=v["ap50_hi"],
                   n_pred_ge025=v["n_pred_op"], precision_ge025=v["precision_op"], recall_ge025=v["recall_op"],
                   median_pred_side_rel_frame=v["median_side_rel_op"],
                   gt_best_iou_median=v["best_iou_median"], gt_best_iou_max=v["best_iou_max"],
                   frac_gt_iou_ge05=v["frac_gt_best_iou_ge05"]))
save(pd.DataFrame(sw).sort_values(["series", "zoom"]), "S_scale_sweep")

# ------------------------------------------------------------------ 9. CLIP appearance distance
E = dict(np.load(ORIG / "gap_fill_results__1_/clip_embeddings_fixed.npz"))
E = {k: v / np.linalg.norm(v, axis=1, keepdims=True) for k, v in E.items()}


def frechet(a, b):
    A = (a - a.mean(0)) / np.sqrt(len(a) - 1)
    Bm = (b - b.mean(0)) / np.sqrt(len(b) - 1)
    tr = np.linalg.svd(A @ Bm.T, compute_uv=False).sum()
    return float(((a.mean(0) - b.mean(0)) ** 2).sum() + (A * A).sum() + (Bm * Bm).sum() - 2 * tr)


fd = []
for a, b in (("flower333", "camera_trap"), ("camera_trap", "GBIF"), ("flower333", "GBIF")):
    boot = [frechet(E[a][rng.integers(0, len(E[a]), len(E[a]))], E[b][rng.integers(0, len(E[b]), len(E[b]))])
            for _ in range(1000)]
    fd.append(dict(pair=f"{a}|{b}", n_a=len(E[a]), n_b=len(E[b]), frechet=frechet(E[a], E[b]),
                   ci_lo=float(np.percentile(boot, 2.5)), ci_hi=float(np.percentile(boot, 97.5))))
for a in E:
    half = []
    for _ in range(500):
        p = rng.permutation(len(E[a]))
        h = len(p) // 2
        half.append(frechet(E[a][p[:h]], E[a][p[h:2 * h]]))
    fd.append(dict(pair=f"{a}|{a} split-half", n_a=len(E[a]) // 2, n_b=len(E[a]) // 2, frechet=float(np.median(half)),
                   ci_lo=float(np.percentile(half, 2.5)), ci_hi=float(np.percentile(half, 97.5))))
save(pd.DataFrame(fd), "T_clip_frechet")

# ------------------------------------------------------------------ 10. corpora
pb = pd.read_csv(REV / "partitions/phonebee_folds.csv")
corp = dict(phonebee_frames=len(pb), phonebee_boxes=int(pb.n_boxes.sum()), phonebee_frames_without_bee=int((pb.n_boxes == 0).sum()),
            phonebee_boxes_per_fold={int(k): int(v) for k, v in pb.groupby("fold").n_boxes.sum().items()},
            camera_trap=ct_state["dataset"] | dict(boxes_per_image=ct_state["dataset"]["kept_boxes"] / ct_state["dataset"]["n_images"]),
            gbif_reference_original=gap["gbif_stats"])
(OUT / "T_corpora.json").write_text(json.dumps(corp, indent=2, default=str))
print("tables written to", OUT)
