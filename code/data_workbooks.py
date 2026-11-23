"""Write the supporting data workbooks (Data S1 and Data S2), each with a README sheet.

Inputs : TABLE_DIR   outputs of analysis_revision.py and compute_cost.py
         PART_DIR    partition files (phonebee_folds.csv, phonebee_gt_boxes_xyxy.json, camera_trap_membership.csv,
                     gbif_membership.csv, gbif_reference_disjoint.txt)
         STATE       state.json of the revision run (pseudo-labels of the GBIF reference set)
         GBIF_DIR    outputs of gbif_provenance.py
Outputs: Data_S1_results.xlsx and Data_S2_partitions.xlsx in OUT_DIR
Usage  : python make_data_workbooks.py TABLE_DIR PART_DIR STATE GBIF_DIR OUT_DIR
"""
import json
import sys
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font

TAB, PART, STATE, GBIF, OUT = (Path(a) for a in sys.argv[1:6])
OUT.mkdir(parents=True, exist_ok=True)

CONFIG = {"coco": "COCO-only control", "gbif500": "GBIF-500", "gbif2000": "GBIF-2000", "gbif5000": "GBIF-5000",
          "ct500": "Camera-trap-500", "ct2000": "Camera-trap-2000", "ct5000": "Camera-trap-5000"}


def cfg(df, col="config"):
    df = df.copy()
    df[col] = df[col].map(lambda c: CONFIG.get(c, c))
    return df


def fold1(df, col="fold"):
    df = df.copy()
    df[col] = df[col].astype(int) + 1
    return df


def write(path, title, description, sheets, readme_rows):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        head = pd.DataFrame({"A": [title, description, "", "Variables"]})
        head.to_excel(xw, sheet_name="README", index=False, header=False)
        pd.DataFrame(readme_rows, columns=["Sheet", "Column", "Description", "Unit"]).to_excel(
            xw, sheet_name="README", index=False, startrow=5)
        for name, df in sheets:
            df.to_excel(xw, sheet_name=name, index=False)
        wb = xw.book
        ws = wb["README"]
        ws["A1"].font = Font(bold=True, size=12)
        ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells("A2:D2")
        ws.row_dimensions[2].height = 120
        ws["A4"].font = Font(bold=True)
        for col, width in zip("ABCD", (24, 30, 95, 18)):
            ws.column_dimensions[col].width = width
        for row in ws.iter_rows(min_row=7):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        for name, df in sheets:
            s = wb[name]
            for i, column in enumerate(df.columns):
                width = min(60, max(10, len(str(column)) + 2, int(df[column].astype(str).str.len().quantile(0.9)) + 2))
                s.column_dimensions[s.cell(row=1, column=i + 1).column_letter].width = width
            s.freeze_panes = "A2"


# ------------------------------------------------------------------ Data S1: results
s1 = []
t1 = pd.read_csv(TAB / "T1_within_domain.csv")
t1 = t1[~t1.test_set.str.startswith("original GBIF")].copy()
t1["test_set"] = t1["test_set"].replace({
    "camera-trap held-out test (1300 images)": "camera-trap held-out test set (1,300 images)",
    "GBIF reference set disjoint from all subsets (300 images)": "GBIF reference set (300 images not in any subset)"})
s1.append(("within_domain", t1))
fm = pd.read_csv(TAB / "S1_fold_metrics.csv")
fm["protocol"] = fm["protocol"].replace({"fixed 60 epochs, final weights": "run 2: fixed 60 epochs, final weights",
                                         "early stopping and best epoch on the held-out fold":
                                             "run 1: early stopping, best epoch on held-out fold"})
s1.append(("finetune_fold_metrics", fold1(cfg(fm))))
s1.append(("finetune_summary", cfg(pd.read_csv(TAB / "T2_finetune_summary.csv"))))
s1.append(("fold_estimates", fold1(cfg(pd.read_csv(TAB / "S1b_fold_mAP_estimates.csv")))))
con = pd.read_csv(TAB / "T3_contrasts.csv").rename(columns={"fold0": "fold1_delta", "fold1": "fold2_delta",
                                                             "fold2": "fold3_delta"})
s1.append(("contrasts", con))
var = json.loads((TAB / "T3b_variation.json").read_text())
var_rows = [(k, v) for k, v in var.items() if not isinstance(v, dict)]
var_rows += [(f"fold_mean_fold{int(k) + 1}", v) for k, v in var["fold_means"].items()]
s1.append(("variation", pd.DataFrame(var_rows, columns=["quantity", "value"])))
s1.append(("convergence", cfg(pd.read_csv(TAB / "T_convergence.csv")).rename(
    columns={"fold0": "fold1", "fold1": "fold2", "fold2": "fold3"})))
s1.append(("compute_cost", cfg(pd.read_csv(TAB / "T_compute_cost.csv"))))
c1 = fold1(cfg(pd.read_csv(TAB / "S_curves_run1.csv")))
c1.insert(0, "run", "run 1")
c2 = fold1(cfg(pd.read_csv(TAB / "S_curves_run2.csv")))
c2.insert(0, "run", "run 2")
s1.append(("learning_curves", pd.concat([c1, c2], ignore_index=True)))
tm = pd.read_csv(TAB / "T4_transfer_matrix.csv")
tm["model"] = tm["model"].map(lambda c: CONFIG.get(c, c))
s1.append(("transfer_matrix", tm))
s1.append(("confidence", pd.read_csv(TAB / "T_confidence.csv")))
s1.append(("object_size", pd.read_csv(TAB / "T5_object_size.csv")))
loc = json.loads((TAB / "T_ct_detection_locations.json").read_text())
loc_rows = []
for series, d in loc.items():
    for rank, (n, (x, y)) in enumerate(zip(d["counts"], d["centres"]), start=1):
        loc_rows.append(dict(series=series, cluster=rank, detections=n, centre_x_px=x, centre_y_px=y,
                             total_detections=d["n_detections"], frames_with_detections=d["n_frames"]))
s1.append(("ct_detection_locations", pd.DataFrame(loc_rows)))
s1.append(("scale_manipulation", pd.read_csv(TAB / "S_scale_sweep.csv")))
s1.append(("clip_frechet", pd.read_csv(TAB / "T_clip_frechet.csv")))

readme1 = [
    ("within_domain", "source, N, imgsz", "Auxiliary source, number of pre-training images, inference size", "-, images, px"),
    ("within_domain", "test_set", "Held-out test data of the source domain", "-"),
    ("within_domain", "precision, recall, f1", "At the maximum-F1 operating point (confidence floor 0.001, NMS IoU 0.6)",
     "proportion"),
    ("within_domain", "mAP50, mAP50_95", "Mean average precision at IoU 0.5 and averaged over IoU 0.5-0.95",
     "proportion"),
    ("finetune_fold_metrics", "config, fold, protocol",
     "Initialization, held-out PhoneBee fold (1-3 in recording order), training run and weights evaluated", "-"),
    ("finetune_fold_metrics", "mAP50_at640", "mAP@0.5 of the same weights at 640 px inference", "proportion"),
    ("finetune_summary", "*_mean, *_sd", "Mean and SD (n - 1) over the three folds, fixed 60-epoch protocol",
     "proportion"),
    ("finetune_summary", "delta_vs_control, delta_min, delta_max",
     "Mean, smallest and largest within-fold difference in mAP@0.5 from the COCO-only control", "proportion"),
    ("finetune_summary", "folds_above_control", "Number of folds in which mAP@0.5 exceeded the control", "count"),
    ("fold_estimates", "estimate", "Fixed 60 epochs (final weights), after 30 epochs (runs 2 and 1), or best "
     "held-out epoch (run 1)", "-"),
    ("contrasts", "mean_delta, fold1_delta..fold3_delta",
     "Mean paired difference in mAP@0.5 over all pairs and within each fold", "proportion"),
    ("contrasts", "folds_positive, pairs_positive", "Folds and fold-level pairs with a positive difference", "count"),
    ("variation", "sd_fold_means, sd_config_means, residual_sd",
     "SD of fold means, of configuration means, and residual configuration x fold SD (fixed protocol)", "proportion"),
    ("variation", "run_to_run_epoch30_*", "Differences in mAP@0.5 after 30 epochs between two training runs with "
     "identical settings; sd_per_run = SD of differences / sqrt(2)", "proportion"),
    ("convergence", "epochs_to_080_mean_curve, fold1..fold3",
     "First epoch at which mAP@0.5 on the held-out fold reached 0.80 (mean curve and each fold)", "epochs"),
    ("compute_cost", "pretrain_units, saved_units",
     "Training cost in image-epochs at 640 x 640 px (one 1,280 px image = 4 units)", "image-epochs"),
    ("compute_cost", "target_datasets_to_break_even", "Pre-training cost divided by the fine-tuning cost saved",
     "count"),
    ("learning_curves", "run, config, fold, epoch, mAP50", "mAP@0.5 on the held-out fold after each epoch",
     "proportion"),
    ("transfer_matrix", "Target@size", "mAP@0.5 without adaptation on each target at 640 or 1,280 px", "proportion"),
    ("confidence", "median_max_conf, max_max_conf",
     "Median and maximum over frames of the highest detection confidence per frame", "confidence score"),
    ("confidence", "frac_frames_ge_025, frac_frames_ge_050", "Fraction of frames with a detection >= 0.25 / 0.5",
     "proportion"),
    ("object_size", "median_side, p5, p95, min, max",
     "Box side sqrt(w*h) relative to sqrt(W*H) of the frame", "proportion of frame side"),
    ("object_size", "ratio_to_truth, area_ratio, ratio_to_training_scale",
     "Median predicted side divided by the annotated median, squared, and divided by the training-corpus median",
     "ratio"),
    ("object_size", "frac_within_truth_p5_p95, frac_within_1p5x_truth_median",
     "Share of predictions inside the 5th-95th percentile range of annotated sizes, and within a factor of 1.5 of "
     "the annotated median", "proportion"),
    ("ct_detection_locations", "cluster, detections, centre_x_px, centre_y_px",
     "Clusters (radius 40 px) of confident (>= 0.25) camera-trap-5000 detections; series gives inference size and "
     "zoom", "count, px"),
    ("scale_manipulation", "series, zoom, frames, tiles_per_frame",
     "Model and inference size, magnification factor, number of frames, tiles per frame", "-"),
    ("scale_manipulation", "bee_side_rel_tile, bee_px_input",
     "Median bee side relative to the input tile, and in input pixels", "proportion, px"),
    ("scale_manipulation", "AP50, AP50_lo, AP50_hi",
     "AP@0.5 over all frames with 95% block-bootstrap interval (blocks of 10 frames, 1,000 resamples)", "proportion"),
    ("scale_manipulation", "n_pred_ge025, precision_ge025, recall_ge025",
     "Number of predictions, precision and recall at confidence >= 0.25", "count, proportion"),
    ("scale_manipulation", "median_pred_side_rel_frame, gt_best_iou_*, frac_gt_iou_ge05",
     "Median predicted box side (frame-relative), largest IoU of each annotated bee with any prediction, and share "
     "of bees with IoU >= 0.5", "proportion"),
    ("clip_frechet", "pair, n_a, n_b, frechet, ci_lo, ci_hi",
     "Fréchet distance between CLIP ViT-B/32 embeddings of two corpora (or two random halves) with 95% bootstrap "
     "interval", "distance"),
]
write(OUT / "Data_S1_results.xlsx", "Data S1. Results of the detection, transfer, diagnostic and scale-manipulation "
      "analyses",
      "Results of all analyses reported in the article and in Appendix S1. PhoneBee folds are numbered 1-3 in "
      "recording order. Configurations: COCO-only control; GBIF-N and Camera-trap-N = YOLO26n pre-trained on N "
      "auxiliary images (N = 500, 2,000, 5,000) before fine-tuning. Run 2 is the primary fixed 60-epoch protocol; run "
      "1 used early stopping monitored on the held-out fold. GBIF scores measure agreement with Grounding DINO "
      "pseudo-labels.", s1, readme1)

# ------------------------------------------------------------------ Data S2: partitions and labels
pb = pd.read_csv(PART / "phonebee_folds.csv")
pb["fold"] = pb["fold"].astype(int) + 1
boxes = json.loads((PART / "phonebee_gt_boxes_xyxy.json").read_text())
box_rows = []
for frame, bl in boxes.items():
    for j, b in enumerate(bl, start=1):
        box_rows.append(dict(frame_file=frame, box=j, x1=b[0], y1=b[1], x2=b[2], y2=b[3]))
gm = pd.read_csv(PART / "gbif_membership.csv")
gm.insert(0, "occurrence_key", gm["image_file"].map(lambda p: Path(p).stem))
gm = gm.drop(columns=["image_file", "reference_original"])
meta = pd.read_csv(GBIF / "gbif_occurrence_metadata.csv", dtype={"occurrence_key": str})
keep_cols = ["occurrence_key", "dataset_key", "record_licence", "media_licence", "creator", "event_date", "country"]
gm = gm.merge(meta[keep_cols], on="occurrence_key", how="left")
state = json.loads(STATE.read_text())
ref_rows = []
for path, lab in state["gbif_labels"]["pairs"]:
    key = Path(path).stem
    lines = [ln.split() for ln in lab.strip().splitlines() if ln.strip()]
    if not lines:
        ref_rows.append(dict(occurrence_key=key, box=0, x_center=None, y_center=None, width=None, height=None))
    for j, p in enumerate(lines, start=1):
        ref_rows.append(dict(occurrence_key=key, box=j, x_center=float(p[1]), y_center=float(p[2]),
                             width=float(p[3]), height=float(p[4])))
ref = pd.DataFrame(ref_rows).merge(meta[keep_cols], on="occurrence_key", how="left")
ct = pd.read_csv(PART / "camera_trap_membership.csv")
counts = pd.read_csv(GBIF / "gbif_derived_dataset_counts.csv")
s2 = [("phonebee_folds", pb), ("phonebee_boxes", pd.DataFrame(box_rows)), ("gbif_auxiliary_subsets", gm),
      ("gbif_reference_set", ref), ("camera_trap_subsets", ct), ("gbif_source_datasets", counts)]
readme2 = [
    ("phonebee_folds", "frame_file, fold", "Annotated video frame (frame number and time stamp in the name) and "
     "cross-validation fold (1-3 in recording order)", "-"),
    ("phonebee_folds", "n_boxes, width, height", "Number of annotated bees, frame width and height", "count, px"),
    ("phonebee_boxes", "x1, y1, x2, y2", "Annotated bee box corners in the native 1920 x 1080 frame", "px"),
    ("gbif_auxiliary_subsets", "occurrence_key", "GBIF occurrence key of the image (https://www.gbif.org/occurrence/"
     "<key>)", "-"),
    ("gbif_auxiliary_subsets", "permutation_rank", "Position in the seeded random order from which the nested subsets "
     "were taken", "rank"),
    ("gbif_auxiliary_subsets", "N500, N2000, N5000", "Use of the image in each auxiliary subset (train or val; empty "
     "= not included)", "-"),
    ("gbif_auxiliary_subsets", "dataset_key ... country", "GBIF metadata of the record: source dataset, record and "
     "media licences, creator, observation date, country", "-"),
    ("gbif_reference_set", "box, x_center, y_center, width, height", "Grounding DINO pseudo-label boxes in normalized "
     "YOLO format (box 0 = image without a box)", "proportion of image width/height"),
    ("camera_trap_subsets", "image_file, heldout_test", "Image of the camera-trap dataset (Bjerge et al.) and "
     "membership of the held-out test set (1 = test)", "-"),
    ("camera_trap_subsets", "N500, N2000, N5000", "Use of the image in each auxiliary subset (train or val)", "-"),
    ("gbif_source_datasets", "dataset_key, count", "Number of GBIF records used per source dataset (auxiliary subsets "
     "and reference set)", "count"),
]
write(OUT / "Data_S2_partitions.xlsx", "Data S2. Data partitions, annotations and image identifiers",
      "PhoneBee fold assignments and annotated boxes; GBIF occurrence keys and metadata of all images in the auxiliary "
      "subsets and the GBIF reference set, with the Grounding DINO pseudo-labels of the reference set; camera-trap "
      "file names of all subsets and of the held-out test set. GBIF images are not redistributed; they can be "
      "retrieved through the occurrence keys under the licences listed.", s2, readme2)
print("written", OUT)
