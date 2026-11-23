"""PhoneBee transfer study: supplementary experiments.

Purpose
    0. Record the exact data partitions (PhoneBee folds, camera-trap and GBIF
       subsets) and test whether the 300-image GBIF reference set used for
       in-domain evaluation overlaps the GBIF training/validation subsets.
    1. Draw a GBIF reference set disjoint from every GBIF subset, pseudo-label it
       with Grounding DINO (identical settings), evaluate all detectors on it,
       and export sheets for manual checking of the pseudo-labels and of the
       identity of annotated PhoneBee insects.
    2. Re-run the 21 PhoneBee fine-tuning runs (7 initialisations x 3 folds)
       for a fixed 60 epochs without early stopping, and evaluate the final
       weights on the held-out fold (no checkpoint selection on test data).
    3. Object-scale manipulation: sliced (tiled) inference on PhoneBee frames
       at a range of zoom factors, so that bees occupy the object scale on
       which each detector was trained, with block-bootstrap intervals.
    4. In-domain reference analyses with the cross-validated COCO-only fold
       models (confidence, predicted box size, example frames, random crops).

Inputs (Kaggle datasets attached to the notebook)
    PhoneBee frames (Roboflow YOLO export with data.yaml), camera-trap images
    with YOLO labels (Apis mellifera = class 1), GBIF Apis mellifera images,
    and the six auxiliary pre-trained weights (gbif500/2000/5000_pretrained.pt,
    ct500/2000/5000_pretrained.pt). COCO weights yolo26n.pt are downloaded if
    not attached.

Outputs (/kaggle/working)
    revision/                  all tables, arrays, sheets and run logs
    revision_results.zip       everything in revision/ except model weights
    revision_fold_weights.zip  final weights of the 21 fine-tuned fold models

Run
    Kaggle notebook with GPU T4 and internet on, after the setup cell that
    installs ultralytics 8.4.108, transformers 4.51.3 and tokenizers 0.21.0.
    Commit with Save & Run All. The script checkpoints after every unit of
    work and resumes from revision/state.json if a previous output is attached.
"""
import json
import os
import random
import shutil
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

CFG = dict(
    seed=42,
    run_refit=True,               # phase 2, about 4.5 h on a T4
    run_scale_sweep=True,         # phase 3, about 1 h
    # explicit paths override auto-detection when set
    flower_root=None, ct_root=None, gbif_root=None,
    exts={".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"},
    # protocol of the original study
    k_folds=3, ft_epochs=60, ft_imgsz=1280, ft_batch=16, aux_imgsz=640,
    gbif_val_frac=0.15, n_sweep=(500, 2000, 5000), gbif_pool_size=98731,
    apis_class=1, test_block=100, test_every=5, ct_n_images=6619,
    n_gbif_ref=300,
    gd=dict(model="IDEA-Research/grounding-dino-base", max_side=1333, prompt="a bee.",
            box=0.35, text=0.25, nms=0.5, min_af=0.001, max_af=0.98),
    # new elements
    gbif_ref_seed=2027, n_gbif_check=100, n_identity_check=150,
    zooms_full=(1, 1.5, 2, 2.5, 3, 3.7, 5, 7, 10),
    zooms_short=(1, 2, 3.7, 7),
    zooms_ct1280=(1, 1.5, 1.85, 2.5, 3.7, 5),
    zooms_infold=(1, 1.5, 2, 3, 5),
    zooms_gbif_high=(14, 24), gbif_high_frames=40,
    overlap=0.25, overlap_high=0.30, overlap_extreme=0.60,
    pred_conf=0.001, nms_iou=0.6, op_conf=0.25, max_det_frame=300, save_conf=0.05,
    boot=1000, block=10, n_random_crops=24,
    wd="/kaggle/working/revision",
)
WD = Path(CFG["wd"])
STATE = WD / "state.json"
DEV = 0 if torch.cuda.is_available() else "cpu"
HALF = torch.cuda.is_available()


# ------------------------------------------------------------------ utilities
def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=1, default=str))
    with zipfile.ZipFile("/kaggle/working/revision_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in WD.rglob("*"):
            if f.is_file() and f.suffix.lower() != ".pt" and "datasets" not in f.parts:
                z.write(f, f.relative_to(WD.parent))


def zip_weights():
    fw = WD / "fold_weights"
    if fw.exists():
        with zipfile.ZipFile("/kaggle/working/revision_fold_weights.zip", "w", zipfile.ZIP_STORED) as z:
            for f in sorted(fw.glob("*.pt")):
                z.write(f, f.name)


def resume_from_input():
    if STATE.exists():
        return
    for s in Path("/kaggle/input").rglob("revision/state.json"):
        shutil.copytree(s.parent, WD, dirs_exist_ok=True, symlinks=True,
                        ignore=shutil.ignore_patterns("datasets", "val_runs"))
        log(f"resumed from {s.parent}")
        return


def imgs_in(root):
    return sorted(p for p in Path(root).rglob("*")
                  if p.is_file() and p.suffix.lower() in CFG["exts"])


def dataset_root(p):
    parts = Path(p).parts
    if len(parts) > 5 and parts[3] == "datasets":
        return Path(*parts[:6])
    return Path(*parts[:4])


def find_flower():
    if CFG["flower_root"]:
        return Path(CFG["flower_root"])
    for p in Path("/kaggle/input").rglob("*"):
        if p.is_dir() and "flower333" in p.name.lower() and (p / "data.yaml").exists() and (p / "train").exists():
            return p
    return None


def find_ct():
    if CFG["ct_root"]:
        return Path(CFG["ct_root"])
    best, best_d = None, None
    for p in Path("/kaggle/input").rglob("*"):
        if not (p.is_dir() and "apis" in p.name.lower()):
            continue
        ims = imgs_in(p)
        n_txt = sum(1 for q in ims if q.with_suffix(".txt").exists())
        if len(ims) < 1000 or n_txt < 0.9 * len(ims):
            continue
        d = abs(len(ims) - CFG["ct_n_images"])
        if best is None or d < best_d:
            best, best_d = p, d
    return best


def find_gbif():
    """Folder with the most images among folders named gbif/honeybee (as in the
    original reference-set script); weights and run folders are excluded."""
    if CFG["gbif_root"]:
        return Path(CFG["gbif_root"])
    best, best_n = None, 0
    for p in Path("/kaggle/input").iterdir():
        for q in [p] + [x for x in p.rglob("*") if x.is_dir()]:
            nm = q.name.lower()
            if "gbif" not in nm and "honeybee" not in nm:
                continue
            if any(t in str(q).lower() for t in ("result", "pretrain", "runs", "weights", "revision")):
                continue
            n = sum(1 for f in q.glob("*") if f.suffix.lower() in CFG["exts"])
            n += sum(1 for f in q.glob("*/*") if f.suffix.lower() in CFG["exts"])
            if n > best_n:
                best, best_n = q, n
    return best if best_n >= 1000 else None


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_weights():
    """Pre-trained weights of the study; copies listed in SHA256SUMS.txt are preferred."""
    sums = sorted(Path("/kaggle/input").rglob("SHA256SUMS.txt"))
    pref = sums[0].parent if sums else None
    expected = {}
    if sums:
        for ln in sums[0].read_text().splitlines():
            parts = ln.split()
            if len(parts) >= 2:
                expected[parts[1].lstrip("*")] = parts[0]
    w, check = {}, {}
    for tag in ("gbif500", "gbif2000", "gbif5000", "ct500", "ct2000", "ct5000"):
        name = f"{tag}_pretrained.pt"
        hits = sorted(Path("/kaggle/input").rglob(name), key=lambda q: (q.parent != pref, str(q)))
        w[tag] = hits[0] if hits else None
        if w[tag] is not None:
            digest = sha256(w[tag])
            check[tag] = dict(path=str(w[tag]), sha256=digest,
                              matches_listed=(expected.get(name) == digest) if expected else None)
    hits = sorted(Path("/kaggle/input").rglob("yolo26n.pt"))
    w["coco"] = hits[0] if hits else "yolo26n.pt"
    return w, check


def flower_pairs(root):
    """All annotated frames sorted by file name (frame index), as in the study."""
    pr = []
    for sp in ("train", "valid", "val", "test"):
        si, sl = Path(root) / sp / "images", Path(root) / sp / "labels"
        if si.exists():
            for img in sorted(si.iterdir()):
                if img.is_file() and img.suffix.lower() in CFG["exts"]:
                    lb = sl / f"{img.stem}.txt"
                    pr.append((img, lb.read_text() if lb.exists() else ""))
    pr.sort(key=lambda x: x[0].name)
    return pr


def ct_pairs(root):
    out = []
    for img in imgs_in(root):
        lb, lines = img.with_suffix(".txt"), []
        if lb.exists():
            for ln in lb.read_text(errors="ignore").strip().splitlines():
                f = ln.split()
                if len(f) >= 5 and int(float(f[0])) == CFG["apis_class"]:
                    cx, cy, w, h = (min(max(float(v), 0.0), 1.0) for v in f[1:5])
                    if w > 0 and h > 0:
                        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        out.append((img, "\n".join(lines)))
    return out


def ct_test_indices(n):
    blocks = [list(range(i, min(i + CFG["test_block"], n))) for i in range(0, n, CFG["test_block"])]
    return [j for b in blocks[CFG["test_every"] - 1::CFG["test_every"]] for j in b]


def link(pairs, idxs, d, split):
    idir, ldir = d / "images" / split, d / "labels" / split
    for x in (idir, ldir):
        if x.exists():
            shutil.rmtree(x)
        x.mkdir(parents=True)
    for j in idxs:
        img, txt = pairs[j]
        dst = idir / f"{j:06d}{Path(img).suffix.lower()}"
        try:
            os.symlink(Path(img).resolve(), dst)
        except OSError:
            shutil.copy2(img, dst)
        (ldir / f"{j:06d}.txt").write_text(txt)


def make_yolo_set(d, pairs, val_idx, train_idx):
    link(pairs, train_idx, d, "train")
    link(pairs, val_idx, d, "val")
    (d / "data.yaml").write_text(f"path: {d.resolve()}\ntrain: images/train\nval: images/val\n"
                                 f"test: images/val\nnames:\n  0: bee\n")
    return d / "data.yaml"


def metrics_of(r):
    P, R = float(r.box.mp), float(r.box.mr)
    return dict(precision=P, recall=R, f1=(2 * P * R / (P + R) if P + R else 0.0),
                mAP50=float(r.box.map50), mAP50_95=float(r.box.map))


def evaluate(weights, yml, imgsz):
    m = YOLO(str(weights))
    r = m.val(data=str(yml), split="val", imgsz=imgsz, conf=0.001, iou=0.6, batch=16,
              device=DEV, verbose=False, plots=False, project=str(WD / "val_runs"),
              name="v", exist_ok=True)
    out = metrics_of(r)
    del m
    torch.cuda.empty_cache()
    return out


def yolo_to_xyxy(txt, W, H):
    b = []
    for ln in txt.strip().splitlines():
        f = ln.split()
        if len(f) >= 5:
            cx, cy, w, h = (float(v) for v in f[1:5])
            b.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    return np.array(b, dtype=np.float32).reshape(-1, 4)


# --------------------------------------------------------- sliced inference
def tiles_for(W, H, z, ov):
    if z <= 1:
        return [(0, 0, W, H)]
    tw, th = W / z, H / z

    def starts(L, t):
        s = list(np.arange(0.0, L - t + 1e-6, t * (1 - ov)))
        if L - t - s[-1] > 0.5:
            s.append(L - t)
        return s
    return [(int(round(x)), int(round(y)), int(round(x + tw)), int(round(y + th)))
            for y in starts(H, th) for x in starts(W, tw)]


def sliced_predict(model, img, z, imgsz, ov):
    batch = 64 if imgsz <= 640 else 16
    H, W = img.shape[:2]
    tl = tiles_for(W, H, z, ov)
    boxes, scores = [], []
    for i in range(0, len(tl), batch):
        chunk = tl[i:i + batch]
        crops = [np.ascontiguousarray(img[y1:y2, x1:x2]) for x1, y1, x2, y2 in chunk]
        res = model.predict(crops, imgsz=imgsz, conf=CFG["pred_conf"], iou=CFG["nms_iou"],
                            max_det=300, device=DEV, half=HALF, verbose=False)
        for (x1, y1, _, _), r in zip(chunk, res):
            if len(r.boxes):
                b = r.boxes.xyxy.float().cpu().numpy()
                b[:, [0, 2]] += x1
                b[:, [1, 3]] += y1
                boxes.append(b)
                scores.append(r.boxes.conf.float().cpu().numpy())
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), len(tl)
    b, s = np.concatenate(boxes), np.concatenate(scores)
    if len(tl) > 1:
        keep = torchvision.ops.nms(torch.from_numpy(b), torch.from_numpy(s), CFG["nms_iou"]).numpy()
        b, s = b[keep], s[keep]
    order = np.argsort(-s)[:CFG["max_det_frame"]]
    return b[order], s[order], len(tl)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def match_frame(pb, ps, gb, thr=0.5):
    """Greedy matching in descending confidence within one frame (COCO rule).
    Returns TP flags for predictions and the best IoU of each ground-truth box
    with any prediction at or above the operating confidence."""
    tp = np.zeros(len(pb), bool)
    if len(gb):
        iou = iou_matrix(pb, gb)
        used = np.zeros(len(gb), bool)
        for i in np.argsort(-ps):
            if iou.shape[1] == 0:
                break
            cand = np.where(~used & (iou[i] >= thr))[0]
            if len(cand):
                j = cand[np.argmax(iou[i, cand])]
                used[j] = True
                tp[i] = True
        op = ps >= CFG["op_conf"]
        best = iou[op].max(0) if op.any() else np.zeros(len(gb), np.float32)
    else:
        best = np.zeros(0, np.float32)
    return tp, best


def ap_from(conf, tp, n_gt):
    """Average precision with 101-point interpolation (Ultralytics/COCO style)."""
    if n_gt == 0:
        return float("nan")
    if len(conf) == 0:
        return 0.0
    o = np.argsort(-conf)
    tpc = np.cumsum(tp[o])
    fpc = np.cumsum(~tp[o])
    rec = tpc / n_gt
    prec = tpc / np.maximum(tpc + fpc, 1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(np.interp(x, mrec, mpre), x))


def summarise(frames, dets, frame_ids):
    """frames: list of dicts (gt boxes, W, H). dets: frame id -> (boxes, scores).
    Returns summary metrics with block-bootstrap interval for AP50."""
    per = {}
    for f in frame_ids:
        pb, ps = dets[f]
        tp, best = match_frame(pb, ps, frames[f]["gt"])
        per[f] = (ps, tp, len(frames[f]["gt"]), best, pb)
    conf = np.concatenate([per[f][0] for f in frame_ids]) if frame_ids else np.zeros(0)
    tp = np.concatenate([per[f][1] for f in frame_ids]) if frame_ids else np.zeros(0, bool)
    n_gt = sum(per[f][2] for f in frame_ids)
    best = np.concatenate([per[f][3] for f in frame_ids])
    sides = [np.sqrt((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])) / np.sqrt(frames[f]["W"] * frames[f]["H"])
             for f in frame_ids for b in [per[f][4][per[f][0] >= CFG["op_conf"]]]]
    sides = np.concatenate(sides) if sides else np.zeros(0)
    op = conf >= CFG["op_conf"]
    out = dict(n_frames=len(frame_ids), n_gt=int(n_gt), ap50=ap_from(conf, tp, n_gt),
               n_pred_op=int(op.sum()), tp_op=int(tp[op].sum()),
               precision_op=float(tp[op].mean()) if op.any() else float("nan"),
               recall_op=float(tp[op].sum() / max(n_gt, 1)),
               median_side_rel_op=float(np.median(sides)) if len(sides) else float("nan"),
               best_iou_median=float(np.median(best)) if len(best) else float("nan"),
               best_iou_max=float(best.max()) if len(best) else float("nan"),
               frac_gt_best_iou_ge05=float(np.mean(best >= 0.5)) if len(best) else float("nan"))
    # block bootstrap over runs of consecutive frames (frame_ids are time-ordered)
    blocks = [frame_ids[i:i + CFG["block"]] for i in range(0, len(frame_ids), CFG["block"])]
    rng = np.random.default_rng(CFG["seed"])
    bs = []
    for _ in range(CFG["boot"]):
        pick = [f for k in rng.integers(0, len(blocks), len(blocks)) for f in blocks[k]]
        c = np.concatenate([per[f][0] for f in pick])
        t = np.concatenate([per[f][1] for f in pick])
        bs.append(ap_from(c, t, sum(per[f][2] for f in pick)))
    out["ap50_lo"], out["ap50_hi"] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    return out


# --------------------------------------------------------------- Grounding DINO
class GDINO:
    def __init__(self):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        g = CFG["gd"]
        self.proc = AutoProcessor.from_pretrained(g["model"])
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(g["model"]).to(
            "cuda" if HALF else "cpu").eval()

    def label(self, path):
        from PIL import Image
        g = CFG["gd"]
        im = Image.open(path).convert("RGB")
        W0, H0 = im.size
        sc = min(1.0, g["max_side"] / max(W0, H0))
        x = im.resize((max(1, round(W0 * sc)), max(1, round(H0 * sc)))) if sc < 1 else im
        dev = "cuda" if HALF else "cpu"
        inp = self.proc(images=x, text=g["prompt"], return_tensors="pt").to(dev)
        with torch.no_grad():
            if HALF:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    o = self.model(**inp)
            else:
                o = self.model(**inp)
        try:
            r = self.proc.post_process_grounded_object_detection(
                o, inp["input_ids"], threshold=g["box"], text_threshold=g["text"],
                target_sizes=[x.size[::-1]])[0]
        except TypeError:
            r = self.proc.post_process_grounded_object_detection(
                o, inp["input_ids"], box_threshold=g["box"], text_threshold=g["text"],
                target_sizes=[x.size[::-1]])[0]
        b, s = r["boxes"].float().cpu(), r["scores"].float().cpu()
        if sc < 1 and b.numel():
            b /= sc
        lines = []
        if b.numel():
            b[:, 0::2].clamp_(0, W0)
            b[:, 1::2].clamp_(0, H0)
            b = b[torchvision.ops.nms(b, s, g["nms"])]
            for x1, y1, x2, y2 in b.tolist():
                bw, bh = x2 - x1, y2 - y1
                if bw <= 1 or bh <= 1:
                    continue
                if not (g["min_af"] <= (bw * bh) / (W0 * H0) <= g["max_af"]):
                    continue
                lines.append("0 %.6f %.6f %.6f %.6f" % ((x1 + x2) / 2 / W0, (y1 + y2) / 2 / H0, bw / W0, bh / H0))
        return "\n".join(lines), (W0, H0)


def draw_sheet_pdf(items, pdf_path, per_page, title_fn):
    """items: list of (rgb_image, list_of_boxes_xyxy, label). Boxes are numbered."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    rows = per_page
    with PdfPages(pdf_path) as pdf:
        for i in range(0, len(items), rows):
            fig, axes = plt.subplots(rows, 1, figsize=(8.27, 11.69))
            axes = np.atleast_1d(axes)
            for ax, it in zip(axes, items[i:i + rows]):
                img, boxes, lab = it
                ax.imshow(img)
                for k, (x1, y1, x2, y2) in enumerate(boxes, 1):
                    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.2, ec="red"))
                    ax.text(x1, max(y1 - 3, 0), str(k), color="red", fontsize=8, va="bottom",
                            bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
                ax.set_title(title_fn(lab), fontsize=9)
                ax.axis("off")
            for ax in axes[len(items[i:i + rows]):]:
                ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)


# =================================================================== RUN
if __name__ == "__main__":
    t0 = time.time()
    WD.mkdir(parents=True, exist_ok=True)
    resume_from_input()
    st = load_state()
    random.seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    FL, CT, GB = find_flower(), find_ct(), find_gbif()
    W, WCHECK = find_weights()
    log(f"PhoneBee: {FL}\ncamera-trap: {CT}\nGBIF: {GB}\nweights: { {k: str(v) for k, v in W.items()} }")
    assert FL and CT and GB, "PhoneBee, camera-trap and GBIF datasets must be attached"
    assert all(W[k] for k in W), "all six *_pretrained.pt weights must be attached"
    st["weights"] = WCHECK
    assert all(v["matches_listed"] is not False for v in WCHECK.values()), \
        "a pre-trained weight file differs from SHA256SUMS.txt"

    fl = flower_pairs(FL)
    folds = [list(a) for a in np.array_split(np.arange(len(fl)), CFG["k_folds"])]
    fold_of = {j: k for k, ix in enumerate(folds) for j in ix}
    from PIL import Image
    frames = []
    for j, (img, txt) in enumerate(fl):
        with Image.open(img) as im:
            Wd, H = im.size
        frames.append(dict(path=str(img), name=img.name, fold=fold_of[j], W=Wd, H=H,
                           gt=yolo_to_xyxy(txt, Wd, H)))
    ctp = ct_pairs(CT)
    cti = ct_test_indices(len(ctp))

    # ------------------------------------------------ phase 0: partitions
    if "phase0" not in st:
        log("phase 0: partitions and reference-set audit")
        part = WD / "partitions"
        part.mkdir(exist_ok=True)
        with open(part / "phonebee_folds.csv", "w") as f:
            f.write("frame_file,fold,n_boxes,width,height\n")
            for fr in frames:
                f.write(f"{fr['name']},{fr['fold']},{len(fr['gt'])},{fr['W']},{fr['H']}\n")
        (part / "phonebee_gt_boxes_xyxy.json").write_text(
            json.dumps({fr["name"]: fr["gt"].round(2).tolist() for fr in frames}))
        # camera-trap subsets (nested: first N of one seeded permutation)
        order = list(range(len(ctp)))
        tset = set(cti)
        pool = [j for j in order if j not in tset]
        sh = pool[:]
        random.Random(CFG["seed"]).shuffle(sh)
        with open(part / "camera_trap_membership.csv", "w") as f:
            f.write("image_file,heldout_test," + ",".join(f"N{n}" for n in CFG["n_sweep"]) + "\n")
            role = {}
            for n in CFG["n_sweep"]:
                nv = max(1, int(n * CFG["gbif_val_frac"]))
                for r, j in enumerate(sh[:n]):
                    role.setdefault(j, {})[n] = "val" if r < nv else "train"
            for j, (img, _) in enumerate(ctp):
                f.write(f"{img.relative_to(CT)},{int(j in tset)}," +
                        ",".join(role.get(j, {}).get(n, "") for n in CFG["n_sweep"]) + "\n")
        # GBIF subsets exactly as drawn for pre-training
        groot = dataset_root(GB)
        gpool = sorted(p for p in groot.rglob("*") if p.is_file() and p.suffix.lower() in CFG["exts"]
                       and "flower" not in str(p).lower())
        s = gpool[:]
        random.Random(CFG["seed"]).shuffle(s)
        ref_pool = sorted(imgs_in(GB))
        random.Random(CFG["seed"]).shuffle(ref_pool)
        ref_old = ref_pool[:CFG["n_gbif_ref"]]
        audit = dict(gbif_root=str(groot), reference_folder=str(GB), pool_size=len(gpool),
                     pool_size_expected=CFG["gbif_pool_size"],
                     reference_folder_size=len(ref_pool),
                     pool_identical_to_reference_folder=set(map(str, ref_pool)) == set(map(str, gpool)))
        rset = set(map(str, ref_old))
        for n in CFG["n_sweep"]:
            nv = max(1, int(n * CFG["gbif_val_frac"]))
            tr, va = set(map(str, s[nv:n])), set(map(str, s[:nv]))
            audit[f"N{n}_reference_in_train"] = len(rset & tr)
            audit[f"N{n}_reference_in_val"] = len(rset & va)
        with open(part / "gbif_membership.csv", "w") as f:
            f.write("image_file,permutation_rank," + ",".join(f"N{n}" for n in CFG["n_sweep"]) + ",reference_original\n")
            for r, p in enumerate(s[:max(CFG["n_sweep"])]):
                roles = []
                for n in CFG["n_sweep"]:
                    nv = max(1, int(n * CFG["gbif_val_frac"]))
                    roles.append("" if r >= n else ("val" if r < nv else "train"))
                f.write(f"{p.relative_to(groot)},{r}," + ",".join(roles) + f",{int(str(p) in rset)}\n")
        # disjoint reference set: outside every pre-training subset and the original reference set
        used = set(map(str, s[:max(CFG["n_sweep"])])) | rset
        cand = [p for p in s[max(CFG["n_sweep"]):] if str(p) not in used]
        ref_new = random.Random(CFG["gbif_ref_seed"]).sample(cand, CFG["n_gbif_ref"])
        (part / "gbif_reference_disjoint.txt").write_text("\n".join(str(p.relative_to(groot)) for p in ref_new))
        (part / "gbif_reference_original.txt").write_text("\n".join(str(Path(p).relative_to(groot)) for p in ref_old))
        st["phase0"] = audit
        st["ref_new"] = [str(p) for p in ref_new]
        save_state(st)
        log(json.dumps(audit, indent=1))

    # ------------------------------------------------ phase 1: GBIF reference
    gset = WD / "datasets" / "gbif_disjoint"
    if "gbif_labels" not in st:
        log("phase 1: Grounding DINO labels for the disjoint GBIF reference set")
        gd = GDINO()
        pairs, sides, hit = [], [], 0
        for i, p in enumerate(st["ref_new"]):
            txt, (W0, H0) = gd.label(p)
            pairs.append((p, txt))
            hit += bool(txt)
            for ln in txt.splitlines():
                f = ln.split()
                sides.append(float(np.sqrt(float(f[3]) * float(f[4]))))
        st["gbif_labels"] = dict(pairs=pairs, n_images=len(pairs), n_with_box=hit,
                                 n_boxes=len(sides), median_box_side=float(np.median(sides)) if sides else None)
        del gd
        torch.cuda.empty_cache()
        save_state(st)
    pairs = [tuple(x) for x in st["gbif_labels"]["pairs"]]
    y_gb = make_yolo_set(gset, pairs, list(range(len(pairs))), list(range(8)))
    yct = make_yolo_set(WD / "datasets" / "ct_test", ctp, cti, cti[:8])

    if "gbif_check_sheet" not in st:
        rng = random.Random(7)
        idx = sorted(rng.sample(range(len(pairs)), CFG["n_gbif_check"]))
        items, rows = [], ["check_id,image_file,gd_boxes,correct_boxes,false_boxes,missed_bees,notes"]
        for c, i in enumerate(idx, 1):
            p, txt = pairs[i]
            im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            H0, W0 = im.shape[:2]
            b = yolo_to_xyxy(txt, W0, H0)
            items.append((im, b.tolist(), (c, Path(p).name, len(b))))
            rows.append(f"{c},{Path(p).name},{len(b)},,,,")
        draw_sheet_pdf(items, WD / "gbif_pseudolabel_check.pdf", 2,
                       lambda lab: f"Check {lab[0]}  |  {lab[1]}  |  {lab[2]} Grounding DINO box(es)")
        (WD / "gbif_pseudolabel_check.csv").write_text("\n".join(rows) + "\n")
        st["gbif_check_sheet"] = idx
        save_state(st)

    if "identity_sheet" not in st:
        allb = [(j, k) for j, fr in enumerate(frames) for k in range(len(fr["gt"]))]
        pick = sorted(random.Random(11).sample(allb, min(CFG["n_identity_check"], len(allb))))
        items, rows = [], ["crop_id,frame_file,box_index,box_side_px,honeybee,other_bee,fly,other,unsure,notes"]
        tiles = []
        for c, (j, k) in enumerate(pick, 1):
            fr = frames[j]
            im = cv2.cvtColor(cv2.imread(fr["path"]), cv2.COLOR_BGR2RGB)
            x1, y1, x2, y2 = fr["gt"][k]
            cx, cy, half = (x1 + x2) / 2, (y1 + y2) / 2, 48
            xa, ya = int(max(cx - half, 0)), int(max(cy - half, 0))
            crop = np.ascontiguousarray(im[ya:ya + 2 * half, xa:xa + 2 * half])
            crop = cv2.resize(crop, (crop.shape[1] * 4, crop.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
            bb = [[(x1 - xa) * 4, (y1 - ya) * 4, (x2 - xa) * 4, (y2 - ya) * 4]]
            tiles.append((crop, bb, (c, fr["name"], k)))
            side = float(np.sqrt((x2 - x1) * (y2 - y1)))
            rows.append(f"{c},{fr['name']},{k},{side:.1f},,,,,,")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(WD / "phonebee_identity_check.pdf") as pdf:
            for i in range(0, len(tiles), 20):
                fig, axes = plt.subplots(5, 4, figsize=(8.27, 11.69))
                for ax, (crop, bb, lab) in zip(axes.ravel(), tiles[i:i + 20]):
                    ax.imshow(crop)
                    x1, y1, x2, y2 = bb[0]
                    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=0.8, ec="red", ls="--"))
                    ax.set_title(f"{lab[0]}", fontsize=9)
                    ax.axis("off")
                for ax in axes.ravel()[len(tiles[i:i + 20]):]:
                    ax.axis("off")
                fig.tight_layout()
                pdf.savefig(fig, dpi=150)
                plt.close(fig)
        (WD / "phonebee_identity_check.csv").write_text("\n".join(rows) + "\n")
        st["identity_sheet"] = [list(x) for x in pick]
        save_state(st)

    st.setdefault("transfer", {})
    for tag in ("gbif500", "gbif2000", "gbif5000", "ct500", "ct2000", "ct5000"):
        for sz in (640, 1280):
            key = f"{tag}->GBIFdisjoint@{sz}"
            if key not in st["transfer"]:
                st["transfer"][key] = evaluate(W[tag], y_gb, sz)
                log(f"{key} mAP50 {st['transfer'][key]['mAP50']:.3f}")
                save_state(st)

    # ------------------------------------------------ phase 3: scale sweep
    SAVE_DETS = ("ct5000@640", "ct5000@1280", "gbif5000@640", "infold@1280", "deploy@1280")

    def record(key, label, tag, imgsz, z, dets, frame_ids, n_tiles):
        out = summarise(frames, dets, frame_ids)
        out.update(model=tag, imgsz=imgsz, zoom=z, tiles_per_frame=n_tiles,
                   bee_rel_side_in_tile=0.01358 * z, bee_px_in_input=19.6 * z * imgsz / 1920)
        st["sweep"][key] = out
        if label in SAVE_DETS:
            arr = {}
            for f in frame_ids:
                b, s = dets[f]
                k = s >= CFG["save_conf"]
                arr[f"f{f}_b"], arr[f"f{f}_s"] = b[k].astype(np.float16), s[k].astype(np.float16)
            np.savez_compressed(WD / "sweep" / f"{label}_z{z}.npz", **arr)
        log(f"sweep {key}: AP50 {out['ap50']:.3f} [{out['ap50_lo']:.3f}, {out['ap50_hi']:.3f}] "
            f"P {out['precision_op']:.3f} R {out['recall_op']:.3f} side {out['median_side_rel_op']:.4f}")
        save_state(st)

    def run_sweep(tag, weights, imgsz, zooms, frame_ids, label):
        """One detector applied to the given frames at each zoom factor."""
        st.setdefault("sweep", {})
        m = None
        for z in zooms:
            key = f"{label}|z{z}"
            if key in st["sweep"]:
                continue
            if m is None:
                m = YOLO(str(weights))
            ov = CFG["overlap_extreme"] if z >= 14 else CFG["overlap_high"] if z >= 10 else CFG["overlap"]
            dets, nt = {}, 1
            for f in frame_ids:
                b, s, nt = sliced_predict(m, cv2.imread(frames[f]["path"]), z, imgsz, ov)
                dets[f] = (b, s)
            record(key, label, tag, imgsz, z, dets, frame_ids, nt)
        del m
        torch.cuda.empty_cache()

    def run_sweep_infold(weight_of_fold, imgsz, zooms, label):
        """In-domain reference: each frame is predicted by the fold model that did
        not see it in training; results are pooled over the three folds."""
        st.setdefault("sweep", {})
        for z in zooms:
            key = f"{label}|z{z}"
            if key in st["sweep"]:
                continue
            ov = CFG["overlap_extreme"] if z >= 14 else CFG["overlap_high"] if z >= 10 else CFG["overlap"]
            dets, nt = {}, 1
            for k in range(CFG["k_folds"]):
                m = YOLO(str(weight_of_fold(k)))
                for f in [j for j in all_ids if frames[j]["fold"] == k]:
                    b, s, nt = sliced_predict(m, cv2.imread(frames[f]["path"]), z, imgsz, ov)
                    dets[f] = (b, s)
                del m
                torch.cuda.empty_cache()
            record(key, label, "coco_infold", imgsz, z, dets, all_ids, nt)

    all_ids = list(range(len(frames)))
    if CFG["run_scale_sweep"]:
        (WD / "sweep").mkdir(exist_ok=True)
        run_sweep("ct5000", W["ct5000"], 640, CFG["zooms_full"], all_ids, "ct5000@640")
        run_sweep("gbif5000", W["gbif5000"], 640, CFG["zooms_full"], all_ids, "gbif5000@640")
        run_sweep("ct5000", W["ct5000"], 1280, CFG["zooms_ct1280"], all_ids, "ct5000@1280")
        for tag in ("ct500", "ct2000", "gbif500", "gbif2000"):
            run_sweep(tag, W[tag], 640, CFG["zooms_short"], all_ids, f"{tag}@640")
        hi_ids = sorted(random.Random(5).sample(all_ids, CFG["gbif_high_frames"]))
        run_sweep("gbif5000", W["gbif5000"], 640, (7, 10) + tuple(CFG["zooms_gbif_high"]), hi_ids,
                  "gbif5000@640_subset")

    # ------------------------------------------------ phase 2: fixed-epoch fine-tuning
    if CFG["run_refit"]:
        st.setdefault("refit", {})
        (WD / "fold_weights").mkdir(exist_ok=True)
        inits = [("coco", W["coco"])] + [(t, W[t]) for t in
                                         ("gbif500", "gbif2000", "gbif5000", "ct500", "ct2000", "ct5000")]
        for tag, init in inits:
            for k in range(CFG["k_folds"]):
                key = f"{tag}_fold{k}"
                if key in st["refit"]:
                    continue
                d = WD / "datasets" / f"ft_{key}"
                tr = [j for f, ix in enumerate(folds) if f != k for j in ix]
                yml = make_yolo_set(d, fl, folds[k], tr)
                m = YOLO(str(init))
                m.train(data=str(yml), epochs=CFG["ft_epochs"], imgsz=CFG["ft_imgsz"],
                        batch=CFG["ft_batch"], patience=0, seed=CFG["seed"], deterministic=True,
                        workers=4, device=DEV, verbose=False, plots=False,
                        project=str(WD / "ft_runs"), name=key, exist_ok=True)
                last = Path(m.trainer.last)
                del m
                torch.cuda.empty_cache()
                dst = WD / "fold_weights" / f"{key}.pt"
                shutil.copy2(last, dst)
                st["refit"][key] = dict(final=evaluate(dst, yml, CFG["ft_imgsz"]),
                                        final_at640=evaluate(dst, yml, 640))
                shutil.rmtree(d, ignore_errors=True)
                for wpt in (WD / "ft_runs" / key / "weights").glob("*.pt"):
                    wpt.unlink()
                log(f"refit {key}: mAP50 {st['refit'][key]['final']['mAP50']:.4f} "
                    f"R {st['refit'][key]['final']['recall']:.4f}")
                save_state(st)
                zip_weights()

    # ------------------------------------------------ phase 4: in-domain reference
    fw = WD / "fold_weights"
    have_folds = all((fw / f"coco_fold{k}.pt").exists() for k in range(CFG["k_folds"]))
    deploy = sorted(Path("/kaggle/input").rglob("flower333*deploy*.pt"))
    if have_folds:
        for k in range(CFG["k_folds"]):
            for tgt, yml in (("CTtest", yct), ("GBIFdisjoint", y_gb)):
                for sz in (640, 1280):
                    key = f"coco_fold{k}->{tgt}@{sz}"
                    if key not in st["transfer"]:
                        st["transfer"][key] = evaluate(fw / f"coco_fold{k}.pt", yml, sz)
                        save_state(st)
        if CFG["run_scale_sweep"]:
            run_sweep_infold(lambda k: fw / f"coco_fold{k}.pt", 1280, CFG["zooms_infold"], "infold@1280")
    elif deploy and CFG["run_scale_sweep"]:
        run_sweep("deploy", deploy[0], 1280, CFG["zooms_infold"], all_ids, "deploy@1280")

    if have_folds and "conf_infold" not in st:
        mx, nd = np.zeros(len(frames), np.float32), np.zeros(len(frames), np.float32)
        for k in range(CFG["k_folds"]):
            m = YOLO(str(fw / f"coco_fold{k}.pt"))
            for j in [j for j in all_ids if frames[j]["fold"] == k]:
                r = m.predict(frames[j]["path"], imgsz=1280, conf=0.001, iou=0.6, device=DEV,
                              half=HALF, verbose=False)[0]
                c = r.boxes.conf.float().cpu().numpy()
                mx[j], nd[j] = (c.max() if len(c) else 0.0), len(c)
            del m
        np.savez_compressed(WD / "conf_infold.npz", max_conf=mx, n_dets=nd)
        st["conf_infold"] = dict(median_max=float(np.median(mx)), frac_ge_025=float(np.mean(mx >= 0.25)),
                                 frac_ge_050=float(np.mean(mx >= 0.5)), mean_dets=float(nd.mean()))
        save_state(st)

    def crop_box(img, b, pad_frac=0.15, size=320):
        """Square-agnostic crop of a box with 15 % padding, resized to size x size."""
        H, Wd = img.shape[:2]
        x1, y1, x2, y2 = b
        pad = pad_frac * max(x2 - x1, y2 - y1)
        xa, ya = int(max(0, x1 - pad)), int(max(0, y1 - pad))
        xb, yb = int(min(Wd, x2 + pad)), int(min(H, y2 + pad))
        return (cv2.resize(np.ascontiguousarray(img[ya:yb, xa:xb]), (size, size),
                           interpolation=cv2.INTER_LANCZOS4), [xa, ya, xb, yb])

    if have_folds and "examples" not in st:
        ex = WD / "examples"
        ex.mkdir(exist_ok=True)
        names = ["frame_037436", "frame_006492", "frame_001543", "frame_016854", "frame_015328", "frame_013842"]
        ids = [j for j, fr in enumerate(frames) if any(fr["name"].startswith(n) for n in names)]
        out = {}
        for j in ids:
            cv2.imwrite(str(ex / f"{frames[j]['name'][:12]}.jpg"), cv2.imread(frames[j]["path"]),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            rec = dict(gt=frames[j]["gt"].tolist(), fold=frames[j]["fold"])
            for tag, wt in (("gbif5000", W["gbif5000"]), ("ct5000", W["ct5000"]),
                            ("infold", fw / f"coco_fold{frames[j]['fold']}.pt")):
                m = YOLO(str(wt))
                r = m.predict(frames[j]["path"], imgsz=1280, conf=CFG["op_conf"], iou=0.6, device=DEV,
                              half=HALF, verbose=False)[0]
                rec[tag] = dict(boxes=r.boxes.xyxy.float().cpu().numpy().tolist(),
                                conf=r.boxes.conf.float().cpu().numpy().tolist())
                del m
            out[frames[j]["name"]] = rec
        (ex / "example_predictions.json").write_text(json.dumps(out))
        m = YOLO(str(W["ct5000"]))
        confident = []
        for j in all_ids:
            r = m.predict(frames[j]["path"], imgsz=1280, conf=0.5, iou=0.6, device=DEV, half=HALF, verbose=False)[0]
            for b, c in zip(r.boxes.xyxy.float().cpu().numpy(), r.boxes.conf.float().cpu().numpy()):
                confident.append((j, b.tolist(), float(c)))
        del m
        rng = random.Random(13)
        pick = rng.sample(confident, min(CFG["n_random_crops"], len(confident)))
        gtall = [(j, b.tolist(), None) for j in all_ids for b in frames[j]["gt"]]
        pick_gt = rng.sample(gtall, CFG["n_random_crops"])
        meta = dict(n_confident_ct5000=len(confident), ct5000=[], gt=[])
        for tagc, lst in (("ct5000", pick), ("gt", pick_gt)):
            for i, (j, b, c) in enumerate(lst):
                cr, win = crop_box(cv2.imread(frames[j]["path"]), b)
                cv2.imwrite(str(ex / f"crop_{tagc}_{i:02d}.jpg"), cr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                side = float(np.sqrt((b[2] - b[0]) * (b[3] - b[1])) / np.sqrt(frames[j]["W"] * frames[j]["H"]))
                meta[tagc].append(dict(frame=frames[j]["name"], box=b, window=win, conf=c, side_rel=side))
        (ex / "crops_meta.json").write_text(json.dumps(meta))
        st["examples"] = dict(n_frames=len(ids), n_confident_ct5000=len(confident))
        save_state(st)

    shutil.rmtree(WD / "datasets", ignore_errors=True)
    shutil.rmtree(WD / "val_runs", ignore_errors=True)
    save_state(st)
    zip_weights()
    log(f"finished in {(time.time() - t0) / 3600:.2f} h")
