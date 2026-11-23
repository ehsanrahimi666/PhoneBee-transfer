# ============================================================================
# GBIF auxiliary source (matched 3-fold protocol)
#   [1] label N GBIF images with Grounding DINO -> [2] pre-train YOLO26n at 640 px
#   -> [3] 3-fold fine-tuning on the PhoneBee frames (flower333) at 1,280 px
# State and the output archive are saved after the pre-training stage and after
# each fold, so an interrupted run resumes where it stopped.
# ============================================================================
import os, json, shutil, random, time, zipfile
from pathlib import Path
from statistics import mean, pstdev
import numpy as np
from PIL import Image
import torch, torchvision
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import YOLO

CFG = dict(
    N=5000,
    gbif_root="/kaggle/input/datasets", exclude="flower",
    flower_dir="/kaggle/input/datasets/ehsanrahimi666/flower333",
    coco="yolo26n.pt",
    gbif_val_frac=0.15,
    pre_epochs=30, pre_imgsz=640, pre_batch=32,
    k_folds=3, ft_epochs=60, ft_imgsz=1280, ft_batch=16, patience=15,
    workers=4, device=0 if torch.cuda.is_available() else "cpu", seed=42,
    wd="/kaggle/working/gbif5000_run",
    exts={".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"},
    gdino="IDEA-Research/grounding-dino-base", max_side=1333, prompt="a bee.",
    box_thr=0.35, text_thr=0.25, nms_iou=0.5, min_af=0.001, max_af=0.98, fp16=True,
)
KNOWN = {
    "C2_coco":     dict(n_gbif=0,    mAP50=[0.896, 0.020], recall=[0.873, 0.029]),
    "C1_gbif500":  dict(n_gbif=500,  mAP50=[0.864, 0.028], recall=[0.826, 0.011]),
    "C1_gbif2000": dict(n_gbif=2000, mAP50=[0.881, 0.021], recall=[0.855, 0.030]),
}
WD = Path(CFG["wd"]); WD.mkdir(parents=True, exist_ok=True)
N = CFG["N"]
STATE = WD / "state.json"
ZIP = Path(f"/kaggle/working/gbif{N}_result.zip")
PRE_W = WD / f"gbif{N}_pretrained.pt"


# ------------------------- checkpoint helpers -------------------------
def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            pass
    return {"n_gbif": N, "label_yield": None, "folds": {}}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=2))
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(STATE, "state.json")
        if PRE_W.exists():
            z.write(PRE_W, PRE_W.name)
        rj = WD / f"C1_gbif{N}_result.json"
        if rj.exists():
            z.write(rj, rj.name)
        for csv in WD.rglob("results.csv"):
            z.write(csv, csv.relative_to(WD.parent))
    print(f"    [checkpoint saved] {ZIP} ({ZIP.stat().st_size/1e6:.1f} MB)")


# ------------------------- Grounding DINO -------------------------
class GDINO:
    def __init__(s, mid, dev, fp16):
        s.dev, s.fp16 = dev, fp16 and dev == "cuda"
        s.p = AutoProcessor.from_pretrained(mid)
        s.m = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(dev).eval()

    @torch.no_grad()
    def detect(s, im, prompt, bt, tt, ms):
        W0, H0 = im.size
        sc = min(1.0, ms / max(W0, H0))
        x = im.resize((max(1, round(W0*sc)), max(1, round(H0*sc)))) if sc < 1 else im
        inp = s.p(images=x, text=prompt, return_tensors="pt").to(s.dev)
        if s.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                o = s.m(**inp)
        else:
            o = s.m(**inp)
        ts = [x.size[::-1]]
        try:
            r = s.p.post_process_grounded_object_detection(
                o, inp["input_ids"], threshold=bt, text_threshold=tt, target_sizes=ts)[0]
        except TypeError:
            r = s.p.post_process_grounded_object_detection(
                o, inp["input_ids"], box_threshold=bt, text_threshold=tt, target_sizes=ts)[0]
        b = r["boxes"].float().cpu()
        if sc < 1 and b.numel():
            b /= sc
        return b, r["scores"].float().cpu()


def yolo_lines(b, sc, W, H, c):
    if b.numel() == 0:
        return []
    b[:, 0::2].clamp_(0, W); b[:, 1::2].clamp_(0, H)
    b = b[torchvision.ops.nms(b, sc, c["nms_iou"])]
    A = float(W*H); out = []
    for x1, y1, x2, y2 in b.tolist():
        bw, bh = x2-x1, y2-y1
        if bw <= 1 or bh <= 1:
            continue
        f = (bw*bh)/A
        if not (c["min_af"] <= f <= c["max_af"]):
            continue
        cx, cy, nw, nh = (min(max(v, 0.), 1.) for v in
                          ((x1+x2)/2/W, (y1+y2)/2/H, bw/W, bh/H))
        out.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return out


def label_split(paths, split, gd, c, ds):
    idir, ldir = ds/"images"/split, ds/"labels"/split
    for d in (idir, ldir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    n = hit = bad = 0
    for i, src in enumerate(paths):
        try:
            im = Image.open(src).convert("RGB")
        except Exception:
            bad += 1; continue
        W, H = im.size
        try:
            b, sc = gd.detect(im, c["prompt"], c["box_thr"], c["text_thr"], c["max_side"])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); bad += 1; continue
        lines = yolo_lines(b, sc, W, H, c)
        stem = f"{split}_{i:06d}"
        dst = idir/f"{stem}{src.suffix.lower()}"
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)
        (ldir/f"{stem}.txt").write_text("\n".join(lines))
        n += 1; hit += bool(lines)
        if (i+1) % 250 == 0:
            print(f"    labeled {i+1}/{len(paths)}  (hits {hit})")
    return n, hit, bad


# ------------------------- data helpers -------------------------
def link_flower(pairs, idxs, idir, ldir):
    for d in (idir, ldir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    for j in idxs:
        img, lb = pairs[j]
        dst = idir/img.name
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(img.resolve(), dst)
        except OSError:
            shutil.copy2(img, dst)
        (ldir/f"{img.stem}.txt").write_text(lb.read_text() if lb else "")


def gbif_list(c):
    ex = c["exclude"].lower()
    return [q for q in Path(c["gbif_root"]).rglob("*")
            if q.is_file() and q.suffix.lower() in c["exts"] and ex not in str(q).lower()]


def flower_pairs(root, exts):
    pr = []
    for sp in ("train", "valid", "val", "test"):
        si, sl = root/sp/"images", root/sp/"labels"
        if si.exists():
            for img in sorted(si.iterdir()):
                if img.is_file() and img.suffix.lower() in exts:
                    lb = sl/f"{img.stem}.txt"
                    pr.append((img, lb if lb.exists() else None))
    pr.sort(key=lambda x: x[0].name)
    return pr


# ============================== RUN ==============================
t0 = time.perf_counter()
st = load_state()
pool = gbif_list(CFG)
flowers = flower_pairs(Path(CFG["flower_dir"]), CFG["exts"])
folds = [list(a) for a in np.array_split(list(range(len(flowers))), CFG["k_folds"])]
print(f"GBIF pool (flower excluded): {len(pool)} | flower333: {len(flowers)} | folds {[len(f) for f in folds]}")
if st["folds"]:
    print(f"RESUMING: folds already done -> {sorted(st['folds'].keys())}")

ds = WD/"gbif_ds"

# ---------- Stages 1+2: skipped entirely if the pretrained weight survives ----------
if PRE_W.exists():
    print(f"\n[1-2/3] SKIPPED — pretrained weight found: {PRE_W}")
else:
    rng = random.Random(CFG["seed"]); s = sorted(pool); rng.shuffle(s); s = s[:N]
    nv = max(1, int(N*CFG["gbif_val_frac"]))
    print(f"\n[1/3] Labeling {N} GBIF images with Grounding DINO  (train {N-nv} / val {nv}) ...")
    gd = GDINO(CFG["gdino"], CFG["device"], CFG["fp16"])
    ntr, htr, _ = label_split(s[nv:], "train", gd, CFG, ds)
    nva, hva, _ = label_split(s[:nv], "val", gd, CFG, ds)
    (ds/"data.yaml").write_text(f"path: {ds.resolve()}\ntrain: images/train\nval: images/val\n"
                                f"test: images/val\nnames:\n  0: bee\n")
    yield_rate = (htr+hva)/max(ntr+nva, 1)
    print(f"  label yield: {htr+hva}/{ntr+nva} images had >=1 bee ({yield_rate:.1%})")
    st["label_yield"] = yield_rate
    del gd
    if CFG["device"] != "cpu":
        torch.cuda.empty_cache()

    print(f"\n[2/3] Pretraining YOLO26n on {N} GBIF imgs @ {CFG['pre_imgsz']} ...")
    mp = YOLO(CFG["coco"])
    mp.train(data=str(ds/"data.yaml"), epochs=CFG["pre_epochs"], imgsz=CFG["pre_imgsz"],
             batch=CFG["pre_batch"], workers=CFG["workers"], seed=CFG["seed"],
             device=CFG["device"], verbose=False, plots=False,
             project=str(WD/"pretrain"), name=f"gbif{N}", exist_ok=True)
    shutil.copy2(mp.trainer.best, PRE_W)
    del mp
    if CFG["device"] != "cpu":
        torch.cuda.empty_cache()
    save_state(st)

# ---------- Stage 3: per-fold, checkpointed ----------
print(f"\n[3/3] Fine-tuning on flower333, {CFG['k_folds']}-fold @ {CFG['ft_imgsz']} ...")
for k in range(CFG["k_folds"]):
    if str(k) in st["folds"]:
        r = st["folds"][str(k)]
        print(f"  fold{k}: already done -> mAP50 {r['mAP50']:.3f}  recall {r['recall']:.3f}")
        continue
    fd = WD/"ft"/f"fold{k}"
    tr = [j for f, ix in enumerate(folds) if f != k for j in ix]
    link_flower(flowers, tr, fd/"images/train", fd/"labels/train")
    link_flower(flowers, folds[k], fd/"images/val", fd/"labels/val")
    (fd/"data.yaml").write_text(f"path: {fd.resolve()}\ntrain: images/train\nval: images/val\n"
                                f"test: images/val\nnames:\n  0: bee\n")
    m = YOLO(str(PRE_W))
    m.train(data=str(fd/"data.yaml"), epochs=CFG["ft_epochs"], imgsz=CFG["ft_imgsz"],
            batch=CFG["ft_batch"], workers=CFG["workers"], patience=CFG["patience"],
            seed=CFG["seed"], device=CFG["device"], verbose=False, plots=False,
            project=str(WD/"ft_runs"), name=f"fold{k}", exist_ok=True)
    mv = m.val(data=str(fd/"data.yaml"), split="val", imgsz=CFG["ft_imgsz"], conf=0.001,
               iou=0.6, batch=CFG["ft_batch"], device=CFG["device"], verbose=False,
               project=str(WD/"ft_val"), name=f"fold{k}", exist_ok=True)
    P, R = float(mv.box.mp), float(mv.box.mr)
    st["folds"][str(k)] = dict(precision=P, recall=R, f1=(2*P*R/(P+R) if P+R else 0.),
                               mAP50=float(mv.box.map50), mAP50_95=float(mv.box.map))
    print(f"  fold{k}: mAP50 {st['folds'][str(k)]['mAP50']:.3f}  "
          f"recall {st['folds'][str(k)]['recall']:.3f}")
    del m; shutil.rmtree(fd/"images", ignore_errors=True)
    if CFG["device"] != "cpu":
        torch.cuda.empty_cache()
    save_state(st)

# ---------- aggregate ----------
rows = [st["folds"][str(k)] for k in range(CFG["k_folds"])]
ag = lambda key: [mean([r[key] for r in rows]), pstdev([r[key] for r in rows])]
res = {"n_gbif": N, "label_yield": st.get("label_yield"), "folds": rows,
       **{key: ag(key) for key in ("mAP50", "mAP50_95", "recall", "precision", "f1")}}
(WD/f"C1_gbif{N}_result.json").write_text(json.dumps(res, indent=2))
save_state(st)

allpts = dict(KNOWN); allpts[f"C1_gbif{N}"] = res
base = KNOWN["C2_coco"]["mAP50"][0]
print("\n" + "=" * 68)
print(f"RESULT  C1_gbif{N}:  mAP50 {res['mAP50'][0]:.3f}±{res['mAP50'][1]:.3f}  "
      f"recall {res['recall'][0]:.3f}±{res['recall'][1]:.3f}  F1 {res['f1'][0]:.3f}±{res['f1'][1]:.3f}")
print("=" * 68)
print("SCALING CURVE (baseline C2 mAP50 = %.3f):" % base)
for tag in sorted(allpts, key=lambda t: allpts[t]["n_gbif"]):
    mm = allpts[tag]
    print(f"  {tag:13s} N={mm['n_gbif']:>5}  mAP50 {mm['mAP50'][0]:.3f}±{mm['mAP50'][1]:.3f}  "
          f"recall {mm['recall'][0]:.3f}±{mm['recall'][1]:.3f}  Δ_vs_base {mm['mAP50'][0]-base:+.3f}")
print(f"\nruntime {(time.perf_counter()-t0)/60:.1f} min")
print(f"Output archive: {ZIP} ({ZIP.stat().st_size/1e6:.1f} MB)")
