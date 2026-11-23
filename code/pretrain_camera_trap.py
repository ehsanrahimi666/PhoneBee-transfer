# ============================================================================
# CAMERA-TRAP (Apis mellifera, class_id=1) vs GBIF as pretraining sources
#   Phase A: zero-shot eval of gbif500/2000/5000 (+flower333 deploy) on CT test
#   Phase B: train YOLO26n on CT at N=500/2000/5000 (COCO init) -> save weights
#   Phase C: fine-tune each CT model on flower333 (3-fold @1280) -> compare
# Protocol matched to the GBIF study (pre 30ep@640, ft 60ep@1280, 3 folds, seed 42).
# Progress is checkpointed after every stage; an interrupted run resumes where it stopped.
# ============================================================================
import os, re, json, shutil, random, time, zipfile
from pathlib import Path
from statistics import mean, pstdev
import numpy as np
import torch
from ultralytics import YOLO

CFG = dict(
    n_sweep=[500, 2000, 5000],
    val_frac=0.15,
    test_block=100, test_every=5,        # blocks of 100 consecutive imgs; every 5th = test
    pre_epochs=30, pre_imgsz=640, pre_batch=32,
    k_folds=3, ft_epochs=60, ft_imgsz=1280, ft_batch=16, patience=15,
    eval_imgsz=[640, 1280],
    workers=4, device=0 if torch.cuda.is_available() else "cpu", seed=42,
    coco="yolo26n.pt",
    wd="/kaggle/working/ct_study",
    exts={".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"},
    keep_class_ids={1},                  # class 1 = Apis mellifera
    run_phase_c=True,
)
KNOWN_FLOWER = {   # completed GBIF study, same protocol
    "C2_coco":     dict(n=0,    mAP50=[0.896, 0.020], recall=[0.873, 0.029]),
    "C1_gbif500":  dict(n=500,  mAP50=[0.864, 0.028], recall=[0.826, 0.011]),
    "C1_gbif2000": dict(n=2000, mAP50=[0.881, 0.021], recall=[0.855, 0.030]),
    "C1_gbif5000": dict(n=5000, mAP50=[0.889, 0.036], recall=[0.846, 0.013]),
}
WD = Path(CFG["wd"]); WD.mkdir(parents=True, exist_ok=True)
STATE = WD / "state.json"
ZIP = Path("/kaggle/working/ct_study_results.zip")
DEV = CFG["device"]


def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            pass
    return {"zero_shot": {}, "ct_pretrain": {}, "flower_ft": {}, "dataset": {}}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=2))
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(STATE, "state.json")
        for f in sorted(WD.glob("ct*_pretrained.pt")):
            z.write(f, f.name)
        for f in sorted(WD.glob("*.json")):
            z.write(f, f.name)
        for csv in WD.rglob("results.csv"):
            z.write(csv, csv.relative_to(WD.parent))
    print(f"    [checkpoint] {ZIP} ({ZIP.stat().st_size/1e6:.1f} MB)")


def find_gbif_weights():
    out = {}
    for p in Path("/kaggle/input").rglob("gbif*_pretrained.pt"):
        m = re.search(r"gbif(\d+)_pretrained", p.name)
        if m:
            out[int(m.group(1))] = p
    return dict(sorted(out.items()))


def find_deploy_weight():
    for p in Path("/kaggle/input").rglob("flower333*deploy*.pt"):
        return p
    return None


def find_dir(keyword, must_have=None):
    hits = [p for p in Path("/kaggle/input").rglob("*")
            if p.is_dir() and keyword.lower() in p.name.lower()]
    if must_have:
        for p in hits:
            if any((p / m).exists() for m in must_have):
                return p
    return hits[0] if hits else None


def collect_ct_samples(root, cfg):
    """Flat YOLO-txt layout: image.jpg + image.txt side by side."""
    imgs = sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in cfg["exts"])
    if not imgs:
        raise FileNotFoundError(f"No images under {root}")
    keep, hist = cfg["keep_class_ids"], {}
    out, kept_boxes, dropped = [], 0, 0
    for img in imgs:
        lb = img.with_suffix(".txt")
        lines = []
        if lb.exists():
            for ln in lb.read_text(errors="ignore").strip().splitlines():
                f = ln.split()
                if len(f) < 5:
                    continue
                c = int(float(f[0]))
                hist[c] = hist.get(c, 0) + 1
                if keep is not None and c not in keep:
                    dropped += 1
                    continue
                cx, cy, w, h = (min(max(float(v), 0.0), 1.0) for v in f[1:5])
                if w > 0 and h > 0:
                    lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    kept_boxes += 1
        out.append((img, lines))
    return out, hist, kept_boxes, dropped


def write_split(samples, idxs, idir, ldir):
    for d in (idir, ldir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    nb = 0
    for j in idxs:
        img, lines = samples[j]
        dst = idir / f"{j:06d}{img.suffix.lower()}"
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(img.resolve(), dst)
        except OSError:
            shutil.copy2(img, dst)
        (ldir / f"{j:06d}.txt").write_text("\n".join(lines))
        nb += len(lines)
    return nb


def write_yaml(d):
    (d / "data.yaml").write_text(
        f"path: {d.resolve()}\ntrain: images/train\nval: images/val\n"
        f"test: images/val\nnames:\n  0: bee\n")
    return d / "data.yaml"


def evaluate(weights, yml, imgsz, tag):
    m = YOLO(str(weights))
    r = m.val(data=str(yml), split="val", imgsz=imgsz, conf=0.001, iou=0.6,
              batch=16, device=DEV, verbose=False,
              project=str(WD / "eval"), name=tag, exist_ok=True)
    P, R = float(r.box.mp), float(r.box.mr)
    del m
    if DEV != "cpu":
        torch.cuda.empty_cache()
    return dict(precision=P, recall=R, f1=(2*P*R/(P+R) if P+R else 0.0),
                mAP50=float(r.box.map50), mAP50_95=float(r.box.map))


def flower_pairs(root, exts):
    pr = []
    for sp in ("train", "valid", "val", "test"):
        si, sl = root/sp/"images", root/sp/"labels"
        if si.exists():
            for img in sorted(si.iterdir()):
                if img.is_file() and img.suffix.lower() in exts:
                    lb = sl/f"{img.stem}.txt"
                    pr.append((img, lb.read_text() if lb.exists() else ""))
    pr.sort(key=lambda x: x[0].name)
    return pr


def link_flower(pairs, idxs, idir, ldir):
    for d in (idir, ldir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    for j in idxs:
        img, txt = pairs[j]
        dst = idir/img.name
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(img.resolve(), dst)
        except OSError:
            shutil.copy2(img, dst)
        (ldir/f"{img.stem}.txt").write_text(txt)


def agg(rows, k):
    v = [r[k] for r in rows]
    return [mean(v), pstdev(v) if len(v) > 1 else 0.0]


# ============================== RUN ==============================
t0 = time.perf_counter()
st = load_state()
gbif_w = find_gbif_weights()
deploy_w = find_deploy_weight()
ct_root = find_dir("apis")
fl_root = find_dir("flower333", must_have=["data.yaml", "train"])
print("GBIF weights :", {k: str(v) for k, v in gbif_w.items()})
print("deploy weight:", deploy_w)
print("CT root      :", ct_root)
print("flower333    :", fl_root)
assert ct_root and gbif_w, "Missing camera-trap dir or GBIF weights."

samples, hist, kept, dropped = collect_ct_samples(ct_root, CFG)
n_bg = sum(1 for _, l in samples if not l)
print(f"\nCT dataset: {len(samples)} images")
print(f"  class histogram (raw ids): {dict(sorted(hist.items()))}")
print(f"  keep_class_ids={CFG['keep_class_ids']} -> kept {kept} boxes, dropped {dropped}")
print(f"  images with >=1 Apis box: {len(samples)-n_bg} | background images: {n_bg}")
assert kept > 0, "No boxes kept -- check keep_class_ids."

order = list(range(len(samples)))
blocks = [order[i:i+CFG["test_block"]] for i in range(0, len(order), CFG["test_block"])]
test_idx = [j for b in blocks[CFG["test_every"]-1::CFG["test_every"]] for j in b]
tset = set(test_idx)
pool_idx = [j for j in order if j not in tset]
print(f"  split: pool {len(pool_idx)} | held-out test {len(test_idx)}")

ct_test = WD / "ct_test"
tb = write_split(samples, test_idx, ct_test/"images/val", ct_test/"labels/val")
write_split(samples, test_idx[:8], ct_test/"images/train", ct_test/"labels/train")
ct_test_yaml = write_yaml(ct_test)
print(f"  CT test boxes: {tb}")
st["dataset"] = dict(n_images=len(samples), kept_boxes=kept, dropped_boxes=dropped,
                     n_test=len(test_idx), test_boxes=tb, hist=hist)

# ---------------- PHASE A ----------------
print("\n########## PHASE A — zero-shot on camera-trap test set ##########")
zs = {f"gbif{n}": p for n, p in gbif_w.items()}
if deploy_w:
    zs["flower333_deploy"] = deploy_w
for tag, w in zs.items():
    for sz in CFG["eval_imgsz"]:
        key = f"{tag}@{sz}"
        if key in st["zero_shot"]:
            continue
        st["zero_shot"][key] = evaluate(w, ct_test_yaml, sz, f"zs_{tag}_{sz}")
        r = st["zero_shot"][key]
        print(f"  {key:26s} mAP50 {r['mAP50']:.3f}  R {r['recall']:.3f}  P {r['precision']:.3f}")
        save_state(st)

# ---------------- PHASE B ----------------
print("\n########## PHASE B — train YOLO26n on camera-trap data ##########")
rng = random.Random(CFG["seed"])
shuffled = pool_idx[:]; rng.shuffle(shuffled)
for N in CFG["n_sweep"]:
    tag = f"ct{N}"
    ptw = WD / f"{tag}_pretrained.pt"
    if tag in st["ct_pretrain"] and ptw.exists():
        print(f"  {tag}: done -> CT-test mAP50 {st['ct_pretrain'][tag]['ct_test']['mAP50']:.3f}")
        continue
    if N > len(shuffled):
        print(f"  {tag}: SKIPPED (pool has only {len(shuffled)})"); continue
    sel = shuffled[:N]
    nv = max(1, int(N * CFG["val_frac"]))
    dd = WD / f"ct_train_{N}"
    write_split(samples, sel[nv:], dd/"images/train", dd/"labels/train")
    write_split(samples, sel[:nv], dd/"images/val", dd/"labels/val")
    yml = write_yaml(dd)
    m = YOLO(CFG["coco"])
    m.train(data=str(yml), epochs=CFG["pre_epochs"], imgsz=CFG["pre_imgsz"],
            batch=CFG["pre_batch"], workers=CFG["workers"], seed=CFG["seed"],
            device=DEV, verbose=False, plots=False,
            project=str(WD/"ct_runs"), name=tag, exist_ok=True)
    shutil.copy2(m.trainer.best, ptw)
    del m
    if DEV != "cpu":
        torch.cuda.empty_cache()
    st["ct_pretrain"][tag] = {"n": N,
                              "ct_test": evaluate(ptw, ct_test_yaml, CFG["pre_imgsz"], f"ct_{N}_test")}
    r = st["ct_pretrain"][tag]["ct_test"]
    print(f"  {tag}: CT-test mAP50 {r['mAP50']:.3f}  R {r['recall']:.3f}")
    save_state(st)

# ---------------- PHASE C ----------------
if CFG["run_phase_c"] and fl_root:
    print("\n########## PHASE C — fine-tune CT models on flower333 (3-fold @1280) ##########")
    fpairs = flower_pairs(fl_root, CFG["exts"])
    ffolds = [list(a) for a in np.array_split(list(range(len(fpairs))), CFG["k_folds"])]
    print(f"flower333: {len(fpairs)} images | folds {[len(f) for f in ffolds]}")
    for N in CFG["n_sweep"]:
        tag = f"ct{N}"
        ptw = WD / f"{tag}_pretrained.pt"
        if not ptw.exists():
            continue
        st["flower_ft"].setdefault(tag, {})
        for k in range(CFG["k_folds"]):
            if str(k) in st["flower_ft"][tag]:
                continue
            fd = WD/"ft"/f"{tag}_fold{k}"
            tr = [j for f, ix in enumerate(ffolds) if f != k for j in ix]
            link_flower(fpairs, tr, fd/"images/train", fd/"labels/train")
            link_flower(fpairs, ffolds[k], fd/"images/val", fd/"labels/val")
            yml = write_yaml(fd)
            mm = YOLO(str(ptw))
            mm.train(data=str(yml), epochs=CFG["ft_epochs"], imgsz=CFG["ft_imgsz"],
                     batch=CFG["ft_batch"], workers=CFG["workers"], patience=CFG["patience"],
                     seed=CFG["seed"], device=DEV, verbose=False, plots=False,
                     project=str(WD/"ft_runs"), name=f"{tag}_fold{k}", exist_ok=True)
            mv = mm.val(data=str(yml), split="val", imgsz=CFG["ft_imgsz"], conf=0.001,
                        iou=0.6, batch=CFG["ft_batch"], device=DEV, verbose=False)
            P, R = float(mv.box.mp), float(mv.box.mr)
            st["flower_ft"][tag][str(k)] = dict(precision=P, recall=R,
                                                f1=(2*P*R/(P+R) if P+R else 0.0),
                                                mAP50=float(mv.box.map50),
                                                mAP50_95=float(mv.box.map))
            print(f"  {tag} fold{k}: mAP50 {mv.box.map50:.3f}  R {R:.3f}")
            del mm; shutil.rmtree(fd/"images", ignore_errors=True)
            if DEV != "cpu":
                torch.cuda.empty_cache()
            save_state(st)

# ---------------- TABLES ----------------
print("\n" + "=" * 78)
print("TABLE 1 — ZERO-SHOT on camera-trap test set (models never saw CT data)")
print("=" * 78)
print(f"{'model':22s} {'imgsz':>6} {'P':>7} {'R':>7} {'F1':>7} {'mAP50':>8} {'mAP50-95':>9}")
for k, r in st["zero_shot"].items():
    nm, sz = k.rsplit("@", 1)
    print(f"{nm:22s} {sz:>6} {r['precision']:>7.3f} {r['recall']:>7.3f} "
          f"{r['f1']:>7.3f} {r['mAP50']:>8.3f} {r['mAP50_95']:>9.3f}")

print("\n" + "=" * 78)
print("TABLE 2 — YOLO26n TRAINED on camera-trap data (evaluated on CT test)")
print("=" * 78)
print(f"{'model':12s} {'N':>6} {'P':>7} {'R':>7} {'F1':>7} {'mAP50':>8} {'mAP50-95':>9}")
for tag, d in sorted(st["ct_pretrain"].items(), key=lambda x: x[1]["n"]):
    r = d["ct_test"]
    print(f"{tag:12s} {d['n']:>6} {r['precision']:>7.3f} {r['recall']:>7.3f} "
          f"{r['f1']:>7.3f} {r['mAP50']:>8.3f} {r['mAP50_95']:>9.3f}")

if st["flower_ft"]:
    print("\n" + "=" * 78)
    print("TABLE 3 — flower333 (3-fold): CT vs GBIF vs COCO pretraining")
    print("=" * 78)
    base = KNOWN_FLOWER["C2_coco"]["mAP50"][0]
    rows = [(d["n"], t, d["mAP50"], d["recall"]) for t, d in KNOWN_FLOWER.items()]
    for tag, folds in st["flower_ft"].items():
        fr = [folds[str(k)] for k in range(CFG["k_folds"]) if str(k) in folds]
        if len(fr) == CFG["k_folds"]:
            rows.append((int(tag.replace("ct", "")), tag.upper(),
                         agg(fr, "mAP50"), agg(fr, "recall")))
    print(f"{'source':14s} {'N':>6} {'mAP50':>15} {'recall':>15} {'Δ vs COCO':>11}")
    for n, t, m50, rc in sorted(rows, key=lambda x: (x[1].lower().startswith("ct"), x[0])):
        print(f"{t:14s} {n:>6} {m50[0]:>8.3f}±{m50[1]:.3f} {rc[0]:>8.3f}±{rc[1]:.3f} "
              f"{m50[0]-base:>+11.3f}")

save_state(st)
print(f"\nruntime {(time.perf_counter()-t0)/60:.1f} min")

print(f"Outputs written to {ZIP.parent}; archive: {ZIP.name} (state.json, pre-trained weights, results.csv files)")
