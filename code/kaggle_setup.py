# Setup: pinned packages, import checks, GPU check, and the required datasets.
import os
import subprocess
import sys

pip = [sys.executable, "-m", "pip", "install", "-q"]
subprocess.run(pip + ["--no-deps", "ultralytics==8.4.108", "ultralytics-thop"], check=True)
subprocess.run(pip + ["transformers==4.51.3", "tokenizers==0.21.0", "huggingface_hub>=0.30,<1.0"], check=True)

import huggingface_hub
import numpy as np
import torch
import transformers
import ultralytics
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # noqa: F401

print("python", sys.version.split()[0], "| torch", torch.__version__, "| ultralytics", ultralytics.__version__,
      "| transformers", transformers.__version__, "| huggingface_hub", huggingface_hub.__version__,
      "| numpy", np.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
if not torch.cuda.is_available():
    raise RuntimeError("No GPU: set Settings > Accelerator > GPU T4.")

OWNER = "ehsanrahimi666"
IMG = (".jpg", ".jpeg", ".png")


def dataset_key(root):
    parts = root.split("/")
    return "/".join(parts[:6]) if len(parts) > 5 and parts[3] == "datasets" else "/".join(parts[:4])


def scan():
    """Flags for each required input, from one walk over /kaggle/input."""
    found = dict(flower=False, ct=False, gbif=False, weights=False)
    ct_tot, gb_tot = {}, {}
    for root, dirs, files in os.walk("/kaggle/input"):
        low = root.lower()
        if "flower333" in os.path.basename(low) and "data.yaml" in files:
            found["flower"] = True
        if "ct5000_pretrained.pt" in files:
            found["weights"] = True
        n_img = sum(f.lower().endswith(IMG) for f in files)
        n_txt = sum(f.lower().endswith(".txt") for f in files)
        if "apis" in low:
            t = ct_tot.setdefault(dataset_key(root), [0, 0])
            t[0] += n_img
            t[1] += n_txt
        if "gbif" in low or "honeybee" in low:
            t = gb_tot.setdefault(dataset_key(root), [0, 0])
            t[0] += n_img
            t[1] += n_txt
    found["ct"] = any(n >= 1000 and x >= 0.9 * n for n, x in ct_tot.values())
    found["gbif"] = any(n >= 1000 and x < 0.5 * n for n, x in gb_tot.values())
    return found


NEEDED = {
    "flower": ("flower333", ["flower333"]),
    "ct": ("Apis_mellifera", ["apis-mellifera", "apis_mellifera", "apismellifera"]),
    "gbif": ("GBIF_honeybees", ["gbif-honeybees", "gbif_honeybees", "gbifhoneybees"]),
    "weights": ("phonebee_revision_weights", ["phonebee-revision-weights", "phonebee_revision_weights"]),
}

found = scan()
for key, (label, slugs) in NEEDED.items():
    if found[key]:
        continue
    try:
        import kagglehub
    except ImportError:
        break
    for s in slugs:
        try:
            print("attached", label, "->", kagglehub.dataset_download(f"{OWNER}/{s}"))
            break
        except Exception as e:  # noqa: BLE001
            print(f"automatic attach of {OWNER}/{s} not possible: {type(e).__name__}")

found = scan()
for key, (label, _) in NEEDED.items():
    print(("found    " if found[key] else "MISSING  ") + label)
missing = [NEEDED[k][0] for k in NEEDED if not found[k]]
if missing:
    raise RuntimeError("Missing input datasets: " + ", ".join(missing)
                       + ". Attach them as notebook inputs and run the notebook again.")
print("All inputs present.")
