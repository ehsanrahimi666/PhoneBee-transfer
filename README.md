# Transfer of honeybee detectors from external imagery to smartphone video

Code, data partitions, saved model outputs and results for the article
"External Imagery for Automated Honeybee–Flower Monitoring: When Does Transfer Work?"
(Ehsan Rahimi and Chuleui Jung).

The study fine-tunes YOLO26n detectors on 333 annotated frames of a smartphone video of honeybees
on a flowering *Delosperma* mat (PhoneBee), after auxiliary pre-training on GBIF occurrence
photographs or on an insect camera-trap dataset, evaluates direct transfer between the three
domains, and tests the role of object scale by sliced inference.

## Contents

| Path | Content |
|------|---------|
| `code/pretrain_gbif.py` | Grounding DINO pseudo-labelling of N GBIF images, auxiliary pre-training at 640 px and the first fine-tuning run (Kaggle script; run once per N = 500, 2,000, 5,000) |
| `code/pretrain_camera_trap.py` | Camera-trap subsets, auxiliary pre-training at 640 px and the first fine-tuning run (Kaggle script) |
| `code/kaggle_setup.py` | Pinned packages and input checks for the experiment notebook |
| `code/experiments_kaggle.py` | Fixed 60-epoch fine-tuning, transfer matrix, GBIF reference set, scale-manipulation experiment, confidence and example predictions |
| `code/analysis.py` | All tables and summary statistics from the saved outputs |
| `code/compute_cost.py` | Training cost of auxiliary pre-training relative to the fine-tuning epochs it saves |
| `code/figures.py` | All figures (PNG at 300 dpi, TIFF at 600 dpi) |
| `code/gbif_provenance.py` | GBIF metadata (source dataset, licences, creator, date) of every GBIF image used |
| `code/data_workbooks.py` | Data S1 and Data S2 workbooks with README sheets |
| `data/` | PhoneBee fold assignments and annotated boxes, auxiliary subset membership, GBIF reference set, GBIF metadata |
| `outputs/original/` | Saved outputs of the auxiliary pre-training runs, the first fine-tuning run and the corpus characterization |
| `outputs/experiments/` | Saved outputs of `experiments_kaggle.py` (state.json, learning curves, sliced-inference predictions, examples) |
| `results/` | Outputs of `analysis.py` and `compute_cost.py` |
| `phonebee/annotated_frames/` | The 333 annotated PhoneBee frames (1920 × 1080 px) with YOLO labels (single class), in the train/valid/test folders of the annotation export; the cross-validation folds used in the article are listed in `data/phonebee_folds.csv` |
| `weights/pretrained/` | YOLO26n weights after auxiliary pre-training (GBIF and camera-trap, N = 500, 2,000, 5,000) with SHA-256 checksums |
| `weights/finetuned/` | Weights of the 21 fixed 60-epoch fine-tuning runs (configuration × fold) with SHA-256 checksums |

## Reproducing the results

The experiments were run in Kaggle notebooks on NVIDIA Tesla T4 GPUs. `experiments_kaggle.py`
expects the PhoneBee frames (`flower333`), the camera-trap images (`Apis_mellifera`), the GBIF
images and the pre-trained auxiliary weights as notebook inputs; run `kaggle_setup.py` first.

The analysis runs on any computer with the packages in `requirements.txt`:

```
python code/analysis.py outputs/original outputs/experiments results
python code/compute_cost.py results
python code/figures.py results outputs/original outputs/experiments figures
python code/data_workbooks.py results data outputs/experiments/state.json data workbooks
```

`analysis.py` reproduces every file in `results/` from the saved outputs. Folds are numbered 0–2 in file
names and code and 1–3 (in recording order) in the article and in the Excel workbooks.

## Data sources

- PhoneBee video frames and annotations: this study (single class, YOLO format; CC BY 4.0).
- Camera-trap images: Bjerge, K., Alison, J., Dyrmann, M., Frigaard, C. E., Mann, H. M. R., & Hoye, T. T. (2022).
  Benchmark datasets for detection and identification of insects from camera trap images with deep learning
  [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.7395752 (CC BY 4.0).
- GBIF images: identified by their occurrence keys in `data/gbif_membership.csv` and
  `data/gbif_occurrence_metadata.csv` (https://www.gbif.org/occurrence/<key>). The images are not
  redistributed here; the licence of each image is listed in `data/gbif_occurrence_metadata.csv`.
- Figure 1 uses three GBIF photographs published under CC BY 4.0 (credits in
  `outputs/original/exemplars/gbif_ccby/gbif_exemplars.json`).
