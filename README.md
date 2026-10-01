# BDD100K-Toolkit

Unofficial, modern, dependency-clean toolkit for [BDD100K](https://www.bdd100k.com/) tasks.

## Why this exists

The official [`bdd100k/bdd100k`](https://github.com/bdd100k/bdd100k) toolkit
has been unmaintained since early 2024 (no maintainer response on any open
issue since 2021), pins `pydantic<2.0.0` behind a hard, unpinned
`git+scalabel` dependency that **crashes its own `to_coco.py` under modern
pydantic** ([bdd100k/bdd100k#354](https://github.com/bdd100k/bdd100k/issues/354)),
has unfixed task gaps (e.g. `to_mask.py` not supporting `lane_mark` mode,
[bdd100k/bdd100k#317](https://github.com/bdd100k/bdd100k/issues/317)), and its
official download infrastructure (`bdd-data.berkeley.edu`, `doc.bdd100k.com`,
even the ETH mirror) is largely unreachable as reported across a dozen open
issues. BDD100K the *dataset* is still widely used; the *tooling* is not.

This project starts small and depends only on modern, actively maintained
libraries (Hydra, Ultralytics, PyTorch), with no `scalabel` and no pinned pydantic.

See [`VISION.md`](VISION.md) for the full rationale, architectural
philosophy, and task status, and [`ROADMAP.md`](ROADMAP.md) for the
granular checklist.

## Tasks

| Task | Status | Classes | Source |
|---|---|---|---|
| Time-of-day (period) classification, unofficial | ✅ | 4 (daytime, night, dawn or dusk, unknown) | `attributes.timeofday` |
| Weather classification, unofficial | ✅ | 7 (clear, partly cloudy, overcast, rainy, snowy, foggy, unknown) | `attributes.weather` |
| Scenario (scene) classification, unofficial | ✅ | 7 (city street, highway, residential, parking lot, gas stations, tunnel, unknown) | `attributes.scene` |
| Object detection | ✅ | 10 | native `box2d` labels |
| Semantic segmentation | ✅ | 19 | native `sem_seg` masks |
| Instance segmentation | planned | n/a | native `ins_seg` masks (RLE) |
| Panoptic segmentation | planned | n/a | native `pan_seg` masks |
| Multi-object tracking | planned | n/a | native `box_track` labels |

The three classification tasks are **unofficial**: BDD100K defines no
classification benchmark, only per-image `attributes: {weather, timeofday,
scene}` in its detection label JSON. The tasks follow three Kaggle datasets
(`bdd100k-period-classification`, `bdd100k-weather-classification`,
`bdd100k-scenario-classification`, uploaded by `marquis03`, Apache-2.0 on
Kaggle). Checked image by image against the official 2018 labels, those
datasets are the same labels and the same images, with `unknown` being the
official `undefined` value and their `test` folder being the official
unlabeled test split (ignored here). `unknown` is kept as a class by default
(`--exclude-unknown` drops it).

Licensing: the Apache-2.0 label on a Kaggle re-upload does not by itself
change the terms of the underlying BDD100K images and labels (see License
below). This toolkit never redistributes data.

## Data layout

### Classification

Two input layouts are auto-detected under `--raw-dir`:

```text
# official BDD100K (images and labels are separate archives; use
# --labels-dir if the labels were extracted somewhere else)
raw_dir/
  images/100k/{train,val}/**/*.jpg
  labels/bdd100k_labels_images_{train,val}.json

# Kaggle class folders (test/ is unlabeled and ignored)
raw_dir/
  {train,val}/<class>/*.jpg
```

Both produce identical output (verified on the real data for all three
tasks). Official `val` becomes `test`; a seeded 15% of `train` becomes
`valid`. The holdout is chosen over the sorted names of all train images, so
it is the same for all three tasks, for detection, and with or without
`--exclude-unknown`.

Canonical output (plain `torchvision.datasets.ImageFolder` layout:
Ultralytics' classification trainer reads this directly, no glue needed):

```text
canonical_out/
  train/<class>/*.jpg
  valid/<class>/*.jpg
  test/<class>/*.jpg
```

### Detection

Same raw input as classification (`box2d` labels live in the same JSON
files). Either the 2020 revision `labels/det_20/det_{train,val}.json`
(preferred when present; the official docs recommend it) or the 2018
`labels/bdd100k_labels_images_{train,val}.json` works; both naming schemes
(`person/motor/bike` and `pedestrian/motorcycle/bicycle`) are accepted. Only
the 2018 file is guaranteed to carry the frame `attributes` that the
classification tasks need. Canonical output is COCO, bridged into Ultralytics YOLO format by a
generic converter:

```text
canonical_coco/
  train/{_annotations.coco.json,*.jpg}
  valid/{_annotations.coco.json,*.jpg}
  test/{_annotations.coco.json,*.jpg}

yolo_out/
  images/{train,val,test}/*.jpg
  labels/{train,val,test}/*.txt
  data.yaml
```

### Semantic segmentation

**Different raw image package**: BDD100K's segmentation tasks ship with
the separate "10K Images" package (`images/10k/...`), not the 100K set used
above; per BDD100K's own docs this "is not a subset of the 100K images, even
though there is a significant overlap", so this needs its own download.

```text
raw_dir/
  images/10k/{train,val,test}/*.jpg
  labels/sem_seg/masks/{train,val}/*.png   # 1-channel, pixel value = class id, 255 = ignore

canonical_out/
  train/{images,masks}/*.{jpg,png}
  valid/{images,masks}/*.{jpg,png}
  test/{images,masks}/*.{jpg,png}
```

## Quickstart

Four commands cover every task: `bdd100k-prepare`, `bdd100k-train`,
`bdd100k-evaluate` and `bdd100k-coco-to-yolo`. The task is inferred from the
dataset key (`--dataset` / `dataset=`); `--task` / `task=` overrides it, and
running a command with `--help` lists every dataset.

```bash
pip install -e ".[dev]"

# --- Classification ---

# 1. Convert a download into canonical weather-classification splits.
#    --raw-dir is either the official download or a Kaggle class-folder tree;
#    add --labels-dir if the label JSON lives outside <raw-dir>/labels, and
#    --exclude-unknown to drop the `unknown` class.
bdd100k-prepare --dataset bdd100k-weather \
    --raw-dir /path/to/bdd100k_raw --output-dir /path/to/canonical_out

# 2. Train (data_dir is required; no default)
bdd100k-train dataset=bdd100k-weather model.name=yolo11n-cls \
    dataset.data_dir=/path/to/canonical_out

# 3. Evaluate (top-1/top-5, per-class precision/recall/F1, macro F1, confusion matrix)
bdd100k-evaluate --dataset bdd100k-weather \
    --checkpoint experiments/bdd100k-weather/yolo11n-cls/weights/best.pt \
    --data-dir /path/to/canonical_out

# --- Detection ---

# 1. Convert the same raw BDD100K download into canonical COCO detection splits
bdd100k-prepare --dataset bdd100k-detection \
    --raw-dir /path/to/bdd100k_raw --output-dir /path/to/canonical_coco

# 2. Bridge into Ultralytics YOLO format
bdd100k-coco-to-yolo \
    --input-dir /path/to/canonical_coco --output-dir /path/to/yolo_out

# 3. Train (evaluates the best checkpoint afterward; dataset_yaml is required)
bdd100k-train dataset=bdd100k-detection model.name=yolo11n \
    dataset.dataset_yaml=/path/to/yolo_out/data.yaml

# 4. Evaluate a checkpoint directly
bdd100k-evaluate \
    --checkpoint experiments/bdd100k-detection/yolo11n/weights/best.pt \
    --dataset bdd100k-detection --dataset-yaml /path/to/yolo_out/data.yaml

# --- Semantic segmentation ---

# 1. Convert a raw BDD100K 10K-images download into canonical image/mask splits
bdd100k-prepare --dataset bdd100k-semantic-seg \
    --raw-dir /path/to/bdd100k_10k_raw --output-dir /path/to/canonical_out

# 2. Train (needs the `segmentation` extra: pip install -e ".[segmentation]")
bdd100k-train dataset=bdd100k-semantic-seg \
    model.architecture=Unet model.encoder_name=resnet34 \
    dataset.data_dir=/path/to/canonical_out

# 3. Evaluate (prints + saves per-class IoU and mIoU)
bdd100k-evaluate \
    --checkpoint experiments/bdd100k-semantic-seg/Unet-resnet34/model.pt \
    --dataset bdd100k-semantic-seg --data-dir /path/to/canonical_out
```

## Classification backends

Two backends train the classification tasks, chosen with `model.backend`:

| Backend | Models | Install |
|---|---|---|
| `ultralytics` (default) | `yolo11n-cls`, `yolov8s-cls`, ... | included |
| `timm` | any of timm's ~1,300 architectures (`convnext_tiny`, `resnet50`, `efficientnet_b0`, ...) with pretrained weights | `pip install "bdd100k-toolkit[timm]"` |

```bash
bdd100k-train dataset=bdd100k-weather model.backend=timm model.name=convnext_tiny \
    dataset.data_dir=/path/to/canonical_out
# timm options go through training.extra (a typo fails loudly):
#   +training.extra.balance=loss        class-weighted loss (or `sampler`)
#   +training.extra.balance_power=0.5   weights = count ** -power
#   +training.extra.label_smoothing=0.1 +training.extra.weight_decay=0.05
bdd100k-evaluate --dataset bdd100k-weather --checkpoint <run>/weights/best.pt \
    --data-dir /path/to/canonical_out   # the backend is detected from the file
```

The timm backend is a small PyTorch loop written for these tasks: it keeps the
checkpoint with the best **macro F1** on `valid` (not top-1), resizes the whole
image to a square at train and test time (no centre crop that would cut off
the sky or the road edge), and uses no colour jitter by default because
brightness is the time-of-day label. Both backends are evaluated by the same
report (per-class F1, macro F1, confusion matrix).

## Failing loudly

Prepare commands never finish "successfully" with an empty `train` or `test`
split (e.g. a wrong raw layout): they raise `EmptySplitError`. Everything an
adapter drops (missing images, degenerate boxes, ignored regions such as
`other person`, `other vehicle` and `trailer`, and unknown categories) is
counted and printed per split; classes with zero boxes raise a warning.

Train configs have no machine-local paths: required values such as
`dataset.data_dir` are Hydra `???` and must be passed on the command line.
`training.device` defaults to `auto` (CUDA if available, else CPU).

## License

BSD-3-Clause for this toolkit's code. BDD100K's own data remains governed by
the [BDD100K License](https://www.bdd100k.com/) (non-commercial
research/education; registration-gated; no redistribution). This project
never redistributes BDD100K data itself.
