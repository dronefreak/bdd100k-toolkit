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
| Time-of-day (period) classification | ✅ | 3 (daytime, night, dawn/dusk) | native `attributes.timeofday` |
| Weather classification | ✅ | 6 (clear, partly cloudy, overcast, rainy, snowy, foggy) | native `attributes.weather` |
| Scenario (scene) classification | ✅ | 6 (city street, highway, residential, parking lot, gas stations, tunnel) | native `attributes.scene` |
| Object detection | ✅ | 10 | native `box2d` labels |
| Semantic segmentation | ✅ | 19 | native `sem_seg` masks |
| Instance segmentation | planned | n/a | native `ins_seg` masks (RLE) |
| Panoptic segmentation | planned | n/a | native `pan_seg` masks |
| Multi-object tracking | planned | n/a | native `box_track` labels |

The three classification tasks above mirror three Kaggle re-exports by user
`marquis03` (`bdd100k-period-classification`, `bdd100k-weather-classification`,
`bdd100k-scenario-classification`), but are derived directly from BDD100K's
own official label release instead of the Kaggle mirrors, since every BDD100K
detection image is already tagged with `attributes: {weather, timeofday,
scene}`, so no separate download or third-party class/split assumptions are
needed.

## Data layout

### Classification

Raw input (identical to what BDD100K's own detection release ships):

```
raw_dir/
  images/100k/{train,val}/**/*.jpg
  labels/bdd100k_labels_images_{train,val}.json
```

Canonical output (plain `torchvision.datasets.ImageFolder` layout:
Ultralytics' classification trainer reads this directly, no glue needed):

```
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

```
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

```
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

# 1. Convert a raw BDD100K download into canonical weather-classification splits
bdd100k-prepare --dataset bdd100k-weather \
    --raw-dir /path/to/bdd100k_raw --output-dir /path/to/canonical_out

# 2. Train (data_dir is required; no default)
bdd100k-train dataset=bdd100k-weather model.name=yolo11n-cls \
    dataset.data_dir=/path/to/canonical_out

# 3. Evaluate
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
