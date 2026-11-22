# Architecture

Why the toolkit exists and how it is built. Task status is in the main README,
open work in `ROADMAP.md`.

## Why

BDD100K is still one of the most useful driving datasets (100K images across
classification, detection and segmentation, plus video for tracking), but its
official toolkit, [`bdd100k/bdd100k`](https://github.com/bdd100k/bdd100k), no longer
works:

- **Unmaintained**: no maintainer response on open issues since 2021.
- **Broken by its own pins**: `pydantic<2.0.0` behind an unpinned `git+scalabel`
  dependency, which crashes `to_coco.py` under any modern pydantic
  ([bdd100k/bdd100k#354](https://github.com/bdd100k/bdd100k/issues/354)).
- **Unfixed gaps**: e.g. `to_mask.py` rejects its own documented modes
  ([bdd100k/bdd100k#317](https://github.com/bdd100k/bdd100k/issues/317)).
- **Dead download links**: `bdd-data.berkeley.edu` and `doc.bdd100k.com` are
  unreachable ([bdd100k/bdd100k#369](https://github.com/bdd100k/bdd100k/issues/369)).

This project is a from-scratch replacement scoped to benchmarking (prepare, train,
evaluate), not a fork. The official visualiser, format converters, GPS maps and
download helpers are out of scope. [official-toolkit-review.md](official-toolkit-review.md)
lists what was taken from it as spec and which of its bugs are avoided.

## Principles

1. **The official toolkit is a read-only spec, never a dependency.** Its docs, class
   tables and test fixtures define the formats; nothing is imported from it or from
   `scalabel`. Every format (box2d JSON, RGBA bitmasks, per-pixel masks) is parsed with
   plain Python, numpy and Pillow in this repo.
2. **One adapter per task and dataset.** Each task package (`classification/`,
   `detection/`, `segmentation/semantic/`) has a `base.py` (spec and abstract adapter),
   a `registry.py` and `datasets/<name>.py`, the only file that knows the raw format.
   Adapters write a canonical format that trainers read without knowing about BDD100K,
   so a new dataset needs one adapter and no trainer or CLI changes.
3. **Canonical formats that existing tools already read:**
   - Classification: `{train,valid,test}/<class>/*.jpg` (ImageFolder).
   - Detection: COCO JSON per split, bridged to YOLO `.txt` by `bdd100k-coco-to-yolo`.
   - Semantic segmentation: `{train,valid}/{images,masks}/`, masks with class ids and
     255 for ignore.
   - Planned: COCO with RLE for instance segmentation, COCO-panoptic for panoptic,
     MOT-challenge `gt.txt` for tracking.
4. **Delegate the training loop.** Trainers are thin wrappers around maintained
   libraries: Ultralytics and timm for classification, Ultralytics and RF-DETR for
   detection, segmentation-models-pytorch for semantic segmentation (Ultralytics has no
   dense per-pixel trainer). The toolkit's own code is the data plumbing and the
   metrics.
5. **No fragile dependencies.** No unpinned VCS installs. Where a reference tool has no
   PyPI release (e.g. `panopticapi`), the small piece needed is written here instead.
6. **Fail loudly.** An empty split raises, and every dropped image, box or mask is
   counted and printed (`utils/checks.py`).

## Commands

There is one command per verb, and the task is inferred from the dataset key
(`bdd100k_toolkit/cli.py`):

| Command | Does | Implemented in |
|---|---|---|
| `bdd100k-prepare` | raw download to canonical splits | `scripts/prepare_<task>.py` |
| `bdd100k-train` | Hydra-configured training (`configs/`) | `scripts/train_<task>.py` |
| `bdd100k-evaluate` | metrics on a split, written to `metrics.json` | `scripts/evaluate_<task>.py` |
| `bdd100k-coco-to-yolo` | COCO detection folders to Ultralytics YOLO | `utils/convert_coco_to_yolo.py` |

## Licence of the data

BDD100K is under its own licence: free for educational, research and not-for-profit
use with the UC Regents copyright notice carried forward; commercial use needs
permission. The code only transforms a local download; the Hugging Face copies carry
the same licence.
