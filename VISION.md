# Project Vision

A single-page reference for *why* BDD100K-Toolkit exists, *how* it's built, and
*where* it's going. `README.md` is the user-facing quickstart; `ROADMAP.md` is
the task-by-task checklist. This file is the connective narrative between
them, meant to survive long gaps between work sessions.

## Why this project exists

[BDD100K](https://www.bdd100k.com/) is still one of the most useful driving
datasets available (100K images across classification, detection, and
segmentation tasks, plus video for tracking). Its official toolkit,
[`bdd100k/bdd100k`](https://github.com/bdd100k/bdd100k), is not in a usable
state:

- **Unmaintained**: last commit over two years ago; no maintainer response on
  any open issue since 2021.
- **Broken by its own dependency pin**: hard-pins `pydantic<2.0.0` behind an
  unpinned `git+scalabel` dependency, which crashes the toolkit's own
  `to_coco.py` under any modern pydantic
  ([bdd100k/bdd100k#354](https://github.com/bdd100k/bdd100k/issues/354)).
- **Unfixed task gaps**: e.g. `to_mask.py` doesn't support `lane_mark` mode
  ([bdd100k/bdd100k#317](https://github.com/bdd100k/bdd100k/issues/317)).
- **Dead download infrastructure**: `bdd-data.berkeley.edu`, `doc.bdd100k.com`,
  and even the ETH mirror are unreachable across a dozen open issues,
  including `box_track_20` labels specifically
  ([bdd100k/bdd100k#369](https://github.com/bdd100k/bdd100k/issues/369)).

The *dataset* is still valuable and actively used in research; the *tooling*
around it is not. This project is a from-scratch, dependency-clean
replacement scoped to benchmarking (prepare → train → evaluate), not a fork
or a reimplementation of the full official toolkit's feature surface
(visualization, format conversion utilities, etc. are explicitly
out of scope).

Built to be genuinely useful to the community: anyone who wants to train or
benchmark a model on BDD100K today without fighting a two-year-old
dependency tree can clone this repo and get a working adapter immediately.

## Architectural philosophy

1. **The official repo is a read-only spec, never a runtime dependency.**
   `~/projects/bdd100k` (a local clone of `bdd100k/bdd100k`) is used purely to
   read label-format documentation (`doc/source/format.rst`), typing
   references, and test fixtures, to understand BDD100K's *raw* label
   encodings precisely. Nothing is ever `import`ed from it or `scalabel`.
   Every format (JSON box2d, RGBA instance/panoptic bitmasks, plain
   per-pixel semantic masks) is parsed with plain Python/numpy/Pillow code
   written in this repo. A review of that toolkit (its spec facts and the
   bugs we must not repeat) lives in
   [`docs/official-toolkit-review.md`](docs/official-toolkit-review.md).

2. **Adapter pattern, one per task.** Each task package
   (`classification/`, `detection/`, `segmentation/semantic/`, ...) exposes
   a `DatasetAdapter`-style class (registered via a small `registry.py`)
   whose job is: read BDD100K's native raw release layout for that task,
   and write out a **canonical, task-appropriate intermediate format**
   that downstream training code can consume without knowing anything
   about BDD100K. This isolates all BDD100K-specific parsing to one
   `datasets/bdd100k.py` file per task; adding a second dataset to any task
   later only requires a new adapter, not touching the trainer/CLI.

3. **Canonical format per task** (chosen for max compatibility with
   existing, actively maintained tooling, not invented from scratch):
   - Classification → `{train,valid,test}/<class>/*.jpg` (ImageFolder-style).
   - Detection → COCO JSON, bridged to YOLO `.txt` for training.
   - Semantic segmentation → `{train,valid,test}/{images,masks}/*.png` pairs.
   - Instance segmentation (planned) → COCO + RLE segmentation, bridged to
     YOLO-seg polygons for training.
   - Panoptic segmentation (planned) → standard COCO-panoptic (PNG +
     `segments_info` JSON).
   - Tracking (planned) → MOT-challenge `gt.txt` per sequence.

4. **Delegate, don't reimplement, the model/training loop.** Every
   trainer is a thin wrapper around an actively-maintained library:
   Ultralytics for classification/detection (and likely instance-seg via
   YOLO-seg), `segmentation-models-pytorch` for semantic segmentation
   (Ultralytics has no dense per-pixel trainer), `motmetrics` for tracking
   evaluation. This project's own code is the *data plumbing* (raw BDD100K
   → canonical format → metrics), not a competing model zoo.

5. **No speculative dependencies.** E.g. `panopticapi` (used by the
   reference ecosystem for PQ eval) has no PyPI release (only a git
   install), which is exactly the kind of fragile dependency that broke
   the official toolkit. Decision: hand-roll the small amount of
   COCO-panoptic encoding/eval logic needed instead of depending on it.

6. **Match DetectionBench's engineering bar.** Same `.pre-commit-config.yaml`,
   CI workflow, ruff/mypy/bandit config, and pinned dev-tool versions as the
   sibling `DetectionBench` project, so contributors moving between the two
   repos have zero context-switch cost.

## Status snapshot

| Task | Status | Notes |
|---|---|---|
| Period / weather / scenario classification | ✅ done | derived from native `attributes.*`, no extra download |
| Object detection | ✅ done | 10 classes, COCO→YOLO bridge, verified end-to-end |
| Semantic segmentation | 🚧 code in place | 19 classes, image+mask pairs, mIoU eval; tested on fixtures only, real-data run still open |
| Instance segmentation | planned | bitmask decoder already built (`utils/bitmask.py`), 8-class subset identified, adapter/trainer not yet written |
| Panoptic segmentation | planned | format researched, no code yet |
| Multi-object tracking | planned | label availability itself is uncertain (bdd100k/bdd100k#369) |

See `ROADMAP.md` for the granular, checklist-level breakdown and
format/implementation details per task, and the session todo list for
exact resume points.

## Constraints to keep respecting

- **Data access**: BDD100K is distributed under its own license
  (free for educational, research and not-for-profit use, with the copyright
  notice carried forward; commercial use needs permission). The code only
  transforms a user's own local download; a YOLO-format detection copy with the
  2018 labels is hosted separately at `dronefreak/BDD100K` on Hugging Face.
- **No `git add`/`git commit` without being explicitly asked**: commits
  happen only on explicit instruction.
