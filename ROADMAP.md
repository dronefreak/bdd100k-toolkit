# Roadmap

Step-by-step plan for BDD100K-Toolkit. See `README.md` for the "why",
`VISION.md` for the architecture, and
[`docs/official-toolkit-review.md`](docs/official-toolkit-review.md) for the
review of the official toolkit that drives phases 1-3.

**Direction:** a dependency-clean, trustworthy toolkit for BDD100K that
(1) prepares every task's data correctly, (2) lets you *inspect and debug* the
data and your models, (3) reports numbers that mean the same thing as the
official benchmark, and (4) plugs in multiple models. Official toolkit =
read-only spec; none of its code is reused (see the review's "rules").

**How to read this file:** phases are ordered by dependency. Each step has a
*done when* line; a phase is finished when all its steps are. Don't start a
phase before its predecessor's exit criteria are met, except where noted.

```text
P0 foundation (done) -> P1 real-data validation -> P2 shared core
   -> P3 official-semantics evaluation -> P4 data + model debugging
   -> P5 remaining tasks -> P6 more models -> P7 baselines & release
```

---

## Phase 0: Foundation (done)

- [x] Classification (period / weather / scenario) from native
      `attributes.*`; `UltralyticsClassificationTrainer` (now one of two classification backends); CLIs; Hydra configs.
- [x] CI + pre-commit + ruff/mypy/bandit ported from DetectionBench.
- [x] Detection: COCO adapter, COCO->YOLO bridge, YOLO trainer, evaluate CLI
      (RF-DETR intentionally not ported).
- [x] Semantic segmentation: image+mask adapter (19 classes), SMP trainer,
      numpy mIoU, CLIs.
- [x] `utils/bitmask.py` RGBA decoder (not yet used by any task).
- [x] Official toolkit review documented (`docs/official-toolkit-review.md`).

All verified on synthetic fixtures only. Nothing has run on real BDD100K yet.

---

## Phase 1: Real-data validation and fidelity fixes

Goal: stop trusting synthetic fixtures. Find out what breaks on the real
download, and fix the silent-failure modes the official review exposed.

1. **Publish and get CI green.** Push to GitHub; fix whatever CI finds
   (suspects: torch-importing modules under the `[dev]` extra,
   template-leftover `docs-build` hook). Add branch protection, CI badge.
   *Progress:* `mypy` failure in `tests/test_miou.py` fixed (`MeanIoU`
   TypedDict); local ruff/mypy/bandit/pytest clean; `CONTRIBUTING.md` added.
   *Remaining (needs a push, which waits for your go-ahead):* confirm the CI
   matrix is green; branch protection; badge.
   *Done when:* lint + test matrix green on `main`.
2. **Fail loudly.** *(done 2026-10-01)* Every adapter raises on empty splits (e.g. 0 image/mask
   pairs) and reports counts of dropped files, unknown categories, and
   missing images.
   *Done when:* tests cover each failure; no adapter can finish "successfully"
   with an empty split.
3. **Detection label sources.** *(done 2026-10-01, on fixtures only: the real
   `det_20` JSON has not been seen, so its field layout is assumed to match the
   legacy `box2d` shape per the official docs)* Support `labels/det_20/det_{train,val}.json`
   next to the legacy 2018 JSON; accept both `person/motor/bike` and
   `pedestrian/motorcycle/bicycle`; warn on any class with zero boxes.
   Keep `crowd/occluded/truncated` and real `iscrowd`; read image size from
   the file, not constants.
   *Done when:* both label files produce identical class sets; tests use
   fixtures for both name schemes.
4. **Real-data prepare runs.** Run all five prepare CLIs on a real download;
   compare class distributions and split sizes with the official
   `bdd100k/data/{train,val,test}.txt` counts and DetectionBench's stats.
   *Done when:* a short `docs/data-notes.md` records real counts and any
   surprises (file layout, naming, missing files).
   *Progress:* classification (period / weather / scenario) done from both the
   official layout and the Kaggle folders; outputs identical; split sizes
   checked against the official lists; findings in `docs/data-notes.md`.
   Detection and semantic segmentation prepare runs still to do (the 10K
   images and masks are in `bdd100k_seg`; detection needs the same raw
   download).
5. **Real smoke-training runs.** One short run per task (yolo11n-cls weather,
   yolo11n detection, Unet/resnet34 seg) on a small real subset.
   *Done when:* each task trains and evaluates end to end on real files.
   *Progress:* classification done (yolo11n-cls, real data, GPU), including a
   per-class evaluation report; detection and segmentation still to do.
   `bdd100k-prepare --max-width 512` (resized train/valid copies) makes
   Ultralytics training about 2.5-3.5x faster (see `docs/data-notes.md`).
   Classification follow-ups: select `best.pt` by macro-F1 instead of
   Ultralytics' top-1 fitness; try class-balanced sampling or loss weights for
   the rare classes; confusion-matrix plot.
6. **Packaging fixes.** *(done 2026-10-01; verified by building a wheel and
   composing every Hydra config; the 5 CLIs themselves were not run because
   torch/ultralytics are not installed in this environment)* Hydra configs resolvable from a non-editable install;
   `convert_coco_to_yolo` omits missing splits, relative `path:`, no stale
   links; classification evaluate falls back to CPU.
   *Done when:* a clean-venv `pip install .` runs every CLI.

**Exit:** all five tasks reproducibly prepare, train and evaluate on real data,
and none can fail silently.

---

## Phase 2: Shared core

Goal: remove duplication *before* adding tasks (the ROADMAP's earlier
"decide on common module" question, decided: yes, now, because tasks 6-9
would otherwise copy it four more times).

1. **Label tables module** (`bdd100k_toolkit.labels`): the single source for
   class names, ids (0- and 1-based), aliases (`person`<->`pedestrian`, ...),
   task subsets (10 / 8 / 19 / 40 classes), stuff/thing flags. Built from the
   official tables, tested against them.
2. **Generic adapter/registry base** shared by classification, detection and
   segmentation (one `Registry[T]`, one `Spec` dataclass family); every
   adapter uses `utils/split.seeded_holdout`.
3. **Bitmask module hardened:** named constants for every bit/shift,
   vectorised decode (no per-instance full-image scans), tests against real
   official fixtures.
4. **One command per verb.** *(done 2026-10-01)* `bdd100k-prepare`,
   `bdd100k-train`, `bdd100k-evaluate` (+ `bdd100k-coco-to-yolo`), task inferred
   from the dataset key or given with `--task`/`task=`; no per-task commands.
   Still to do: `bdd100k-inspect` (Phase 4) and a shared argument layer so the
   per-task scripts stop duplicating flags.
5. **Config story:** packaged default configs plus user overrides; no
   machine-local `/path/to/...` defaults.

*Exit:* adding a task means writing one adapter + one trainer; no copied
registry code; existing tests unchanged and passing.

---

## Phase 3: Official-semantics evaluation

Goal: numbers that mean what the official benchmark's numbers mean.
Pure numpy / pycocotools; no `scalabel`. Each evaluator is tested against the
official toolkit's own test cases (`tests/eval/testcases`) where licence
permits, and documents any intentional deviation.

1. **Semantic seg:** native-resolution mIoU, ignore=255, average over classes
   present in GT, per-class IoU, fIoU, pAcc. Replaces the resized-mask path.
2. **Detection:** COCO 12 scores with BDD ignore handling (`other person`,
   `other vehicle`, `trailer`, crowd suppression at IoU/IoF 0.5). Keep the
   Ultralytics metric as a secondary, clearly labelled number.
3. **Drivable area:** same engine as sem-seg with background-aware IoU.
4. Evaluators for later tasks are added with their tasks in Phase 5
   (instance: mask COCO eval; panoptic: PQ/SQ/RQ; tracking: CLEAR-MOT via
   `motmetrics`).

*Exit:* sem-seg and detection scores reproduce the official evaluator on the
shared test cases; README states exactly where our numbers are and aren't
leaderboard-comparable.

---

## Phase 4: Data and model debugging

Goal: the "see what's wrong" layer. Everything works on the canonical format,
so it applies to every task. BDD100K's per-image attributes (weather / scene /
time of day) make slice analysis a first-class feature.

1. **`validate`:** integrity checks on a canonical dataset or raw download:
   missing/corrupt images, size mismatches, mask values outside the class
   range, degenerate boxes, duplicate names, split leakage between
   train/valid/test, empty images.
2. **`stats`:** class histograms, boxes per image, box size/aspect
   distributions, mask pixel frequencies, per-attribute breakdowns, split
   comparisons; JSON + HTML/markdown report.
3. **`visualize`:** render boxes / masks / bitmask instances over images to a
   folder (matplotlib-free path via Pillow; no official viewer).
4. **Prediction error analysis:** given a checkpoint + split, produce
   per-class and per-attribute-slice metrics, confusion matrices, a ranked
   gallery of worst false positives / false negatives (detection) and
   lowest-IoU images (segmentation).
5. **Label-issue finder:** flag likely label errors (high-confidence model
   disagreements, tiny/huge outlier boxes, near-duplicate boxes).

*Exit:* `validate` + `stats` + `visualize` work for classification,
detection and semantic seg; error analysis works for detection and seg;
a worked example in `docs/` finds a real issue in real data.

---

## Phase 5: Remaining tasks

Order by cost and by how much shipped data each needs. Each task = adapter +
canonical format + trainer + official-semantics evaluator + `validate/stats/
visualize` support + tests with real fixtures.

1. **Drivable area** (cheap: shipped masks, 3 classes, background-aware IoU).
2. **Instance segmentation:** decode `ins_seg` bitmasks with the Phase 2
   decoder; 8 classes (ids 1-8); canonical COCO + RLE (`pycocotools`);
   bridge to YOLO-seg polygons; mask-COCO evaluator.
3. **Panoptic segmentation:** `pan_seg` bitmasks, 40 classes (1-30 stuff,
   31-40 things); canonical COCO-panoptic (`id = R + 256*G`); hand-rolled
   PNG re-encoder and PQ evaluator (no `panopticapi`).
4. **Multi-object tracking:** `box_track_20` -> MOT-challenge `gt.txt`;
   `motmetrics` evaluation (mMOTA / mIDF1 over 8 classes + 3 super-
   categories). Field names (`videoName`, `frameIndex`, `labels[].id`) are
   confirmed from official docs but not from a real file; label availability
   is uncertain (bdd100k/bdd100k#369). **Check availability first.**
5. **Lane marking** (optional): 1-channel byte with direction/style/category
   bits; boundary F-score at 1/2/5 px.
6. Segmentation tracking and pose: out of scope unless requested.

*Exit per task:* real-data prepare + train + evaluate verified, as in Phase 1.

---

## Phase 6: More models

Goal: swap models without touching data code.

1. **Model registry per task:** a `ModelSpec` + `Trainer` protocol so
   `model.name=` selects any registered backend.
2. **Detection:** RF-DETR (deferred earlier), torchvision (Faster R-CNN /
   RetinaNet), keep Ultralytics.
3. **Segmentation:** more SMP architectures (DeepLabV3+, FPN), plus a
   Hugging Face SegFormer backend.
4. **Classification:** `timm` backbones next to Ultralytics. *(done 2026-10-01:
   optional `timm` extra, `model.backend`, macro-F1 checkpointing, class
   balancing, EMA, early stopping and metric-based best-checkpoint saving
   (`training.monitor`) in both backends, backend-aware evaluation; comparison
   results in
   `docs/classification-results.md`.)*
5. Every backend must be evaluated by the *same* Phase 3 evaluator.

*Exit:* two or more backends per task trainable and evaluated through one
CLI, with identical evaluation code.

---

## Phase 7: Baselines and release

1. Reproducible baseline table (config + seed + checkpoint hash + official-
   semantics score) per task, under `docs/baselines.md`.
2. Docs site (mkdocs; the existing `docs-build` pre-commit hook becomes real).
3. PyPI release `0.2.0`, changelog, semver policy.
4. Second dataset behind the same adapter interface (e.g. another driving
   dataset) to prove the abstraction. Only if there's demand.

---

## Cross-cutting rules (apply to every phase)

- Official toolkit is a spec, never a runtime dependency; no unpinned VCS
  dependencies.
- No float round-trip / colour-space encoding of ids; every bit/shift constant
  is named and tested against a real fixture.
- Adapters and evaluators fail loudly; nothing is dropped without a count.
- Every metric documents which official semantics it follows.
- Real-data verification before a task is marked done; synthetic fixtures are
  for unit tests only.
- No `git add`/`git commit` without being explicitly asked.

## Decisions needed (owner: you)

1. Phase 4 before Phase 5? Default: yes, because debugging tools make the new tasks'
   adapters much easier to verify.
2. Is detection evaluation allowed to depend on `pycocotools` (maintained, but
   a C-extension)? Default: yes.
3. Lane marking and pose: in or out? Default: lane optional, pose out.
4. Second dataset (Phase 7.4): in scope or not? Default: not until asked.
5. Real BDD100K data location and which label release you have (legacy
   `bdd100k_labels_images_*.json` vs `det_20`); this gates Phase 1.3-1.5.

## Explicitly out of scope

- Domain adaptation / imitation learning tasks.
- Depending on or wrapping the official toolkit or `scalabel` at runtime.
- Reimplementing the official visualiser, GPS trajectory maps, or data
  rsync/zip helpers.
