# Roadmap

One line per task. `[x]` done, `[ ]` open. Details: `VISION.md` (architecture).

```text
P0 foundation -> P1 real data -> P2 shared core -> P3 official evaluation
   -> P4 debugging -> P5 more tasks -> P6 more models -> P7 baselines and release
```

## P0 Foundation (done)

- [x] Classification adapters (period, weather, scenario) from `attributes.*`
- [x] CI, pre-commit, ruff, mypy, bandit
- [x] Detection: COCO adapter, COCO to YOLO bridge, YOLO trainer, evaluate CLI
- [x] Semantic segmentation: image and mask adapter (19 classes), SMP trainer, mIoU
- [x] Official toolkit reviewed (spec facts and the bugs to avoid)

## P1 Real data and fidelity

- [x] Adapters fail loudly on empty splits and report dropped files
- [x] Detection labels: `det_20` and legacy JSON, both class-name schemes (fixtures only)
- [x] Packaging: Hydra configs resolve from a non-editable install
- [x] Classification: prepare, train and evaluate on real data (official and Kaggle layouts)
- [x] Classification: 3 tasks, 2 backends, 7+ models each, 3 tuning rounds, demo for image and video
- [ ] CI green on `main`, branch protection
- [x] Detection: real prepare (59,384 / 10,479 / 10,000 images) and a real smoke training run
- [x] Detection baselines: 9 YOLO models from DetectionBench re-evaluated here (match to 0.00004 mAP)
- [ ] Detection: higher resolution, longer schedule and larger models (leads in the baselines doc)
- [ ] Semantic segmentation: real prepare run and real smoke training
- [ ] Classification results write-up and Model Zoo refreshed with the final picks

## P2 Shared core

- [x] One command per verb: `bdd100k-prepare`, `-train`, `-evaluate`, `-coco-to-yolo`
- [ ] `labels` module: class names, ids, aliases and task subsets in one place
- [ ] Generic adapter registry shared by all tasks
- [ ] Harden `utils/bitmask.py`: named constants, vectorised decode, real fixtures
- [ ] Packaged default configs plus user overrides, no machine-local paths
- [ ] Shared argument layer so per-task scripts stop duplicating flags

## P3 Official-semantics evaluation

- [ ] Semantic seg: native-resolution mIoU, ignore 255, fIoU, pAcc
- [ ] Detection: COCO scores with BDD ignore rules (`other person`, `other vehicle`, `trailer`, crowd)
- [ ] Drivable area: background-aware IoU
- [ ] Tests against the official evaluator's test cases
- [ ] State clearly where our numbers are and are not leaderboard-comparable

## P4 Data and model debugging

- [ ] `validate`: missing or corrupt images, bad mask values, degenerate boxes, split leakage
- [ ] `stats`: class histograms, box and mask statistics, per-attribute and per-split reports
- [ ] `visualize`: boxes, masks and instances over images (Pillow only)
- [ ] Error analysis: per-class and per-attribute metrics, confusion matrices, worst-case gallery
- [ ] Label-issue finder: confident disagreements, outlier and duplicate boxes

## P5 More tasks

Each task: adapter, canonical format, trainer, evaluator, `validate/stats/visualize`, real-data check.

- [ ] Drivable area (shipped masks, 3 classes)
- [ ] Instance segmentation (8 classes, bitmasks to COCO RLE, mask AP)
- [ ] Panoptic segmentation (40 classes, COCO-panoptic, hand-rolled PQ): gap in the old zoo
- [ ] Box tracking, MOT (`box_track_20` to MOT-challenge, `motmetrics`): gap, check label availability first
- [ ] Segmentation tracking, MOTS (reuses MOT): gap
- [ ] Pose (18 keypoints, top-down, OKS AP): gap
- [ ] Lane marking (optional, boundary F-score)

## P6 More models

- [x] Classification: `timm` backend, EMA, early stopping, macro-F1 best checkpoint, class balancing
- [ ] One `ModelSpec` and `Trainer` protocol per task so `model.name=` selects any backend
- [x] Detection: RF-DETR train and evaluate (nano reproduces 31.58 / 56.90 on the val set)
- [ ] Detection: torchvision models
- [ ] Segmentation: more SMP architectures, SegFormer
- [ ] Every backend scored by the same P3 evaluator

## P7 Baselines and release

- [ ] Baselines table: config, seed, checkpoint hash and score per task, next to the old zoo's numbers
- [ ] Docs site (mkdocs)
- [ ] PyPI `0.2.0`, changelog, semver policy
- [ ] Unify best-checkpoint metric across backends (macro F1)

## Rules

- The official toolkit is a spec, never a runtime dependency
- No unpinned VCS dependencies
- Every bit and shift constant is named and tested against a real fixture
- Nothing is dropped without a count
- Every metric states which official semantics it follows
- A task is done only after a real-data check; fixtures are for unit tests
- No `git add` or `git commit` unless asked

## Open decisions (owner: you)

- Detection evaluation may depend on `pycocotools`? Default: yes
- Pose and lane marking: in scope? Default: pose in (later), lane optional
- Second dataset behind the same adapter interface? Default: not until asked

## Out of scope

- Domain adaptation and imitation learning tasks
- Wrapping the official toolkit or `scalabel` at runtime
- Reimplementing the official visualiser, GPS maps or download helpers
