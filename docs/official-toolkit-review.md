# Review of the official `bdd100k/bdd100k` toolkit

Reference review, written 2026-10-01 against a local clone of
`bdd100k/bdd100k` (last commit `9ac17c6`, 2024-03-09, "Update download link").
Purpose: use the official toolkit as a **read-only spec** (formats, class
tables, field names, fixtures) and record where its *code* is wrong or fragile
so BDD100K-Toolkit does not repeat it. See `VISION.md` principle 1.

**Scope caveat:** this was a read-through, not an execution. `scalabel` and
`pydantic` were not installed in the review environment (matplotlib 3.6.3,
numpy 1.26.4 were), so none of the bugs below were reproduced at runtime; each
cites the line that produces it. Items marked *(from memory)* rely on library
changelogs, not on anything verified here.

## 1. What the official toolkit is

A thin layer over `scalabel` (label I/O, COCO conversion, detection / MOT /
pose evaluation, poly2d rendering). The repo itself adds:

| Area | Files | Notes |
|---|---|---|
| Label tables | `label/label.py`, `configs/*.toml` | Cityscapes-style `trainId`s; per-task category lists |
| Poly2d -> masks | `label/to_mask.py` | matplotlib rasterisation |
| Masks -> COCO / RLE / COCO-panoptic | `label/to_coco.py`, `to_rle.py`, `to_coco_panseg.py`, `from_coco_panseg.py` | |
| Evaluation | `eval/{seg,ins_seg,pan_seg,mots,lane,run}.py` | seg/pan/lane are hand-written; det/MOT/pose delegate to scalabel |
| Misc | `vis/`, `data/` (split lists, rsync/zip helpers) | out of scope for the toolkit |

Dependency state: `requirements.txt` has `pydantic<2.0.0` and
`git+https://github.com/scalabel/scalabel.git` (unpinned HEAD). CI uses
`actions/*@v2` and Python 3.7-3.9.

## 2. Bugs found in the official code (do not port)

| # | Where | Problem | Effect |
|---|---|---|---|
| 1 | `label/to_mask.py` `main()` uses `to_coco.parse_args` | `--mode` choices there are `det, ins_seg, box_track, seg_track, pose` | Documented `to_mask -m sem_seg\|drivable\|lane_mark\|pan_seg` is rejected by argparse. Likely the real cause of bdd100k/bdd100k#317 |
| 2 | `to_mask.py:95` `fig.canvas.tostring_rgb()` | Removed in matplotlib 3.10 *(from memory)* | Mask rendering crashes on current matplotlib |
| 3 | `to_mask.py` `frame_to_mask` | Writes `((i+1)>>8)` and `((i+1)%255)`, decodes `(R<<8)+G`. `%255` vs `&255`/`>>8` mismatch | At 255 instances the id decodes to 0 (background); at 256 it decodes to 257. Images with >=255 polygons silently lose instances |
| 4 | `to_coco.py:707` `bitmask2coco_with_ids` | `(B << 2) + A` where the format is `(B << 8) + A` | Wrong instance masks for ids above 3 |
| 5 | `to_coco_panseg.py` `bitmask2pan_json` | `cat_id_to_idx[category_id] = len(segment_info) - 1` takes the length of the *dict* (5 keys), not of `segments_info` | Stuff-segment areas merged into the wrong segment |
| 6 | `to_mask.py` `frame_to_mask` | No `return` after saving the empty-frame image | Falls through into the matplotlib render |
| 7 | Rasterising via matplotlib | Anti-aliased edges blend colours | Boundary pixels can decode to wrong ids; inherent to the approach |
| 8 | Packaging | Unpinned git dependency + `pydantic<2` | Install is not reproducible; breaks under modern pydantic (#354) |

Design lesson: the official converters encode ids in colour channels and then
rely on exact float round-trips and magic shifts. The toolkit reads the
*released* masks/bitmasks directly (they are already shipped) and only decodes.

## 3. Facts to treat as spec (verified against official docs/code)

**Bitmask (`doc/source/format.rst`, `common/bitmask.py`, fixture
`tests/label/testcases/bitmasks/quasi-video/insseg_bitmask.png`):**
- RGBA PNG. R = category id (1-based, 0 = background). G =
  `(truncated<<3)+(occluded<<2)+(crowd<<1)+ignore`. Instance id =
  `(B<<8)+A`. Fixture check: G values `{0, 4}` (occluded bit) as expected.
- Crowd/ignore instances are excluded from evaluation.

**Class tables:**
- Detection: 10 classes, official ids 1-based
  (`pedestrian, rider, car, truck, bus, train, motorcycle, bicycle,
  traffic light, traffic sign`). Raw legacy labels use `person`, `motor`,
  `bike`; official `name_mapping` maps them (`person->pedestrian`,
  `motor->motorcycle`, `bike->bicycle`, `van/caravan->car`).
  `other person`, `other vehicle`, `trailer` are *ignored regions*
  (`ignored_mapping`), not dropped.
- Instance seg / box tracking / seg tracking: first 8 of the above.
- Semantic seg: 19 `trainId` classes, 255 = unknown/ignore.
- Panoptic: 40 classes, ids 1-30 stuff, 31-40 things, 0 unlabeled.
  Stuff segments have instance id == category id; things are numbered from 31
  per image. COCO-panoptic PNG encoding is `R = A`, `G = B`, i.e.
  `id = R + 256*G`. `iscrowd = crowd or ignored`.
- Drivable: direct / alternative / background (background not in mIoU).
- Lane: 1-channel byte, bits `-|-|d|s|b|c|c|c` (direction, style,
  background, 3 category bits).

**Tracking field names** (`evaluate.rst`, result format): `videoName`,
`name`, `frameIndex`, `labels[].id`, `category`, `box2d`. Verified from docs
only, not from a real `box_track_20` file.

**Release layout (`download.rst`):**
- 100K images: `images/100k/{train,val,test}` (detection, drivable, lane).
- 10K images: `images/10k/{train,val,test}` (sem/ins/pan seg); *not* a subset
  of the 100K images.
- Labels: legacy 2018 detection JSON (107 MB, includes `weather/scene/
  timeofday`) is **deprecated, "kept for comparison with legacy results"**;
  the recommended set is `labels/det_20/det_{train,val}.json`.
- `labels/sem_seg/{masks,colormaps,polygons,rles}`, `labels/ins_seg/bitmasks`,
  `labels/pan_seg/bitmasks`, `labels/box_track_20/{train,val}`,
  `labels/seg_track_20/...`, `labels/pose_21/pose_{train,val}.json`.
- `bdd100k/data/{train,val,test}.txt` list 70,000 / 10,000 / 20,000 video
  names: the authoritative split sizes.

**Evaluation semantics (what "official numbers" mean):**
- Detection / ins-seg: COCO-style 12 scores; crowd and ignored-class regions
  suppress false positives (>50% overlap).
- Semantic seg: confusion matrix over native-resolution masks; ignore pixels
  excluded from GT; mIoU averaged over classes **present in GT**; predictions
  >= num_classes treated as ignore.
- Panoptic: standard PQ/SQ/RQ over things / stuff / all; IoU > 0.5 match.
- MOT: CLEAR-MOT, mMOTA/mIDF1 over the 8 classes plus 3 super-categories.

## 4. Implications for BDD100K-Toolkit

These are open issues in *this* repo exposed by the comparison. They are
scheduled in `ROADMAP.md` (Phases 1-3).

1. **Detection reads the deprecated labels.** The adapter only supports
   `bdd100k_labels_images_{train,val}.json`. Results are not comparable to the
   current benchmark. Unverified risk: if `det_20` uses `pedestrian /
   motorcycle / bicycle`, the current category filter would silently drop
   those classes. Needs: `det_20` support, an alias map for both naming
   schemes, and a loud warning when a split yields zero boxes for a class.
2. **Metrics are not leaderboard-comparable.** Detection mAP comes from
   Ultralytics without ignored-region handling; semantic mIoU is computed on
   masks resized to `imgsz x imgsz` (not native 1280x720) and `mean_iou`
   includes predicted-only classes at IoU 0 (official excludes them). Either
   implement official-semantics evaluators or label our numbers clearly.
3. **Silent empty output.** Semantic seg pairs masks by identical file stem;
   a layout with `*_train_id.png` names yields `0 image/mask pairs` with no
   error. Empty splits must raise.
4. **Toolkit-side defects found in the same pass** (not from the official
   code): `convert_coco_to_yolo` writes `val:` in `data.yaml` even if no
   valid split exists and uses an absolute `path:`; Hydra `CONFIGS_DIR`
   (`parents[3]`) only resolves for editable installs; the semantic-seg
   trainer has no validation loop, squashes to a square, ignores
   `dataset.ignore_index` from config; `decode_bitmask` is O(instances x
   pixels); COCO `iscrowd` is always 0 and `crowd/occluded/truncated` are
   dropped; classification `--device` defaults to `cuda` unconditionally.
5. **Use the official fixtures as test data.** `tests/label/testcases/
   {bitmasks,panseg_bdd100k,panseg_coco,to_rle}` and `tests/eval/testcases`
   are small real files; `tests/test_bitmask.py` currently uses only
   synthetic arrays despite `VISION.md` claiming verification against the
   fixture. Check the toolkit's BSD-3 licence before copying.
6. **Split sanity.** Use `bdd100k/data/*.txt` counts to verify the
   70,000 / 10,000 / 20,000 claims instead of relying on the Kaggle mirror.

## 5. Rules adopted to avoid repeating the official mistakes

- Never encode information through float round-trips or colour-space
  rendering; decode shipped masks, don't re-rasterise polygons unless a task
  has no shipped masks.
- Every bit/shift constant lives in one named place with a test against a
  real fixture (no bare `<< 2`, `% 255`).
- Every adapter fails loudly on empty splits, unknown categories (count and
  report them) and missing files.
- Evaluation code states which official semantics it follows and is tested
  against official test cases where available.
- No unpinned VCS dependencies; modern pydantic/matplotlib/numpy only.
- CLIs share one argument layer, so a mode accepted in docs is accepted in
  code (official bug 1).
