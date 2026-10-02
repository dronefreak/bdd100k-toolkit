# Reference notes from the old BDD100K model zoo

Source: [`SysCV/bdd100k-models`](https://github.com/SysCV/bdd100k-models), the
official model zoo (read from a local clone, last commit November 2022). It is
an unmaintained collection of about 360 OpenMMLab 1.x configs (mmcls, mmdet,
mmseg) plus weights on `dl.cv.ethz.ch`, covering nine tasks. Its code cannot be
reused (old mmcv stack, unpinned `git+git://` scalabel dependency), so this file
keeps only three things worth having: what data each task needs, the baseline
scores it reported, and where its coverage was thin.

All scores below are copied from its READMEs, not re-run. They come from
OpenMMLab-era models trained on 8 GPUs, so treat them as order-of-magnitude
reference points. Our numbers become comparable only once the official-semantics
evaluators of Phase 3 exist (see `ROADMAP.md`).

## 1. Task coverage and data needs

| Task | Images (official download) | Labels (official download) | Classes | Split (train/val/test) |
|---|---|---|---|---|
| Image tagging | 100K Images | Detection 2020 Labels | weather 6, scene 6, time of day 3 (plus undefined) | 70K / 10K / 20K |
| Object detection | 100K Images | Detection 2020 Labels | 10 | 70K / 10K / 20K |
| Drivable area | 100K Images | Drivable Area | 3 | 70K / 10K / 20K |
| Semantic segmentation | 10K Images | Semantic Segmentation | 19 | 7K / 1K / 2K |
| Instance segmentation | 10K Images | Instance Segmentation | 8 | 7K / 1K / 2K |
| Panoptic segmentation | 10K Images | Panoptic Segmentation | 40 (10 thing, 30 stuff) | 7K / 1K / 2K |
| Pose estimation | 100K Images | Pose Estimation Labels | 18 keypoints | 10K / 1.5K / 2.5K of 14K |
| Box tracking (MOT) | MOT 2020 Images | MOT 2020 Labels | 8 | 1.4K / 200 / 400 videos |
| Segmentation tracking (MOTS) | MOTS 2020 Images | MOTS 2020 Labels | 8 | 154 / 32 / 37 videos |

Notes:

- Only train and val labels are public. The old zoo also reports test scores,
  which we cannot reproduce locally, so our comparisons use val.
- Tracking videos are about 40 s each, annotated at 5 fps (around 200 frames).
- The old zoo expects this folder layout (useful as a checklist of what a raw
  download contains):
  `images/{100k,10k,track,seg_track_20}/{train,val,test}` and
  `labels/{det_20,pose_21,ins_seg,sem_seg,pan_seg,drivable,box_track_20,seg_track_20}`.
  Segmentation labels ship as `bitmasks` or `masks`, `colormaps`, `polygons` and
  `rles`; detection and pose ship as one JSON per split (`det_train.json`,
  `pose_train.json`).
- Its conversion to COCO relied on the official `bdd100k.label.to_coco` (which
  pulls in scalabel). We write our own adapters instead, so only the layout and
  the label formats above matter to us.
- Where we stand: classification is done (see `docs/data-notes.md`); detection
  and semantic segmentation have adapters tested on fixtures only; the rest are
  not started.

## 2. Baseline numbers (reported by the old zoo)

Metrics are the ones the old READMEs use: Box AP and Mask AP (COCO style),
mIoU, PQ, Pose AP, mMOTA / mMOTSA with mIDF1. `val` / `test` are given as
reported. `MS` means multi-scale training.

| Task | Models in zoo | Best reported (val / test) | Reference points (val / test) |
|---|---:|---|---|
| Detection (Box AP) | 64 | ConvNeXt-S, 3x, MS: 36.11 / 35.43 | Faster R-CNN R-50-FPN 1x: 31.04 / 29.78; 3x MS: 32.30 / 31.45. Cascade R-CNN R-50-FPN 3x: 33.72 / 33.07. RetinaNet R-50-FPN 3x: 30.91 / 30.21. Lowest: FCOS R-50-FPN 26.59 / 24.76 |
| Drivable area (mIoU) | 56 | DeepLabV3 R-101-D8, 80K, 512x1024: 85.15 / 84.75 | DeepLabV3+ R-50-D8, 80K: 84.35 / 83.93. PSPNet R-50-D8, 80K: 83.67 / 83.43. Lowest: FCN R-50-D8, 40K, 769x769: 74.84 / 74.42 |
| Semantic seg (mIoU) | 53 | ConvNeXt-B, 80K, 512x1024: 67.26 / 59.82 | DeepLabV3+ R-50-D8, 80K: 63.96 / 56.08. PSPNet R-50-D8, 80K: 62.03 / 54.99. Lowest: Semantic FPN R-50-FPN, 40K: 59.24 / 52.89 |
| Instance seg (Mask AP) | 26 | Mask R-CNN HRNet-w40, 3x, MS: 22.57 / 19.38 | Mask R-CNN R-50-FPN 1x: 16.24 / 14.86; 3x MS: 19.88 / 17.46 |
| Panoptic seg (PQ) | 5 | Panoptic FPN R-101-FPN, 5x, MS: 23.90 / 22.50 | Panoptic FPN R-50-FPN 5x MS: 23.38 / 22.04 |
| Pose (Pose AP) | 26 | HRNet-w48, 256x192: 50.32 / 47.36 | ResNet-50, 256x192: 46.15 / 43.73. Lowest: MobileNetV2 43.82 / 41.02 |
| MOT | 1 | QDTrack R-50: mMOTA 36.6 / 35.7, mIDF1 51.6 / 52.3, ID switches 6193 / 10822 | none |
| MOTS | 1 | PCAN R-50: mMOTSA 28.1 / 31.9, mIDF1 45.4 / 50.4, ID switches 874 / 845 | none |

Image tagging (accuracy only, six weather classes plus undefined; the old zoo
has no time-of-day models and no per-class numbers):

| Task | Models | Best reported (val / test) | Notes |
|---|---:|---|---|
| Weather | 14 | ResNet-50, 640x640: 81.94 / 81.56 | VGG, ResNet, DLA at 224 and 640; all within about 2 points |
| Scene | 14 | ResNet-18, 640x640: 78.07 / 77.48 | same spread |

Our own classification results (macro F1 on the official val split as test,
`unknown` kept as a class) are not directly comparable to those accuracies:
different split, one extra class, a different metric. See
`docs/classification-results.md`.

Observations that matter for planning:

- Detection: the whole zoo spans only about 10 AP (26.6 to 36.1). Backbone
  matters more than the detector family.
- Semantic segmentation: a large val-to-test gap (over 7 mIoU for ConvNeXt-B),
  so val-only numbers will look better than test would.
- Instance segmentation and panoptic scores are low in absolute terms
  (Mask AP about 22, PQ about 24), which suggests the 10K-image tasks are hard
  and data-limited.

## 3. Gaps worth filling later

The old zoo is thinnest where a modern toolkit would add the most value.

| Task | What the old zoo had | Why it is a good gap |
|---|---|---|
| Panoptic segmentation | 5 models, a single family (Panoptic FPN) | No modern models, and the PQ evaluator in the official toolkit is hand-written, so a clean one is useful |
| MOT (box tracking) | 1 model (QDTrack, R-50) | A single tracker, an unclear label-availability situation (bdd100k/bdd100k#369), and a `motmetrics` evaluator is easy to add |
| MOTS (segmentation tracking) | 1 model (PCAN, R-50) | Same as MOT, plus the smallest dataset (223 videos) |
| Pose estimation | 26 models, all top-down: ResNet, MobileNetV2, HRNet and its DARK and UDP variants (best about 50 Pose AP) | Many models but all old; a modern top-down baseline would be a real step up |

Rough order if we pick them up: panoptic (reuses the bitmask decoder and the
segmentation evaluators), then MOT, then MOTS (reuses MOT), then pose.
