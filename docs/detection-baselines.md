# Detection baselines (BDD100K, 10 classes)

Nine Ultralytics YOLO models trained earlier in DetectionBench, evaluated here with
this toolkit's own pipeline (`bdd100k-prepare`, `bdd100k-coco-to-yolo`,
`bdd100k-evaluate`) on the official 10,000-image val set. BDD100K has no labelled test
split, so the prepared folders store these images under the name `test`.

## Reproduction check

Our evaluator reproduces DetectionBench's reported numbers: across all nine models the
largest difference in mAP@0.5 or mAP@0.5:0.95 is 0.00004 (precision and recall differ
by at most 0.002). So the prepared data, the YOLO bridge and the evaluation here match
what those models were trained and scored on. The val images are identical in both
projects; the train/valid holdout differs (random shuffle there, sorted-name holdout
here), which does not affect the val scores.

## Results (official val set, 10,000 images, imgsz 960)

| model | mAP@0.5:0.95 | mAP@0.5 | precision | recall |
|---|---:|---:|---:|---:|
| YOLO26s | 0.3386 | 0.5876 | 0.7542 | 0.5313 |
| YOLOv10s | 0.3335 | 0.5764 | 0.7504 | 0.5213 |
| YOLOv8s | 0.3325 | 0.5793 | 0.7538 | 0.5163 |
| YOLO11s | 0.3310 | 0.5763 | 0.7449 | 0.5238 |
| YOLOv9t | 0.2946 | 0.5204 | 0.7134 | 0.4675 |
| YOLOv10n | 0.2931 | 0.5196 | 0.7160 | 0.4662 |
| YOLO26n | 0.2923 | 0.5225 | 0.7245 | 0.4695 |
| YOLOv8n | 0.2909 | 0.5167 | 0.7085 | 0.4668 |
| YOLO11n | 0.2907 | 0.5163 | 0.7168 | 0.4634 |

Per-class mAP@0.5:0.95:

| model | person | rider | car | truck | bus | train | motor | bike | traffic light | traffic sign |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO26s | 0.362 | 0.259 | 0.518 | 0.499 | 0.525 | 0.001 | 0.261 | 0.271 | 0.282 | 0.408 |
| YOLOv10s | 0.358 | 0.265 | 0.517 | 0.496 | 0.516 | 0.000 | 0.241 | 0.266 | 0.275 | 0.400 |
| YOLOv8s | 0.358 | 0.258 | 0.516 | 0.498 | 0.514 | 0.001 | 0.236 | 0.269 | 0.274 | 0.399 |
| YOLO11s | 0.354 | 0.251 | 0.516 | 0.495 | 0.513 | 0.001 | 0.240 | 0.266 | 0.275 | 0.399 |
| YOLOv9t | 0.312 | 0.215 | 0.494 | 0.454 | 0.468 | 0.000 | 0.184 | 0.217 | 0.245 | 0.358 |
| YOLOv10n | 0.304 | 0.215 | 0.491 | 0.447 | 0.466 | 0.000 | 0.189 | 0.216 | 0.246 | 0.358 |
| YOLO26n | 0.306 | 0.202 | 0.491 | 0.449 | 0.466 | 0.000 | 0.196 | 0.200 | 0.251 | 0.362 |
| YOLOv8n | 0.308 | 0.207 | 0.493 | 0.448 | 0.466 | 0.000 | 0.176 | 0.207 | 0.246 | 0.359 |
| YOLO11n | 0.303 | 0.204 | 0.491 | 0.444 | 0.461 | 0.000 | 0.194 | 0.208 | 0.246 | 0.357 |

## Recipe (identical for every model, and the defaults of `config_detection.yaml`)

COCO-pretrained weights, 50 epochs, imgsz 960, SGD with lr0 0.01, cosine schedule,
AutoBatch, 1 warmup epoch, `close_mosaic` 10, AMP, patience 10, seed 0, train split
only (59,384 images).

Checkpoints and logs live in DetectionBench under `experiments/bdd100k/<model>/`.
Hugging Face model cards exist for all of them, under `dronefreak/bdd100k-<model>`
(collection: 🤗 <https://huggingface.co/collections/dronefreak/bdd100k-object-detection-model-zoo>),
including RF-DETR Nano (mAP@0.5 56.9, mAP@0.5:0.95 31.58, as reported in DetectionBench).
The RF-DETR run is not re-evaluated here because this toolkit has no RF-DETR backend.

## What the numbers say

- Size matters, family barely does: the four `s` models land at 0.331 to 0.339, the
  five nano-size models at 0.291 to 0.295. YOLO26s is best, and the spread inside
  each size class is smaller than the gap between classes.
- Rare and small classes are the weak spot. `train` has about 0.001 AP (15 val
  boxes, 113 training boxes); `traffic light` (0.28) and `traffic sign` (0.41) trail
  `car` (0.52), `bus` and `truck` (about 0.5). Together lights and signs are a third
  of all boxes and are mostly tiny.
- Every run was still improving at the end (best epoch 41 to 50 of 50), so a longer
  schedule would probably help.

## Comparing with the old official zoo

`docs/reference-bdd100k-models.md` lists Box AP of 29.8 to 31.9 (Faster R-CNN R-50-FPN)
and 35.4 (best, ConvNeXt-S) on that zoo's test split. Our 29.1 to 33.9 on val is in the same range, but the
metrics are not identical: Ultralytics' COCO-style mAP here, against the official
BDD evaluation (with ignore regions) there. Treat the comparison as a rough
reference until the Phase 3 evaluator exists.

## Leads for improving

- Longer training (100 epochs) and higher resolution (1280, native width) for the
  small-object classes.
- Larger models (`m`, `l`) if GPU time allows; going from nano to `s` is worth about
  4 mAP points.
- Class-aware tricks for `train`, `rider` and `motor` (rare, low AP).
- Replace Ultralytics' metric with the official BDD one (Phase 3).
