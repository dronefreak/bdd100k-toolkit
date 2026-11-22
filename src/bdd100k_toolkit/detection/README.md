# Object Detection

Detect 10 classes (person, rider, car, truck, bus, train, motor, bike, traffic light,
traffic sign) in the 100K-image BDD100K release, all 1280x720, on the official splits: 70,000 `train`,
10,000 `valid` (the official val) and 20,000 `test` images. Boxes come from the official labels:
the per-image JSON files of `bdd100k_labels.zip`, which cover all three splits, or the single-file
2018 or 2020 `det_20` labels, which have no test labels (then only `train` and `valid` are written).
Lanes and drivable-area polygons are ignored. The dataset is redistributed, as received, as
🤗 [dronefreak/BDD100K](https://huggingface.co/datasets/dronefreak/BDD100K) on Hugging Face, and
`bdd100k-prepare` converts it.

The scores in the Model Zoo predate this layout: those models were trained on 59,384 images (a seeded
85% of the official train set), selected on the other 10,479, and scored on the official val images,
which the older layout stored in a folder called `test` (they are the `valid` folder now).

## Model Zoo

Val split, imgsz 960, mAP@0.5:0.95 (Ultralytics' COCO-style metric).
YOLO models: COCO-pretrained, 50 epochs, SGD, cosine schedule, seed 0.

### YOLO

| Model | mAP@0.5:0.95 | mAP@0.5 | Weights |
|---|---|---|---|
| [YOLO26s, 2026](CITATIONS.md#yolo26) | 33.86 | 58.76 | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolo26s) |
| [YOLOv10s, 2024](CITATIONS.md#yolov10) | 33.35 | 57.64 | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolov10s) |
| [YOLOv8s, 2023](CITATIONS.md#yolov8) | 33.25 | 57.93 | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolov8s) |
| [YOLO11s, 2024](CITATIONS.md#yolo11) | 33.10 | 57.63 | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolo11s) |
| [YOLOv9t, 2024](CITATIONS.md#yolov9) | 29.46 | 52.04 | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolov9t) |
| [YOLOv10n, 2024](CITATIONS.md#yolov10) | 29.31 | 51.96 | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolov10n) |
| [YOLO26n, 2026](CITATIONS.md#yolo26) | 29.23 | 52.25 | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolo26n) |
| [YOLOv8n, 2023](CITATIONS.md#yolov8) | 29.09 | 51.67 | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolov8n) |
| [YOLO11n, 2024](CITATIONS.md#yolo11) | 29.07 | 51.63 | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolo11n) |

### RF-DETR

| Model | mAP@0.5:0.95 | mAP@0.5 | Weights |
|---|---|---|---|
| [RF-DETR Nano, 2026](CITATIONS.md#rf-detr) | 31.58 | 56.90 | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-rfdetr-nano) |

All weights: the 🤗 [BDD100K object detection model zoo](https://huggingface.co/collections/dronefreak/bdd100k-object-detection-model-zoo) collection on Hugging Face.

Papers and citations for these models: [CITATIONS.md](CITATIONS.md).

Weakest classes: `train` (15 val boxes), `traffic light` and `traffic sign` (small objects).

## Download Data

The official BDD100K website is down (the original zips are still on a [Berkeley file server](http://128.32.162.150/bdd100k/)), and this is an unofficial redistribution on Hugging Face: the
100K images and per-image label JSON files exactly as the Berkeley packages ship them, on the
official splits (70,000 train, 10,000 val and 20,000 test images),
🤗 [dronefreak/BDD100K](https://huggingface.co/datasets/dronefreak/BDD100K). Each folder holds at
most 8,000 files, so the splits are sharded:

```text
data/images_100k/100k/{train,val,test}/shard_NNN/<id>.jpg
data/labels/100k/{train,val,test}/shard_NNN/<id>.json
```

```bash
hf download dronefreak/BDD100K --repo-type dataset --local-dir /path/to/bdd100k_raw
```

The `metadata.jsonl` next to the images is a table view of the same labels for the Hub's dataset
viewer and `datasets.load_dataset("dronefreak/BDD100K")`; `bdd100k-prepare` reads the label JSON
and ignores it.

Convert it to the COCO and YOLO folders the training commands below read (the same command also
accepts the unpacked Berkeley zips, `images/100k` + `labels`, and the single-file 2018 or `det_20`
label JSON):

```bash
bdd100k-prepare --dataset bdd100k-detection --raw-dir /path/to/bdd100k_raw --output-dir /path/to/coco
bdd100k-coco-to-yolo --input-dir /path/to/coco --output-dir /path/to/yolo
```

The YOLO folder has the `data.yaml` used below (`/path/to/yolo/data.yaml`).

## Usage

### Train

This is the recipe behind every YOLO model in the table (YOLO26s shown), written out in full so it does not depend on config defaults. It trains on `train`, selects the best checkpoint on `valid` after every epoch, and evaluates it on `test` at the end. The COCO-pretrained weights download
automatically.

```bash
bdd100k-train dataset=bdd100k-detection model.name=yolo26s \
    dataset.dataset_yaml=/path/to/yolo/data.yaml \
    training.output_dir=experiments/bdd100k-detection/yolo26s \
    training.epochs=50 training.imgsz=960 training.batch_size=-1 \
    training.optimizer=SGD training.lr=0.01 training.cos_lr=true training.use_amp=true \
    training.patience=10 training.workers=8 training.device=cuda \
    training.extra.warmup_epochs=1.0 training.extra.close_mosaic=10 +training.extra.seed=0
```

`batch_size=-1` is Ultralytics AutoBatch (about 60% of GPU memory). Results land in
`experiments/bdd100k-detection/yolo26s/`: the checkpoint is `yolo26s/weights/best.pt` and the val
metrics are in `evaluation/metrics.json`. Any other model in the table works by changing
`model.name` and `training.output_dir` (`yolov8n`, `yolov8s`, `yolov9t`, `yolov10n`, `yolov10s`,
`yolo11n`, `yolo11s`, `yolo26n`, `yolo26s`)

`scripts/train_detection.sh` (nano tier) and `scripts/train_detection_small.sh` (small tier) run whole
suites with these batch sizes, the largest measured to fit a 12 GB GPU with headroom (they came from
DetectionBench; `BATCH=N` overrides):

| Tier | YOLO batch (960 px) | RF-DETR resolution | RF-DETR batch x accumulation |
|---|---|---|---|
| nano (`n`, `t`) | 24 | 576 | 8 x 2 = 16 |
| small (`s`) | 12 | 640 | 6 x 3 = 18 |
| medium (`m`) | 6 | 704 | 4 x 4 = 16 |

The RF-DETR numbers are passed as `training.batch_size=8 training.grad_accum_steps=2` and so on (the
config default is `auto`).

Other Ultralytics training arguments go through `+training.extra.<key>=<value>`
(for example `+training.extra.fraction=0.1` for a quick check on 10% of the data).

### Train RF-DETR

RF-DETR needs the extra (`pip install -e ".[rfdetr]"`) and reads the COCO folder written by
`bdd100k-prepare` instead of the YOLO `data.yaml`. The same command dispatches on `model.name`
(`rfdetr-nano`, `rfdetr-small`, `rfdetr-medium`). RF-DETR squares the whole frame, so the resolution
matters: the nano baseline used 576 (small 640, medium 704), a multiple of 64.

```bash
bdd100k-train dataset=bdd100k-detection model.name=rfdetr-nano model.resolution=576 \
    dataset.dataset_dir=/path/to/coco \
    training.output_dir=experiments/bdd100k-detection/rfdetr-nano
```

Defaults (30 epochs, cosine, EMA, early stopping) are in `configs/config_detection_rfdetr.yaml`.
Training does not evaluate; use the evaluate command below. The checkpoint is
`checkpoint_best_total.pth`, and the resolution is read back from `training_config.json` beside it.
For a quick GPU check add `training.epochs=2 training.batch_size=4` on a small COCO subset.

### Evaluate

`--split test` (the default) scores the official test images; `--split valid` scores the official val images.

```bash
bdd100k-evaluate --dataset bdd100k-detection --model yolo26s \
    --checkpoint experiments/bdd100k-detection/yolo26s/yolo26s/weights/best.pt \
    --dataset-yaml /path/to/yolo/data.yaml --split test \
    --output-dir experiments/bdd100k-detection/yolo26s/evaluation
```

For RF-DETR, pass the COCO folder instead of `--dataset-yaml`
(`--limit N` scores only the first N images):

```bash
bdd100k-evaluate --dataset bdd100k-detection --model rfdetr-nano \
    --checkpoint experiments/bdd100k-detection/rfdetr-nano/checkpoint_best_total.pth \
    --dataset-dir /path/to/coco --split test
```

### Demo

```bash
python demo/detect.py --image /path/to/frame.jpg \
    --model experiments/bdd100k-detection/yolo26s/yolo26s/weights/best.pt
python demo/detect.py --video /path/to/clip.mp4 \
    --model experiments/bdd100k-detection/yolo26s/yolo26s/weights/best.pt
```

Images, folders of images, videos and folders of videos are all accepted; outputs go to
`demo/outputs/`. `--model` also takes an RF-DETR `checkpoint_best_total.pth` (with its
`training_config.json` beside it, and the `rfdetr` extra installed).
