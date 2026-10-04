# Object Detection

Detect 10 classes (person, rider, car, truck, bus, train, motor, bike, traffic light,
traffic sign) in the 100K-image BDD100K release, all 1280x720: 59,384 train and 10,479 valid
images (a seeded 15% split of the official train set) and the 10,000 official val images, which
are what every score here is measured on. BDD100K has no labelled test split, so the prepared
folders store the official val images under the name `test`. Boxes come from the official labels
(the 2018 JSON, or the 2020 `det_20` files if you have them); lanes and drivable-area
polygons are ignored. The models below were trained on the 2018 labels, the same ones
redistributed in the YOLO-format dataset
🤗 [dronefreak/BDD100K](https://huggingface.co/datasets/dronefreak/BDD100K) on Hugging Face.

## Model Zoo

Val split, imgsz 960, mAP@0.5:0.95 (Ultralytics' COCO-style metric).
YOLO models: COCO-pretrained, 50 epochs, SGD, cosine schedule, seed 0.

### YOLO

| Model | mAP@0.5:0.95 | mAP@0.5 | Weights | Citation |
|---|---|---|---|---|
| YOLO26s | 33.86 | 58.76 | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolo26s) | [YOLO26](CITATIONS.md#yolo26) |
| YOLOv10s | 33.35 | 57.64 | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolov10s) | [YOLOv10](CITATIONS.md#yolov10) |
| YOLOv8s | 33.25 | 57.93 | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolov8s) | [YOLOv8](CITATIONS.md#yolov8) |
| YOLO11s | 33.10 | 57.63 | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolo11s) | [YOLO11](CITATIONS.md#yolo11) |
| YOLOv9t | 29.46 | 52.04 | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolov9t) | [YOLOv9](CITATIONS.md#yolov9) |
| YOLOv10n | 29.31 | 51.96 | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolov10n) | [YOLOv10](CITATIONS.md#yolov10) |
| YOLO26n | 29.23 | 52.25 | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolo26n) | [YOLO26](CITATIONS.md#yolo26) |
| YOLOv8n | 29.09 | 51.67 | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolov8n) | [YOLOv8](CITATIONS.md#yolov8) |
| YOLO11n | 29.07 | 51.63 | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolo11n) | [YOLO11](CITATIONS.md#yolo11) |

### RF-DETR

| Model | mAP@0.5:0.95 | mAP@0.5 | Weights | Citation |
|---|---|---|---|---|
| RF-DETR Nano | 31.58 | 56.90 | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-rfdetr-nano) | [RF-DETR](CITATIONS.md#rf-detr) |

All weights: the 🤗 [BDD100K object detection model zoo](https://huggingface.co/collections/dronefreak/bdd100k-object-detection-model-zoo) collection on Hugging Face.

Papers and citations for these models: [CITATIONS.md](CITATIONS.md).

Weakest classes: `train` (15 val boxes), `traffic light` and `traffic sign` (small objects).

## Download Data

The official BDD100K website is down, so this is an unofficial redistribution on Hugging Face: the
YOLO-format dataset with the original 2018 labels (59,384 train, 10,479 valid and 10,000 val images, the last stored in a `test` folder),
🤗 [dronefreak/BDD100K](https://huggingface.co/datasets/dronefreak/BDD100K).

```bash
hf download dronefreak/BDD100K --repo-type dataset --local-dir /path/to/bdd100k
sed -i "s#^path:.*#path: /path/to/bdd100k/data#" /path/to/bdd100k/data/data.yaml
```

Its `data.yaml` has `path: .`, which Ultralytics resolves from the current directory, so the `sed`
points it at the folder. A copy of the official download (images plus `det_20` or 2018 labels) works
too, see the last Usage section.

## Usage

### Train

This is the recipe behind every YOLO model in the table (YOLO26s shown), written out in full so it does not depend on config defaults. It trains on `train`, validates on `valid` after every epoch, and evaluates the best checkpoint on the official val set (the `test` folder) at the end. The COCO-pretrained weights download
automatically.

```bash
bdd100k-train dataset=bdd100k-detection model.name=yolo26s \
    dataset.dataset_yaml=/path/to/bdd100k/data/data.yaml \
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

Other Ultralytics training arguments go through `+training.extra.<key>=<value>`
(for example `+training.extra.fraction=0.1` for a quick check on 10% of the data).

### Train RF-DETR

RF-DETR needs the extra (`pip install "bdd100k-toolkit[rfdetr]"`) and reads the COCO folder written by
`bdd100k-prepare` instead of the YOLO `data.yaml`. The same command dispatches on `model.name`
(`rfdetr-nano`, `rfdetr-small`, `rfdetr-medium`). RF-DETR squares the whole frame, so the resolution
matters: the nano baseline used 576 (small 640, medium 704), a multiple of 64.

```bash
bdd100k-train dataset=bdd100k-detection model.name=rfdetr-nano model.resolution=576 \
    dataset.dataset_dir=/path/to/bdd100k/coco_dataset \
    training.output_dir=experiments/bdd100k-detection/rfdetr-nano
```

Defaults (30 epochs, cosine, EMA, early stopping) are in `configs/config_detection_rfdetr.yaml`.
Training does not evaluate; use the evaluate command below. The checkpoint is
`checkpoint_best_total.pth`, and the resolution is read back from `training_config.json` beside it.
For a quick GPU check add `training.epochs=2 training.batch_size=4` on a small COCO subset.

### Evaluate

`--split test` selects the folder holding the official val images.

```bash
bdd100k-evaluate --dataset bdd100k-detection --model yolo26s \
    --checkpoint experiments/bdd100k-detection/yolo26s/yolo26s/weights/best.pt \
    --dataset-yaml /path/to/bdd100k/data/data.yaml --split test \
    --output-dir experiments/bdd100k-detection/yolo26s/evaluation
```

For RF-DETR, pass the COCO folder instead of `--dataset-yaml`
(`--limit N` scores only the first N images):

```bash
bdd100k-evaluate --dataset bdd100k-detection --model rfdetr-nano \
    --checkpoint experiments/bdd100k-detection/rfdetr-nano/checkpoint_best_total.pth \
    --dataset-dir /path/to/bdd100k/coco_dataset --split test
```

### Demo

```bash
python demo/detect.py --image /path/to/frame.jpg \
    --model experiments/bdd100k-detection/yolo26s/yolo26s/weights/best.pt
python demo/detect.py --video /path/to/clip.mp4 \
    --model experiments/bdd100k-detection/yolo26s/yolo26s/weights/best.pt
```

Images, folders of images, videos and folders of videos are all accepted; outputs go to
`demo/outputs/`.

### Official download instead of the Hugging Face copy

Convert it first, then train and evaluate exactly as above with `dataset.dataset_yaml` and
`--dataset-yaml` pointing at `/path/to/yolo/data.yaml`:

```bash
bdd100k-prepare --dataset bdd100k-detection --raw-dir /path/to/bdd100k_raw --output-dir /path/to/coco
bdd100k-coco-to-yolo --input-dir /path/to/coco --output-dir /path/to/yolo
```
