# Object Detection

Detect 10 classes (person, rider, car, truck, bus, train, motor, bike, traffic light,
traffic sign) in the 100K-image BDD100K release: 59,384 train, 10,479 valid and 10,000
test images (the official val set), all 1280x720. Boxes come from the official labels
(the 2018 JSON, or the 2020 `det_20` files if you have them); lanes and drivable-area
polygons are ignored. The models below were trained on the 2018 labels, the same ones
redistributed in the YOLO-format dataset
[dronefreak/BDD100K](https://huggingface.co/datasets/dronefreak/BDD100K) on Hugging Face.

## Model Zoo

Test split, imgsz 960, mAP@0.5:0.95 (Ultralytics' COCO-style metric). Every YOLO model:
COCO-pretrained, 50 epochs, SGD, cosine schedule, seed 0.

| Model | mAP@0.5:0.95 | mAP@0.5 | Weights |
|---|---|---|---|
| YOLO26s | 33.86 | 58.76 | [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolo26s) |
| YOLOv10s | 33.35 | 57.64 | [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolov10s) |
| YOLOv8s | 33.25 | 57.93 | [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolov8s) |
| YOLO11s | 33.10 | 57.63 | [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolo11s) |
| RF-DETR Nano | 31.58 | 56.90 | [HuggingFace](https://huggingface.co/dronefreak/bdd100k-rfdetr-nano) |
| YOLOv9t | 29.46 | 52.04 | [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolov9t) |
| YOLOv10n | 29.31 | 51.96 | [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolov10n) |
| YOLO26n | 29.23 | 52.25 | [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolo26n) |
| YOLOv8n | 29.09 | 51.67 | [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolov8n) |
| YOLO11n | 29.07 | 51.63 | [HuggingFace](https://huggingface.co/dronefreak/bdd100k-yolo11n) |

All weights: the [BDD100K object detection model zoo](https://huggingface.co/collections/dronefreak/bdd100k-object-detection-model-zoo) collection on Hugging Face.

Weakest classes: `train` (15 test boxes), `traffic light` and `traffic sign` (small objects).

## Usage

Easiest input: the YOLO-format Hugging Face dataset, which needs no prepare step. Its `data.yaml`
has `path: .`, which Ultralytics resolves from the current directory, so point it at the folder first:

```bash
hf download dronefreak/BDD100K --repo-type dataset --local-dir /path/to/bdd100k
sed -i "s#^path:.*#path: /path/to/bdd100k/data#" /path/to/bdd100k/data/data.yaml
bdd100k-train dataset=bdd100k-detection model.name=yolo11n \
    dataset.dataset_yaml=/path/to/bdd100k/data/data.yaml
```

Or start from the official download (also accepts the 2020 `det_20` labels):

```bash
bdd100k-prepare --dataset bdd100k-detection --raw-dir /path/to/bdd100k --output-dir /path/to/coco
bdd100k-coco-to-yolo --input-dir /path/to/coco --output-dir /path/to/yolo
bdd100k-train dataset=bdd100k-detection model.name=yolo11n dataset.dataset_yaml=/path/to/yolo/data.yaml
bdd100k-evaluate --dataset bdd100k-detection --checkpoint <run>/weights/best.pt \
    --dataset-yaml /path/to/yolo/data.yaml
```

Defaults (50 epochs, imgsz 960, SGD, cosine) are in `configs/config_detection.yaml`; extra
Ultralytics arguments go through `+training.extra.<key>=<value>`.
