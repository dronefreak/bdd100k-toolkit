# Image Classification

Predict the per-image attributes of BDD100K: weather, time of day (period) and scenario.
These tasks are **unofficial**: BDD100K defines no classification benchmark, only
`attributes: {weather, timeofday, scene}` in its detection label JSON. The splits follow
three Kaggle datasets (`marquis03`, Apache-2.0), which are the same images and labels as
the official release, with `unknown` being the official `undefined`. The labels are the 2018
`attributes` in `bdd100k_labels_images_{train,val}.json`.

![BDD100K detector and weather, period and scenario classifiers running on two city drives, side by side](../../../assets/demo.gif)

| Task | Classes | Source |
|---|---|---|
| Period (time of day) | 4: daytime, night, dawn or dusk, unknown | `attributes.timeofday` |
| Weather | 7: clear, partly cloudy, overcast, rainy, snowy, foggy, unknown | `attributes.weather` |
| Scenario | 7: city street, highway, residential, parking lot, gas stations, tunnel, unknown | `attributes.scene` |

## Model Zoo

Every model trained so far, 22 in total, all on Hugging Face. Scores are on the official val set (the
`test` folder of the prepared data); macro F1 is the main metric because the classes are heavily
imbalanced, and the best macro F1 of each task is in bold. Two backends: `ultralytics` (the `-cls` YOLO
models) and `timm` (everything else). Papers and citations: [CITATIONS.md](CITATIONS.md).

### Weather

| Model | Macro F1 | Top-1 | Balanced acc | Macro precision | Weights | Citation |
|---|---|---|---|---|---|---|
| **ConvNeXt-Atto** | **67.44%** | **83.00%** | **65.25%** | **81.38%** | 🤗 [**HuggingFace**](https://huggingface.co/dronefreak/bdd100k-weather-convnext_atto) | [**ConvNeXt**](CITATIONS.md#convnext) |
| MobileNetV4-Conv-Small | 66.51% | 82.16% | 64.37% | 80.44% | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-weather-mobilenetv4_conv_small) | [MobileNetV4](CITATIONS.md#mobilenetv4) |
| EfficientViT-B0 | 65.10% | 82.79% | 63.62% | 67.11% | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-weather-efficientvit_b0) | [EfficientViT](CITATIONS.md#efficientvit) |
| ResNet-18 | 64.30% | 82.19% | 62.97% | 66.08% | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-weather-resnet18) | [ResNet](CITATIONS.md#resnet) |
| YOLO26n-cls | 64.09% | 82.04% | 62.46% | 66.57% | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-weather-yolo26n-cls) | [YOLO26](CITATIONS.md#yolo26) |
| YOLOv8n-cls | 63.07% | 81.22% | 61.42% | 65.52% | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-weather-yolov8n-cls) | [YOLOv8](CITATIONS.md#yolov8) |
| YOLO11n-cls | 62.91% | 81.32% | 61.19% | 65.68% | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-weather-yolo11n-cls) | [YOLO11](CITATIONS.md#yolo11) |

### Period (time of day)

| Model | Macro F1 | Top-1 | Balanced acc | Macro precision | Weights | Citation |
|---|---|---|---|---|---|---|
| **EfficientViT-B0** | **80.98%** | **93.71%** | **77.14%** | **86.44%** | 🤗 [**HuggingFace**](https://huggingface.co/dronefreak/bdd100k-period-efficientvit_b0) | [**EfficientViT**](CITATIONS.md#efficientvit) |
| YOLOv8n-cls | 80.85% | 93.56% | 76.26% | 87.87% | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-period-yolov8n-cls) | [YOLOv8](CITATIONS.md#yolov8) |
| ConvNeXt-Atto | 80.75% | 93.95% | 76.82% | 86.39% | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-period-convnext_atto) | [ConvNeXt](CITATIONS.md#convnext) |
| ResNet-18 | 80.50% | 93.32% | 76.72% | 85.87% | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-period-resnet18) | [ResNet](CITATIONS.md#resnet) |
| YOLO26n-cls | 80.41% | 93.61% | 77.72% | 83.88% | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-period-yolo26n-cls) | [YOLO26](CITATIONS.md#yolo26) |
| MobileNetV4-Conv-Small | 80.22% | 93.57% | 76.22% | 86.06% | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-period-mobilenetv4_conv_small) | [MobileNetV4](CITATIONS.md#mobilenetv4) |
| YOLO11n-cls | 80.13% | 93.79% | 75.19% | 88.20% | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-period-yolo11n-cls) | [YOLO11](CITATIONS.md#yolo11) |

### Scenario

| Model | Macro F1 | Top-1 | Balanced acc | Macro precision | Weights | Citation |
|---|---|---|---|---|---|---|
| **ConvNeXt-Atto** | **61.06%** | **77.44%** | **60.34%** | **67.92%** | 🤗 [**HuggingFace**](https://huggingface.co/dronefreak/bdd100k-scenario-convnext_atto) | [**ConvNeXt**](CITATIONS.md#convnext) |
| ConvNeXt-Tiny | 56.88% | 76.69% | 59.82% | 54.60% | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-scenario-convnext_tiny) | [ConvNeXt](CITATIONS.md#convnext) |
| MobileNetV4-Conv-Small | 52.89% | 78.20% | 48.69% | 61.41% | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-scenario-mobilenetv4_conv_small) | [MobileNetV4](CITATIONS.md#mobilenetv4) |
| YOLO11n-cls | 49.47% | 78.58% | 46.05% | 60.98% | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-scenario-yolo11n-cls) | [YOLO11](CITATIONS.md#yolo11) |
| YOLO26n-cls | 48.33% | 76.78% | 43.94% | 58.99% | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-scenario-yolo26n-cls) | [YOLO26](CITATIONS.md#yolo26) |
| ResNet-18 | 47.29% | 77.14% | 44.83% | 54.25% | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-scenario-resnet18) | [ResNet](CITATIONS.md#resnet) |
| YOLOv8n-cls | 46.40% | 76.13% | 43.26% | 54.14% | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-scenario-yolov8n-cls) | [YOLOv8](CITATIONS.md#yolov8) |
| EfficientViT-B0 | 46.10% | 75.94% | 42.64% | 53.74% | 🤗 [HuggingFace](https://huggingface.co/dronefreak/bdd100k-scenario-efficientvit_b0) | [EfficientViT](CITATIONS.md#efficientvit) |

All models are plain baseline runs (224 px input, the same recipe), except two scenario models: ConvNeXt-Atto
was retrained with a class-weighted loss at 512 px input, and ConvNeXt-Tiny with a class-weighted loss at
224 px. The differences between the small baseline models are within noise.

## Download Data

The official BDD100K website is down, so these are unofficial redistributions on Hugging Face (official
train and val images, 2018 labels, no test split; the official val set becomes `test`). Download the
dataset for the task you need:

| Task | Dataset |
|---|---|
| Weather | 🤗 [dronefreak/BDD100K-Weather-Classification](https://huggingface.co/datasets/dronefreak/BDD100K-Weather-Classification) |
| Period | 🤗 [dronefreak/BDD100K-Period-Classification](https://huggingface.co/datasets/dronefreak/BDD100K-Period-Classification) |
| Scenario | 🤗 [dronefreak/BDD100K-Scenario-Classification](https://huggingface.co/datasets/dronefreak/BDD100K-Scenario-Classification) |

```bash
hf download dronefreak/BDD100K-Weather-Classification --repo-type dataset --local-dir /path/to/weather
```

`bdd100k-prepare` detects the layout itself, so a copy of the official download (images plus label
JSON) and the Kaggle class folders work too. All three give the same `{train,valid,test}/<class>/*.jpg` output.

## Usage

```bash
bdd100k-prepare  --dataset bdd100k-weather --raw-dir /path/to/weather --output-dir /path/to/out
bdd100k-train    dataset=bdd100k-weather model.name=yolo11n-cls dataset.data_dir=/path/to/out
bdd100k-evaluate --dataset bdd100k-weather --checkpoint <run>/weights/best.pt --data-dir /path/to/out

# timm backend (pip install "bdd100k-toolkit[timm]"), with class balancing
bdd100k-train dataset=bdd100k-weather model.backend=timm model.name=convnext_atto \
    +training.extra.balance=loss dataset.data_dir=/path/to/out

# demo on an image, a folder or a video
python demo/classify.py --image frame.jpg --model weather=<run>/weights/best.pt
```

The same commands work for `bdd100k-period` and `bdd100k-scenario`.
