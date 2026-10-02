# Image Classification

Predict the per-image attributes of BDD100K: weather, time of day (period) and scenario.
These tasks are **unofficial**: BDD100K defines no classification benchmark, only
`attributes: {weather, timeofday, scene}` in its detection label JSON. The splits follow
three Kaggle datasets (`marquis03`, Apache-2.0), which are the same images and labels as
the official release, with `unknown` being the official `undefined`. The labels are the 2018
`attributes` in `bdd100k_labels_images_{train,val}.json`.

![BDD100K detector and weather, period and scenario classifiers running on two city drives, side by side](../../../docs/assets/demo.gif)

| Task | Classes | Source |
|---|---|---|
| Period (time of day) | 4: daytime, night, dawn or dusk, unknown | `attributes.timeofday` |
| Weather | 7: clear, partly cloudy, overcast, rainy, snowy, foggy, unknown | `attributes.weather` |
| Scenario | 7: city street, highway, residential, parking lot, gas stations, tunnel, unknown | `attributes.scene` |

## Model Zoo

Test split (the official val set), macro F1 as the main metric because the classes are
heavily imbalanced. Two backends: `ultralytics` (YOLO classifiers) and `timm` (any timm model).

| Task | Model | Macro F1 | Top-1 | Balanced acc | Weights |
|---|---|---|---|---|---|
| Weather | ConvNeXt-Atto | 67.44% | 83.00% | 65.25% | [HuggingFace](https://huggingface.co/dronefreak/bdd100k-weather-convnext_atto) |
| Period | EfficientViT-B0 | 80.98% | 93.71% | 77.14% | [HuggingFace](https://huggingface.co/dronefreak/bdd100k-period-efficientvit_b0) |
| Scenario | ConvNeXt-Atto | 61.06% | 77.44% | 60.34% | [HuggingFace](https://huggingface.co/dronefreak/bdd100k-scenario-convnext_atto) |

Seven models per task were compared, plus class balancing, bigger models and higher
resolution; the differences between the small models are within noise.

## Download Data

The official BDD100K website is down, so these are unofficial redistributions on Hugging Face (official
train and val images, 2018 labels, no test split; the official val set becomes `test`). Download the
dataset for the task you need:

| Task | Dataset |
|---|---|
| Weather | [dronefreak/BDD100K-Weather-Classification](https://huggingface.co/datasets/dronefreak/BDD100K-Weather-Classification) |
| Period | [dronefreak/BDD100K-Period-Classification](https://huggingface.co/datasets/dronefreak/BDD100K-Period-Classification) |
| Scenario | [dronefreak/BDD100K-Scenario-Classification](https://huggingface.co/datasets/dronefreak/BDD100K-Scenario-Classification) |

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
