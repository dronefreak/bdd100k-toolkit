# Segmentation

Pixel-level labels for the 10K BDD100K images (7K train, 1K val, 2K test without public labels).

| Task | Status |
|---|---|
| Semantic segmentation (19 classes) | Zero-shot results for Cityscapes-trained models; own trainer in place, not yet run on real data |
| Instance segmentation (8 classes) | Planned |
| Panoptic segmentation (40 classes) | Planned |

## Model Zoo

Cityscapes-trained, zero-shot on BDD100K val: public checkpoints trained on Cityscapes, scored on the
1,000 BDD100K val images without any BDD training. BDD100K uses the same 19 classes in the same order, so
no remapping is needed. These are transfer results, not BDD100K-trained models, and the weights are the
original authors' (not retrained here). Every model sees the frame resized to 1024x1824 (the Cityscapes
test scale, aspect ratio kept), with single scale, no flip and fp32, and the prediction is resized back to
720x1280 for scoring. Metrics follow the Cityscapes and official BDD100K evaluators: one dataset-wide
confusion matrix, ignore pixels (255) dropped. mIoU averages the classes present in the ground truth,
category mIoU uses the 7 Cityscapes categories, and the best row is in bold.
Papers and citations: [CITATIONS.md](CITATIONS.md).

| Model | mIoU | Category mIoU | fIoU | Pixel acc | Weights |
|---|---|---|---|---|---|
| **[Mask2Former Swin-L, 2022](CITATIONS.md#mask2former)** | **57.74%** | **82.88%** | **86.90%** | **92.40%** | 🤗 [facebook/mask2former-swin-large-cityscapes-semantic](https://huggingface.co/facebook/mask2former-swin-large-cityscapes-semantic) |
| [SegFormer B5, 2021](CITATIONS.md#segformer) | 54.09% | 79.80% | 83.57% | 90.57% | 🤗 [nvidia/segformer-b5-finetuned-cityscapes-1024-1024](https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024) |
| [SegFormer B4, 2021](CITATIONS.md#segformer) | 53.94% | 79.17% | 82.72% | 90.12% | 🤗 [nvidia/segformer-b4-finetuned-cityscapes-1024-1024](https://huggingface.co/nvidia/segformer-b4-finetuned-cityscapes-1024-1024) |
| [SegFormer B3, 2021](CITATIONS.md#segformer) | 52.84% | 78.18% | 82.16% | 89.79% | 🤗 [nvidia/segformer-b3-finetuned-cityscapes-1024-1024](https://huggingface.co/nvidia/segformer-b3-finetuned-cityscapes-1024-1024) |
| [Mask2Former Swin-S, 2022](CITATIONS.md#mask2former) | 52.63% | 79.51% | 84.25% | 90.59% | 🤗 [facebook/mask2former-swin-small-cityscapes-semantic](https://huggingface.co/facebook/mask2former-swin-small-cityscapes-semantic) |
| [Mask2Former Swin-T, 2022](CITATIONS.md#mask2former) | 49.93% | 76.87% | 82.10% | 89.23% | 🤗 [facebook/mask2former-swin-tiny-cityscapes-semantic](https://huggingface.co/facebook/mask2former-swin-tiny-cityscapes-semantic) |
| [SegFormer B2, 2021](CITATIONS.md#segformer) | 47.37% | 75.51% | 79.75% | 88.02% | 🤗 [nvidia/segformer-b2-finetuned-cityscapes-1024-1024](https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024) |
| [SegFormer B1, 2021](CITATIONS.md#segformer) | 42.55% | 72.07% | 77.16% | 86.17% | 🤗 [nvidia/segformer-b1-finetuned-cityscapes-1024-1024](https://huggingface.co/nvidia/segformer-b1-finetuned-cityscapes-1024-1024) |
| [SegFormer B0, 2021](CITATIONS.md#segformer) | 41.61% | 70.25% | 75.34% | 84.89% | 🤗 [nvidia/segformer-b0-finetuned-cityscapes-1024-1024](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024) |

Mask2Former Swin-B (IN21k) is not listed: its checkpoint does not load correctly with the current
`transformers`. Per-class IoU, precision and recall, and the confusion matrix, are in each run's
`metrics.json`. The hardest classes for every model are `train` (near 0), `wall`, `bus` and `truck`.

## Download Data

The official BDD100K website is down, so there is an unofficial redistribution on Hugging Face:

| Task | Dataset |
|---|---|
| Semantic segmentation | 🤗 [dronefreak/BDD100K-Semantic-Segmentation](https://huggingface.co/datasets/dronefreak/BDD100K-Semantic-Segmentation) |

```bash
hf download dronefreak/BDD100K-Semantic-Segmentation --repo-type dataset --local-dir /path/to/bdd100k_10k
```

The adapter is not yet checked against this copy's layout. Instance and panoptic segmentation need the
`Instance Segmentation` and `Panoptic Segmentation` labels, which have no Hugging Face copy yet.

## Usage

```bash
bdd100k-prepare --dataset bdd100k-semantic-seg --raw-dir /path/to/bdd100k_10k --output-dir /path/to/out
bdd100k-train dataset=bdd100k-semantic-seg dataset.data_dir=/path/to/out   # Unet + resnet34 by default
```

### Evaluate a pretrained Hugging Face model

```bash
pip install "bdd100k-toolkit[transformers]"
bdd100k-evaluate --dataset bdd100k-semantic-seg \
    --hf-model nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
    --images-dir /path/to/images/val --masks-dir /path/to/labels/val
```

`--images-dir` holds `<id>.jpg` images and `--masks-dir` the matching `<id>_train_id.png` (or
`<id>.png`) masks (for example `seg/images/val` and `seg/labels/val` of the 10K segmentation
download). `--short-side` (default 1024) sets the input height,
`--dtype` the precision and `--limit N` evaluates only the first N images. The command refuses to run
if the checkpoint's class names differ from ours or if weights are missing from the checkpoint.
