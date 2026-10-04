# Segmentation

Pixel-level labels for the 10K BDD100K images (7K train, 1K val, 2K test without public labels).

| Task | Status |
|---|---|
| Semantic segmentation (19 classes) | Data pipeline and trainer in place; not yet run on real data |
| Instance segmentation (8 classes) | Planned |
| Panoptic segmentation (40 classes) | Planned |

## Model Zoo

No trained models yet. Semantic segmentation will be the first, once the real-data run and
the official-style mIoU evaluation are done.

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
