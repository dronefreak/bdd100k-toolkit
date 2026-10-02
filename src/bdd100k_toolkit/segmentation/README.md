# Segmentation

Pixel-level labels for the 10K BDD100K images (7K train, 1K val, 2K test).

| Task | Status |
|---|---|
| Semantic segmentation (19 classes) | Data pipeline and trainer in place; not yet run on real data |
| Instance segmentation (8 classes) | Planned |
| Panoptic segmentation (40 classes) | Planned |

## Model Zoo

No trained models yet. Semantic segmentation will be the first, once the real-data run and
the official-style mIoU evaluation are done.

## Usage

```bash
bdd100k-prepare --dataset bdd100k-semantic-seg --raw-dir /path/to/bdd100k_10k --output-dir /path/to/out
bdd100k-train dataset=bdd100k-semantic-seg dataset.data_dir=/path/to/out   # Unet + resnet34 by default
```
