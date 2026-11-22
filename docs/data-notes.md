# Data notes

What the toolkit finds in the real BDD100K downloads, and how it turns them into
splits. Checked against the Berkeley zips and the Hugging Face copies.

## Sources and layouts

`bdd100k-prepare` detects the layout itself. Each row is one way to get the data.

| Source | Images | Labels |
|---|---|---|
| Hugging Face `dronefreak/BDD100K` | `data/images_100k/100k/{train,val,test}/shard_NNN/` | `data/labels/100k/{train,val,test}/shard_NNN/<id>.json` |
| Berkeley zips, unpacked as below | `images/100k/{train,val,test}/` | `labels/100k/{train,val,test}/<id>.json` |
| 2018 label download | `images/100k/{train,val}/` | `labels/bdd100k_labels_images_{train,val}.json` |
| 2020 detection labels | `images/100k/{train,val}/` | `labels/det_20/det_{train,val}.json` |
| Hugging Face classification repos | `data/images/{train,val,test}/shard_NNN/` | `metadata.jsonl` per shard, raw values (`undefined`, `dawn/dusk`) |
| Kaggle class folders | `{train,val}/<class>/*.jpg` | the folder name |

The Berkeley zips have no top-level folder of their own, so unpack each into its
own folder:

```bash
unzip bdd100k_images_100k.zip -d raw/images   # raw/images/100k/...
unzip bdd100k_labels.zip      -d raw/labels   # raw/labels/100k/...
bdd100k-prepare --dataset bdd100k-detection --raw-dir raw --output-dir coco

unzip bdd100k_images_10k.zip  -d seg/images_10k   # seg/images_10k/10k/...
unzip bdd100k_seg_maps.zip    -d seg/seg_maps     # seg/seg_maps/labels/...
bdd100k-prepare --dataset bdd100k-semantic-seg --raw-dir seg --output-dir seg_out
```

`--labels-dir` points at the label folder when it is not `<raw-dir>/labels`. The
segmentation adapter also accepts `images/10k` + `labels/sem_seg/masks` (the later
label release) and `seg/images` + `seg/labels` (the older `bdd100k_seg.zip`).

## Splits

All tasks use the official splits, with nothing carved out of `train`:

| Official split | Written as | 100K tasks (detection, classification) | 10K task (semantic seg) |
|---|---|---:|---:|
| `train` | `train` | 70,000 | 7,000 |
| `val` | `valid` | 10,000 | 1,000 |
| `test` | `test` | 20,000, when its labels are present | not written (no public masks) |

`valid` is for model selection and `test` for final scores. With the 2018 label
file there are no test labels, so only `train` and `valid` are written, and YOLO
`bdd100k-train` stops before training unless you pass `evaluation.split=valid`.

The per-image labels in `bdd100k_labels.zip` (also on the Hub) cover the test split.
They look like full annotations: 18.4 boxes per test image, against 18.4 in train and
18.6 in val, with the same class mix. They have not been checked against an independent source.

Models trained before this change used a seeded 85% of `train`, selected checkpoints
on the other 15%, and were scored on the official val split, which the old layout
called `test`. Their published scores are therefore val scores.

## Label files

- The 2018 file has 69,863 train entries; 137 train images have no entry. The
  per-image labels cover all 70,000.
- Val is identical in both: the same 10,000 images and the same class counts.
- Frame attributes (`weather`, `timeofday`, `scene`) are only in the 2018 file and the
  per-image labels, not in `det_20`.
- Detection drops `other person`, `other vehicle` and `trailer` (ignored regions in the
  official evaluation) and counts them. The 2020 names `pedestrian`, `motorcycle`,
  `bicycle` are read as `person`, `motor`, `bike`.
- Boxes with zero width or height are dropped and counted (395 in train with the
  per-image labels).

## Class distributions (per-image labels)

| Task | Class | Train | Val | Test |
|---|---|---:|---:|---:|
| weather | clear | 37,412 | 5,346 | 10,756 |
| | overcast | 8,784 | 1,239 | 2,568 |
| | unknown | 8,134 | 1,157 | 2,340 |
| | snowy | 5,571 | 769 | 1,551 |
| | rainy | 5,083 | 738 | 1,306 |
| | partly cloudy | 4,886 | 738 | 1,441 |
| | foggy | 130 | 13 | 38 |
| period | daytime | 36,800 | 5,258 | 10,446 |
| | night | 28,028 | 3,929 | 8,036 |
| | dawn or dusk | 5,033 | 778 | 1,476 |
| | unknown | 139 | 35 | 42 |
| scenario | city street | 43,581 | 6,112 | 12,288 |
| | highway | 17,414 | 2,499 | 5,069 |
| | residential | 8,105 | 1,253 | 2,382 |
| | parking lot | 378 | 49 | 108 |
| | unknown | 366 | 53 | 98 |
| | tunnel | 129 | 27 | 49 |
| | gas stations | 27 | 7 | 6 |

| Detection boxes | person | rider | car | truck | bus | train | motor | bike | traffic light | traffic sign |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 91,405 | 4,521 | 713,917 | 30,003 | 11,684 | 136 | 3,002 | 7,225 | 186,224 | 239,893 |
| val | 13,262 | 649 | 102,506 | 4,245 | 1,597 | 15 | 452 | 1,007 | 26,885 | 34,908 |
| test | 24,641 | 1,293 | 205,094 | 8,701 | 3,217 | 28 | 841 | 1,997 | 52,817 | 68,992 |

Consequences:

- Heavy imbalance. Top-1 accuracy is dominated by `clear`, `city street` and
  `daytime`, so evaluation reports macro metrics and per-class F1.
- The rarest classes have very few evaluation samples (7 `gas stations` and 13
  `foggy` in val, 15 `train` boxes), so their per-class numbers are noisy.
- `unknown` is the official `undefined` value and is kept as a class by default
  (`--exclude-unknown` drops it). `dawn or dusk` is the official `dawn/dusk`, since a
  slash cannot be a folder name.

## Kaggle classification datasets

The three Kaggle datasets (`bdd100k_{period,weather,scenario}_classification`) match
the 2018 labels exactly: 0 class mismatches over all 79,863 labelled images, and
every file has the byte size of the raw image. Their `test/` folder has no labels and
is ignored.

## Image size and training speed

Ultralytics classification training on the full 1280x720 images is limited by the
data loader, not the GPU: about 1,100 img/s on an i5-14600KF with an RTX 4070 SUPER.
Writing smaller images with `--max-width 512` gives about 4,000 img/s, and a 6-epoch
check gave macro F1 0.629 against 0.635 at full size (one seed each, within the noise
of the rarest class). The `test` split keeps the original images unless you pass
`--shrink-test`. The timm backend already decodes at half size, so it gains less.

Preparing one classification task from the 100K images takes about 12 s, and
detection about 75 s: images are hard-linked, not copied, when the output is on the
same filesystem.
