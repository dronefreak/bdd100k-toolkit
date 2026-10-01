# Data notes (real BDD100K download)

What was found when running the toolkit on a real download (2026-10-01).
Sources: the official 100K images and 2018 label JSON, and the three Kaggle
classification datasets (`bdd100k_{period,weather,scenario}_classification`).

## Raw download

- `images/100k/train` has 70,004 jpgs (70,000 official + 4 conflict copies) split over four subfolders
  (`trainA`, `trainB`, `testA`, `testB`); `val` is flat with 10,002; the
  unlabeled `test` has 20,001 spread over the same four subfolders plus some
  loose files. All names are unique across the tree (no duplicates), so name
  lookup is safe.
- Against the official split lists (`bdd100k/data/{train,val,test}.txt` in the
  official repo: 70,000 / 10,000 / 20,000 video names), every listed name is
  present and the only extras are 4 train, 2 val and 1 test files named
  `<name> [conflicted].jpg`, copies left by a file-sync tool. They match no
  label, so the adapters ignore them. Delete them if you want a clean tree.
- The 2018 label JSON has 69,863 train and 10,000 val entries, all inside the
  official lists. Every label entry has an image. The other 137 official train
  images have no label entry and are ignored.
- Images and labels came as separate archives, extracted into separate
  folders. The adapters take `--labels-dir` for that case.

## Kaggle classification datasets vs the official labels

Checked all 79,863 labelled images, for each of the three datasets:

- 0 class mismatches against the official attributes.
- Every image has the same byte size as the raw image.
- `unknown` is the official `undefined` value; `dawn or dusk` is
  `dawn/dusk` (a slash cannot be a folder name).
- `test/` holds 20,000 loose, unlabeled images (the official test split).

The prepare step reproduces the Kaggle datasets exactly from either layout:
the three tasks' outputs from the Kaggle folders and from the raw images plus
JSON are identical file for file.

## Class distributions (train 69,863 / val 10,000; before the valid holdout)

| Task | Class | Train | Val |
|---|---|---:|---:|
| weather | clear | 37,344 | 5,346 |
| | overcast | 8,770 | 1,239 |
| | unknown | 8,119 | 1,157 |
| | snowy | 5,549 | 769 |
| | rainy | 5,070 | 738 |
| | partly cloudy | 4,881 | 738 |
| | foggy | 130 | 13 |
| period | daytime | 36,728 | 5,258 |
| | night | 27,971 | 3,929 |
| | dawn or dusk | 5,027 | 778 |
| | unknown | 137 | 35 |
| scenario | city street | 43,516 | 6,112 |
| | highway | 17,379 | 2,499 |
| | residential | 8,074 | 1,253 |
| | parking lot | 377 | 49 |
| | unknown | 361 | 53 |
| | tunnel | 129 | 27 |
| | gas stations | 27 | 7 |

Consequences:

- Heavy imbalance. Top-1 accuracy is dominated by `clear` / `city street` /
  `daytime`, so evaluation reports macro metrics and per-class F1.
- The test split has very few samples of the rarest classes (7 `gas
  stations`, 13 `foggy`, 27 `tunnel`), so their per-class numbers are noisy.
- `unknown` is 12% of weather train; it is a real class by default, which
  makes it the second most common weather label after `clear`/`overcast`.

## Splits

Official `val` is the canonical `test`. A seeded 15% of train (10,479 images)
is `valid`, chosen over the sorted names of all train images, so the same
images are `valid` for every task and for detection, and the holdout does not
change with `--exclude-unknown`. Resulting sizes: train 59,384, valid 10,479,
test 10,000.

## Image size and training speed

Ultralytics classification training on the full-size 1280x720 images runs at about
1,100 img/s (about 60 s per epoch for 59k images) with the GPU mostly idle, and
the rate swings between 7 and 29 it/s within an epoch: eight loader workers each
build a whole batch of 128 and finish together, so batches arrive in bursts.
The progress bar's ETA follows that recent rate, so it reads 10 to 17 s while the
epoch really takes about 55 s.

Measured on an i5-14600KF (6 fast + 8 slow cores), RTX 4070 SUPER, real
Ultralytics training on 25% of the weather data, one epoch per variant:

| Variant | img/s | batches stalled over 300 ms |
|---|---:|---:|
| full-size, 8 workers (default) | 1,107 | 13 |
| full-size, 16 workers | 1,103 | 8 |
| full-size, augmentation off, 8 workers | 917 | 13 |
| full-size, augmentation off, 16 workers | 1,104 | 10 |
| 512 px wide, 8 workers | 2,729 | 1 |
| 512 px wide, 16 workers, no augmentation | 3,386 | 3 |
| 512 px wide, 16 workers | 3,990 | 1 |

A loader-only test (Ultralytics dataset, no model, no GPU) gave the same numbers,
so the data pipeline, not the model, is the limit. Decoding a full-size JPEG
costs only about 3 ms (0.6 ms at 512 px); the remaining per-image cost also
scales with image size. More workers and lighter augmentation do not help on the
slow cores. `--max-width 512` does: 6-epoch training took 125 s against 360 s,
with macro F1 0.629 against 0.635 on the original-resolution test split (one
seed each, within the noise of a 13-image rarest class).

The timm backend already decodes at half size, and a large model such as
convnext_tiny is GPU-bound either way, so it gains less.

## Timing

Preparing one classification task from 80k images takes about 10 s (hard
links, no copies, on the same filesystem).
