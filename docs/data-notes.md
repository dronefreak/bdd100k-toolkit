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

## Timing

Preparing one classification task from 80k images takes about 10 s (hard
links, no copies, on the same filesystem).
