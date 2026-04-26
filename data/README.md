# Data

Datasets are NOT in this repo. Files are too large for git and licenses block redistribution. Download them yourself using the steps below.

## Folder layout expected by the code

```
data/
├── utkface/              # primary dataset for age training
│   └── *.jpg             # filename format: [age]_[gender]_[race]_[date].jpg.chip.jpg
├── pad/                  # presentation attack detection — NUAA Imposter Database
│   ├── real/             # genuine face photos -> label 0
│   └── attack/           # printed photo attacks -> label 1
└── README.md             # this file
```

`src/config.py` reads these paths. Do not rename folders.

The PAD dataset class also accepts NUAA's native folder names (`ClientFace/` and `ImposterFace/`), so a fresh extract works without renaming.

## UTKFace — primary dataset

Used for: age estimation training, bias audit (Phase 2 to Phase 5).

Source: https://susanqq.github.io/UTKFace/

Mirror: https://www.kaggle.com/datasets/jangedoo/utkface-new

### Download steps

```bash
# step 1 — go to Kaggle mirror, accept terms, download the zip
# step 2 — extract into data/utkface/
unzip utkface-new.zip -d data/utkface/

# step 3 — verify count (~23k images)
ls data/utkface/ | wc -l
```

### Filename format

```
[age]_[gender]_[race]_[date].jpg.chip.jpg
        e.g.  25_0_2_20170104203021098.jpg.chip.jpg
              age=25, gender=0 (male), race=2 (Asian)
```

Mapping used in `src/config.py`:
- gender: 0 = Male, 1 = Female
- race: 0 = White, 1 = Black, 2 = Asian, 3 = Indian, 4 = Others

## PAD — NUAA Imposter Database

Used for: presentation attack detection (Phase 6).

Picked NUAA over CelebA-Spoof (~600 GB, too big) and Replay-Attack DB (institutional license required). NUAA is small (~12k images), free, and exercises the full PAD pipeline end to end. The trade-off: NUAA only covers print attacks (no replay/video).

Source: http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/NUAAImposterDB_download.html

If the original site is offline, the dataset is mirrored on GitHub research repos. Search for "NUAA Imposter Database Detectedface".

### Download steps

```bash
# step 1 — request access from the NUAA contact email on the page above.
#          they reply within a few days with a download link.

# step 2 — once downloaded, extract the archive. it produces a folder like:
#            NUAAImposterDB/
#            └── Detectedface/
#                ├── ClientFace/
#                └── ImposterFace/

# step 3 — option A: keep NUAA's native folder names
mv NUAAImposterDB/Detectedface data/pad

# step 3 — option B: rename to the simpler real/attack layout
mkdir -p data/pad
mv NUAAImposterDB/Detectedface/ClientFace data/pad/real
mv NUAAImposterDB/Detectedface/ImposterFace data/pad/attack

# step 4 — verify counts (~5k real and ~7k attack images)
find data/pad -name "*.jpg" -type f | wc -l
```

### What NUAA contains

- `ClientFace/` (or `real/`) — frontal face photos taken with a webcam from real people, typically ~640x480
- `ImposterFace/` (or `attack/`) — the same faces printed on paper and re-photographed
- Subjects organised in per-person subfolders (`subj_01/`, `subj_02/`, etc.). The dataset class scans recursively, so subfolder structure does not need to be flattened.

### Why NUAA is enough for this project

The PAD module exists to prove the architecture can detect spoof attempts at all. NUAA gives a working baseline; in production, the model would be retrained on a richer dataset (CelebA-Spoof, Replay-Attack) once available. This is documented in `COMPLIANCE.md` as a known limitation.

## Why datasets are not in the repo

- Size — UTKFace alone is ~250 MB, NUAA ~600 MB
- License — both datasets carry redistribution restrictions
- Repro — anyone cloning the repo can grab the same data from the same source

## Verifying the data

After download, run a quick sanity check:

```bash
python -c "from pathlib import Path; n = len(list(Path('data/utkface').glob('*.jpg'))); print(f'utkface images: {n}')"
python -c "from pathlib import Path; n = sum(1 for _ in Path('data/pad').rglob('*.jpg')); print(f'pad images: {n}')"
```

Expected: roughly 23,000 UTKFace images, roughly 12,000 NUAA images.
