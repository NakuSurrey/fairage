# Data

Datasets are NOT in this repo. Files are too large for git and licenses block redistribution. Download them yourself using the steps below.

## Folder layout expected by the code

```
data/
├── utkface/              # primary dataset for age training
│   └── *.jpg             # filename format: [age]_[gender]_[race]_[date].jpg.chip.jpg
├── pad/                  # presentation attack detection — set in Phase 6
│   ├── real/
│   ├── print/
│   └── replay/
└── README.md             # this file
```

`src/config.py` reads these paths. Do not rename folders.

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

## PAD — presentation attack detection

Used for: spoofing detection (Phase 6).

Decision deferred until Phase 6 — exact dataset choice depends on access at that time. Candidates:
- CelebA-Spoof (~600 GB, Microsoft Research)
- Replay-Attack DB (Idiap, requires institutional license)
- NUAA Imposter Database (smaller, easier to obtain)

When chosen, the layout will be three subfolders: `real/`, `print/`, `replay/`. Update this README at that point with download steps.

## Why datasets are not in the repo

- Size — UTKFace alone is ~250 MB, well past GitHub's friendly limit
- License — datasets carry redistribution restrictions, mirroring them is not safe
- Repro — anyone cloning the repo can grab the same data from the same source

## Verifying the data

After download, run a quick sanity check:

```bash
python -c "from pathlib import Path; n = len(list(Path('data/utkface').glob('*.jpg'))); print(f'utkface images: {n}')"
```

Expected: roughly 23000 images.
