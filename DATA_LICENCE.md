# Data licence notice for the committed split manifests

This repository commits **derived split manifests** (`data/splits/*.csv`,
`data/splits/manifest.json`) — lists of image filenames, labels, camera-site ids and
sequence ids. These are *not* the original images, which must be downloaded separately
(see `docs/DATASET_CARD.md`).

## What these files are

The manifests are a **modification of the iWildCam v2.0 (WILDS) release annotations**: the
original `train`/`val`/`test` split was cut to 12 species + `empty` and capped by whole
camera-trap sequence, and the split assignment is recorded per filename UUID. Under the
release's licence this makes the committed manifests **"Enhanced Data"**.

## Licence of the underlying data

- iWildCam v2.0 release: **CDLA-Permissive-1.0** — https://cdla.dev/permissive-1-0/
  (agreement named per CDLA §3.1; provider attribution preserved below).
- Underlying photographs: **Snapshot Safari / Oxford WildCam, CC BY 4.0**
  — https://creativecommons.org/licenses/by/4.0/

## Change notice (CDLA §3.1)

The files under `data/splits/` are **changed from the original release**: species set
subsampled to 13 classes, sequences capped whole, and split membership recomputed by camera
site. The full audit and rationale are in `notebooks/01_dataset_audit_and_splits.ipynb` and
`docs/DATASET_CARD.md`.

## Attribution (preserved per CDLA §3.1 / CC BY)

> Beery, S., Cole, E., & Gjoka, A. (2020). *The iWildCam 2020 Challenge Dataset.*
> Based on imagery from the Snapshot Safari / Oxford WildCam camera-trap network.
> Released via the WILDS benchmark (Koh et al., 2021).

No non-commercial or share-alike restriction applies (an earlier draft of the dataset card
misread the RxRx1 licence; corrected — see `docs/LICENCE_EVIDENCE.md`).

## Code licence

The software in this repository is separately licensed under the MIT License (`LICENSE`).
Under CDLA §3.4 the data agreement imposes **no obligation on results**, so trained weights
and metrics are unencumbered.
