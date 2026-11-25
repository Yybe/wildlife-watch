# Source and Contribution Log

Mandatory in both phases. Every dataset, code fragment, architecture, pretrained weight, notebook,
repository, tutorial and paper used must appear here, with what was taken and what was changed.
Add a row the moment you use something, not at submission time.

| # | Date | Type | Name and link | What it provides | What I changed or adapted | Where used |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-10-23 | dataset | `kaggle datasets thanhquang71/iwildcam-v2-0-2020-wilds-dataset` — mirror of iWildCam v2.0 (WILDS benchmark, Oxford WildCam / Snapshot Safari) | 203,029 labelled camera-trap images with `seq_id`, `location`, `datetime`; 206-entry taxonomy; all 217,640 image files, every split, under one `iwildcam_v2.0/train/` folder | Used as-is, with the benchmark's own `train`/`val`/`test` splits, cut to 12 species + `empty` and capped by whole sequence. Also added a `--splits recarved` alternative (held-out sites for test, held-out camera-days for val) as a robustness check. An initial claim that val/test images were missing was disproved by filename matching after the full extraction | `src/vision/{dataset_audit,make_splits,fetch_images}.py`, `notebooks/01_dataset_audit_and_splits.ipynb` |
| 2 | 2026-10-23 | dataset | `kaggle datasets ankanghosh651/object-detection-wildlife-dataset-yolo-format` (471 MB) | Bounding-box wildlife set in YOLO layout (`final_data/{train,valid,test}`, `data_wl.yaml`) | Planned: label-map harmonisation to the species list; nothing adapted yet | detection benchmark (Phase I day 8 / Phase II) |
| 3 | 2026-10-23 | dataset (candidate) | `kaggle datasets wangsheng0352/ir-wildlife` (2.2 GB, COCO) and `mrlaurent/big-four` (585 MB) | Infra-red/night frames for robustness; extra conflict species | Not used yet — evaluated by file listing only | error analysis (Phase II) |
| 4a | 2026-10-24 | licence | WILDS datasets page https://wilds.stanford.edu/datasets/ → licence text https://cdla.dev/permissive-1-0/ (redirect from cdla.io) | iWildCam is released under **CDLA-Permissive-1.0**. Conditions (§3.1) bind only on *publishing data*: pass on the Agreement text/name/hyperlink, mark changed files, preserve provider attribution. §3.4 imposes no obligation on *Results*, so the trained classifier and metrics are unencumbered. Verbatim quotes with heading boundaries in `docs/LICENCE_EVIDENCE.md` | Read only. An earlier entry here said CC BY-NC-SA 4.0 and called the data non-commercial — that sentence is the **RxRx1** licence, nine heading-boundaries lower on the same page, and it was misread from a search snippet that carries no section boundary. Corrected in `DATASET_CARD.md` §1 after walking the page structure in document order | dataset card, report, configs, `docs/LICENCE_EVIDENCE.md` |
| 4b | 2026-10-24 | licence (upstream) | Snapshot Safari organisation page, Zooniverse — https://www.zooniverse.org/organizations/meredithspalmer/snapshot-safari | "Images captured by Snapshot Safari are licensed under a Creative Commons Attribution 4.0 International License" (anchor → creativecommons.org/licenses/by/4.0). This closes the "upstream terms unknown" item: attribution required, no NC or ShareAlike. The same page asks researchers publishing with the metadata to email `snapshotsafari@gmail.com` (courtesy, not a licence term) and warns that some *displayed* species-menu photos come from other databases — which is why per-image provenance is recorded as unproven rather than assumed | Read only, from the rendered DOM (the page is a client-side app; a plain fetch returns nothing). Recorded in `docs/LICENCE_EVIDENCE.md` §5 | dataset card §1, notebook 01 open items, one-pager risks |
| 4 | 2026-10-23 | reference | WILDS benchmark documentation and the iWildCam v2.0 `RELEASE_v2.0.txt` shipped inside the mirror | Explains the v2.0 split policy (splits do not overlap on date), the human/indoor image removal and the resize to height 448 | Read only; the resolution claim was re-measured locally because the annotation JSON contradicts it | `docs/DATASET_CARD.md` |
| 5 | 2026-10-23 | tool | `kaggle` CLI 2.2.5 (installed into `Desktop/PYTHOM/.venv`) | Dataset search, per-file download, kernel push/output retrieval | Configuration only: dataset attached to notebooks instead of downloaded, per-file sample fetch script | whole pipeline |
| 6 | 2026-10-23 | model | `resnet50-11ad3fa6.pth` (torchvision IMAGENET1K_V1, 98 MB), cached locally and uploaded to my private dataset `shivam2212/p11-weights` | ImageNet-pretrained ResNet-50 backbone, needed because Kaggle notebooks run offline | Loaded with `weights=None` + `load_state_dict(strict=False)`, asserting no non-`fc` tensor is missing; `fc` replaced (Identity for the frozen baseline, `Linear(2048,13)` for the fine-tune) | `notebooks` 02, `src/vision/wildlife.py` |
| 7 | 2026-10-23 | data (derived) | `shivam2212/p11-splits` (private) — my own `data/splits/*.csv` + `class_map.csv` + `manifest.json` + `src/vision/wildlife.py` | Lets the Kaggle kernel use the recarved splits and the same module code as the local repo, so no logic is duplicated in the notebook | Written by me from row 1 | Kaggle kernel `shivam2212/p11-wildlife-baseline` |
| 8 | 2026-10-23 | compute (**withdrawn as evidence — see row 11**) | Kaggle notebook session, `NvidiaTeslaT4` accelerator, internet off, 3 h timeout | GPU training without moving 12 GB over the local link | Configuration only | https://www.kaggle.com/code/shivam2212/p11-wildlife-baseline |

| 9 | | model | YOLOv8n pretrained weights — *not yet used* | Detector initialisation | To record: fine-tuning strategy and epochs | detector |
| 10 | 2026-10-24 | compute (diagnostic, **withdrawn**) | my kernel `shivam2212/p11-mount-probe` (T4 session, completed) + `kaggle datasets files/version shivam2212/p11-splits` | Measured evidence about the platform, not about the model: attached datasets mount at `/kaggle/input/datasets/<owner>/<slug>/`; the 12.0 GB mirror mounts completely inside a notebook (217,651 files, 11.42 GB) so the two failed pushes were **my depth-1 path assumption**, not a Kaggle attachment failure; and the probe itself took ~11 h because it recursed through the image corpus — resolution now stops at depth 3 | Same pass found the aux dataset was stale (old recarved splits 32,888/7,275 train/test and `wildlife.py` from 2026-10-23); re-versioned to the current official splits 36,060/10,234/19,928 and the current module before pushing replication v3 | notebook 02 (both copies), `kaggle/baseline/kernel-metadata.json`, notebook 01 open items, `MODEL_CARD`, `PLAN.md` §8 |

## Faculty decisions recorded here

| Date | Question | Answer / evidence |
| --- | --- | --- |
| 2026-10-23 | The project compendium requires a C/C++/Java implementation; the two-phase hackathon instructions that define the marks contain no language clause. Is Python accepted? | **OPEN — awaiting written reply.** Until then, `src/from_scratch/` is written as plain array code so a C++ (Eigen) or Java (ND4J) port of the neuron/MLP/optimizer part stays cheap |
| | Is generative-AI assistance permitted, and in what form must it be declared? | **OPEN** — declared below on the assumption that it must be |

## Generative AI assistance

Allowed only if the faculty permits it, and every use must be declared prompt-wise.

| Date | Tool | Prompt (verbatim, condensed to the ask) | What it produced | What I verified or corrected myself |
| --- | --- | --- | --- | --- |
| 2026-10-23 | Qoder (agentic IDE assistant) | "make a new folder of DL_POB … i'm doing project 11 (we're allowed to use python) how would we get started" + follow-ups on dataset source, resources, and notebook format | Repository scaffold, `PLAN.md`, the audit / split / sample-fetch scripts, `notebooks/01` and the doc templates | Ran every script; found the Kaggle mirrors lack val/test images and recarved the splits; found and fixed a duplicate-row bug in the split merge; replaced a 5-site test holdout with a 40-site one; measured image resolution instead of trusting the annotation file; rejected the mislabelled `silviamatoke/serengeti-dataset` mirror |
| 2026-10-23 | ChatGPT (earlier session, share link supplied by the student) | Project-11 scope clarification | A summary of the four product functions, the fundamentals checklist and the C/C++/Java warning | Cross-checked against both source PDFs; nothing new was adopted beyond the two points already logged here |

## Pretrained weights

| Model | Source | Frozen / unfrozen layers | Justification |
| --- | --- | --- | --- |
| | | | |

## Rejected or discarded sources

| Date | Candidate | Why rejected |
| --- | --- | --- |
| 2026-10-23 | `silviamatoke/serengeti-dataset` (10.1 GB, titled "Serengeti Dataset") | Folder tree is `Set1/1.02-Agouti/SEQ…`; agoutis are not part of the Serengeti fauna, so the mirror is mislabelled and cannot be cited as such |
| 2026-10-23 | `qitvision/iwildcam2020-256` (11.3 GB) | Its image uuids are absent from iWildCam v2.0 metadata, so it cannot be joined to the label table we audited |

| 11 | 2026-10-24 | withdrawal | my kernel `shivam2212/p11-wildlife-baseline` v3 and the pulled output folder | Honest removal, not a silence. The kernel reached COMPLETE and its log shows the executed notebook being written, but `kernels output` returned only artifacts from an earlier version — no retrievable executed notebook, so the replication cannot be verified. Every number in `REPORT.md`, `MODEL_CARD.md` and `configs/experiments.yaml` is now attributed to the local RTX 3060 Laptop run only. The `kaggle/` deployment directory, the ingestion script and the cloud output folders were moved out of the repo; rows 8 and 10 stay here because the attempt was made and disclosure is required. | `README.md`, `PLAN.md` §8, `PROPOSAL_onepager.md`, `configs/experiments.yaml`, `scripts/check_notebooks.py` |
