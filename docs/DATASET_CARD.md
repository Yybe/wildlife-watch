# Dataset Card — Project 11 (iWildCam v2.0 camera-trap corpus)

Every number here was computed by `src/vision/dataset_audit.py` and
`notebooks/01_dataset_audit_and_splits.ipynb` on 2026-10-23 from the label files themselves. Nothing is
quoted from a dataset README.

## 1. Identity

| Field | Entry |
| --- | --- |
| Name | iWildCam v2.0 (WILDS camera-trap benchmark; Oxford WildCam / Snapshot Safari network) |
| Access date (IST) | 2026-10-23 |
| Mirror used | `kaggle datasets thanhquang71/iwildcam-v2-0-2020-wilds-dataset` (12.0 GB) |
| Local label files | `data/raw/iwildcam_meta/iwildcam_v2.0/{metadata.csv, categories.csv, iwildcam2021_train_annotations_final.json, RELEASE_v2.0.txt}` (117 MB, no images) |
| Images used locally | `data/raw/iwildcam_v2.0/train/` — stratified smoke-test sample only (`src/vision/fetch_images.py`); full training runs on Kaggle with the dataset attached |
| Licence | **CDLA-Permissive-1.0** for iWildCam, per the WILDS datasets page ([wilds.stanford.edu/datasets](https://wilds.stanford.edu/datasets/), licence text https://cdla.dev/permissive-1-0/) — verbatim extraction with heading boundaries and clause quotes in `docs/LICENCE_EVIDENCE.md`; the Kaggle mirror ships no licence file of its own. Obligations only trigger on *publishing the data* (§3.1): give recipients the text, name or hyperlink of the Agreement, put prominent notices on any file you changed, and preserve attribution to the data provider (the Beery–Cole–Gjoka citation). Training is unrestricted, and §3.4 states the Agreement imposes **no** obligation on *Results*, so the trained classifier and its metrics carry no licence conditions. **Correction:** this row previously said CC BY-NC-SA 4.0 and claimed commercial use was prohibited — that sentence belongs to the RxRx1 section nine heading-boundaries lower on the same page. Upstream imagery terms are now checked too: the Snapshot Safari organisation page on Zooniverse states "Images captured by Snapshot Safari are licensed under a Creative Commons Attribution 4.0 International License" (link resolves to creativecommons.org/licenses/by/4.0), so the two layers together read **CDLA-Permissive-1.0 for the WILDS release, over CC BY 4.0 photographs — attribute both, no non-commercial or share-alike restriction**. What that pairing does *not* prove is per-image provenance: the network aggregates projects and the same page warns some displayed photos come from other databases, so a frame outside Snapshot Safari inherits its own contributor's terms. Evidence and exact quotes: `docs/LICENCE_EVIDENCE.md` §5 |
| Derivative of | Snapshot Safari camera-trap imagery; resized to height 448 by the WILDS v2.0 release |

## 2. Contents and labels

| Field | Measured |
| --- | --- |
| Labelled images in `metadata.csv` | 203,029 (one row per image, no duplicate filenames or ids) |
| Camera sites (`location`) | 323 |
| Sequences / animal visits (`seq_id`) | 36,484 |
| Frames per visit | median 5, p90 10, max 10 (burst photography) |
| Camera-day groups (site x date) | 7,538 — median 10 images, p95 90, max 3,476 |
| Date range | 2013-01-01 18:26:53 → 2015-11-12 04:49:18 |
| Taxonomy | 206 entries in `categories.csv`; **182** appear in these files |
| Empty frames | label 0 = `empty`, **34.2 %** of all rows |
| Longest / shortest class | `empty` 69,487 vs `paguma larvata` 101 (train pool); 110 categories have under 100 images |
| Classes never appearing in train | 24 taxonomy entries |
| Night activity | 32.1 % of labelled animal frames are between 18:00 and 06:00 |
| Image resolution | measured over sampled files: height fixed **448**, width 560–796 (aspect preserved). The annotation JSON still declares 1920x1080 — it is stale; quote the measurement |
| Bounding boxes | **none in this corpus** — detection/segmentation metrics come from the separate boxed set (§6) |

## 3. Splits actually used

**Correction made on 2026-10-23 at 21:59, after the full corpus was on disk.** An earlier version of
this card claimed the Kaggle mirror served images for the official `train` split only, because
`iwildcam_v2.0/val/<uuid>.jpg` returned 404. That was a wrong inference from a wrong path: the mirror
stores **every** split's images in one folder named `train`, keyed by uuid. Matching filenames against
the extracted tree gives 129,809/129,809 train, 14,961/14,961 val, 42,791/42,791 test, 7,314 id_val and
8,154 id_test present, plus 14,611 unlabelled files that no manifest row references. So the benchmark's
own splits are used, which are disjoint on camera site *and* on date by design. The recarved holdouts
(`--splits recarved`) are kept as a robustness check, not as the primary split.

| Split | Images | Camera sites | Sequences | Source |
| --- | --- | --- | --- | --- |
| train | 36,060 | 199 | 5,543 | official `train`, class-cut and capped |
| val | 10,234 | 32 | 1,843 | official `val` |
| test | 19,928 | 44 | 2,952 | official `test` (unseen sites → out-of-distribution) |

Leakage checks, all zero: sequences shared, camera-days shared and camera sites shared for
train/val, train/test and val/test; duplicate filenames zero.

Taxonomy in use: `empty` plus the 12 most-populated species across the whole corpus — `tayassu pecari`,
`dasyprocta punctata`, `puma concolor`, `leopardus pardalis`, `panthera onca`, **`bos taurus`**,
`aepyceros melampus`, `cephalophus nigrifrons`, `urocyon cinereoargenteus`, `meleagris ocellata`,
`crax rubra`, `mazama temama`. Cattle enters the set now that selection runs over all splits rather
than over `train` alone, which is convenient for the conflict narrative: predator and livestock
classes coexist in the same split.

All 13 classes appear in test, but `tayassu pecari` has only **2** test images, so macro F1 over the
full label set is dominated by a class that two predictions can swing by 7 points. Both numbers are
reported: macro F1 over all 13 classes and macro F1 over classes with ≥ 50 test images. Manifests,
class counts and sha256 prefixes are in `data/splits/manifest.json`.

Capping is per class **by whole sequence**, so trimming a class can never split a burst. Balance is
enforced per split by that cap; val and test keep whatever their sites contain.


## 4. Preprocessing and augmentation

| Field | Entry |
| --- | --- |
| Resize / normalisation | status quo 448 px height → train at 224 (classification) and 640 (detection); ImageNet mean/std with pretrained backbones |
| Train-only augmentation | horizontal flip, brightness/contrast jitter, random cropping; heavy blur and rain for robustness testing (§5), never on val/test |
| Normalisation source | pretrained backbone statistics, recorded per config |
| Cache | decoded tensors under `data/processed/` if a second epoch pass is needed |
| Seeds | `SEED = 13` in every config; `torch.manual_seed`, `numpy`, `random`, DataLoader `worker_init_fn` |
| Audio features | deferred — not in Phase I scope |

## 5. Known bias and risks

| Risk | Evidence and handling |
| --- | --- |
| Geographic bias | Single region network (Neotropical + African sites mixed by WILDS); report per-site results, not only overall |
| Empty-frame dominance | 34.2 % of frames; accuracy alone meaningless — macro F1 and false alerts per camera-day are the headline metrics |
| Rare-species recall | 110 categories under 100 images; the 12-class cut is a documented limitation, rare classes reported separately |
| Night / infrared distribution shift | 32.1 % night frames; error analysis split by hour and by site |
| Humans in frames | v2.0 removed some human/indoor images; any frame containing a person is excluded from the demo and stated in the responsible-use section |
| Test-set provenance | This mirror labels every split; the public WILDS `test` is unlabelled. Our reported test set is *our own held-out sites*, not the public benchmark leaderboard |
| Conflict labels | **None exist.** No ground-truth human-wildlife-conflict labels in this corpus, so the alert rule is a designed, justified function evaluated by false/missed alerts — never claimed as validated risk prediction |

## 6. Second dataset (detection and segmentation)

| Field | Entry |
| --- | --- |
| Name | `kaggle datasets ankanghosh651/object-detection-wildlife-dataset-yolo-format` (471 MB) |
| Contents | `final_data/{train,valid,test}/images` + YOLO labels + `data_wl.yaml` (verified by file listing) |
| Why | the species corpus has no boxes, so mAP/IoU and the two-architecture detection benchmark come from here |
| Licence | MIT, as reported by the Kaggle API on download (`License(s): MIT`); mirror ships no licence file of its own — re-confirm at submission |
| Optional additions | `wangsheng0352/ir-wildlife` (2.2 GB, COCO, infra-red) for night robustness; `mrlaurent/big-four` (585 MB) for extra conflict species |
| Rejected | `silviamatoke/serengeti-dataset` (10.1 GB) — titled Serengeti but contains `Set1/1.02-Agouti/SEQ…` folders; agoutis are Neotropical, so the mirror is mislabelled and must not be cited |
