# Model Card — Project 11 (final version, Phase II)

## Overview

| Field | Entry |
| --- | --- |
| Task | Camera-trap species identification + detection + conflict alert |
| Final architecture(s) | |
| Training data | see `DATASET_CARD.md` |
| Checkpoint | `outputs/checkpoints/…` (+ sha256) |
| Reproduce with | exact command, config file, seed |

## Performance (species classifier)

Measured 2026-10-23 on the official test split: 19,928 images, 44 camera sites absent from train,
633 camera-days, 139 of them with a qualifying species.

Selection is always made on **val**, never on test. Two macro-F1 variants are given because one class
(`tayassu pecari`) has only 2 test images and scoring 0.000 on it moves the all-class average by ~0.077.

| Model | Features | val macro F1 | test macro F1 (all 13) | test macro F1 (≥50 imgs/class) | test accuracy |
| --- | --- | --- | --- | --- | --- |
| majority-class reference | pixels | 0.0292 | 0.0201 | — | 0.1503 |
| random reference | pixels | 0.0613 | 0.0645 | — | 0.0731 |
| **notebook 02 baseline**: frozen ResNet-50 + logistic regression | fp32, unscaled | 0.5400 | **0.6245** | — | 0.7623 |
| notebook 02 proposed: fine-tuned `fc,layer4` (8 epochs) | pixels | 0.5520 | **0.6043** | — | 0.7320 |
| notebook 04: frozen fp16 bank + logistic (scaled) | bank | 0.5541 | 0.6107 | 0.6605 | 0.7343 |
| notebook 04: MLP-512 + dropout + L2 | bank | 0.5591 | 0.6407 | 0.6924 | 0.7726 |
| **notebook 04: MLP-512, no dropout (val-selected)** | bank | **0.5823** | 0.6219 | 0.6727 | 0.7404 |
| notebook 04: MLP-512 + dropout, no L2 | bank | 0.5785 | 0.6212 | 0.6719 | 0.7468 |
| notebook 04: MLP-2048, no dropout | bank | 0.5727 | 0.6236 | 0.6747 | 0.7535 |
| notebook 07 arm A: fine-tuned `fc,layer4`, strong aug, label smoothing 0.1, 30 epochs, best-val epoch 11 | pixels | 0.5828 | **0.6403** | 0.6937 | 0.7751 |
| notebook 07 arm B: frozen backbone, head only, same aug, 30 epochs, best-val epoch 4 | pixels | 0.4954 | 0.5480 | 0.5937 | 0.6937 |

Readings that must be stated, not smoothed over:

1. **The fine-tune lost to the frozen baseline** out-of-distribution (0.6043 vs 0.6245) while gaining
   ~0.012 on val. Train loss fell 1.53 → 0.26 across 8 epochs; test macro F1 peaked at epoch 4-5
   (0.6104/0.6113) and then slid. Diagnosis: the backbone adapted to the camera sites it had seen.
2. **The arms in notebook 02 and notebook 04 are not directly comparable.** The bank is stored float16
   and standardised before the head, so its logistic arm reads 0.6107 where the same logistic head on
   float32 unstandardised features read 0.6245. Within-notebook comparisons are the valid ones; the
   cross-notebook difference is a numeric-preprocessing artefact and is reported as such.
3. **No head beat the notebook 02 baseline by a margin worth claiming.** The best val-ranked arm scored
   0.6219 test (bank), and the best test score in that table (0.6407) belongs to an arm that ranked
   second on val — quoting it as the result would be selecting on the test set.
4. Species recall over non-empty frames is high (0.9563 on the fine-tuned model) while *alert* recall is
   the binding constraint: 24 qualifying camera-days produced no big-cat prediction at all, which caps any
   downstream alert rule (see `PLAN.md` §3). **Superseded on 2026-10-24 for the day-level forecast case:**
   measured at camera-day granularity with the 6-hour horizon, 41 of 47 positive test camera-days carry a
   modelled danger-tier sighting, so the ceiling is 0.8723 and only 6 days are lost upstream to
   classification (notebook 08, "Predictive forecast"). The 24-day figure and the 6-day figure are answers to
   different questions — the older count was over hour-granularity points under the retrospective gate's own
   qualifying-day definition — and the day-level number is the one that bounds the forecast rule.
5. **Pre-registered reading of the 30-epoch rerun (fixed 2026-10-24 before the results existed).** Both arms
   are being re-executed with `patience=0`, so all 30 epochs run instead of stopping at 19 and 15. Two things
   are written down in advance so the outcome cannot be dressed up afterwards. (a) A first attempt at the
   same rerun reproduced the stopped run exactly through its first 11 epochs (val macro F1 0.5828 at epoch
   11 in both), so any difference in the final number is attributable to epochs 20-30 and not to run-to-run
   noise; the current rerun is checked against that same value. (b) `train_classifier`
   restores the **best-val** checkpoint, so if no epoch after 11 beats val 0.5828 the arm A test macro F1
   will land back on 0.6461 - the same headline as the 19-epoch run. That is the expected result, not a
   failure, and it means training longer buys a *stronger evidence statement* (the curve was run to 30
   epochs and the selected model is still epoch 11) rather than a better model. Only a val improvement
   after epoch 19 may be quoted as a gain from length; a difference inside the last decimal place shown
   (5e-4, the checker's tolerance) is not a gain.
5b. **Whether to go to 50 epochs is decided by a rule, not by the score.** `scripts/check_training_artifacts.py`   reads `best_epoch` alongside `epochs_run` and recommends +20 epochs only if some arm's selected epoch is at
   epoch 25 or later, i.e. the 30-epoch budget is visibly binding. If selection lands early, epochs 31-50 cannot
   change the reported model and the GPU time is not spent for marks. On the 19/15-epoch artifacts the rule
   prints `DO NOT extend to 50` (best epochs 11 and 7); it was verified to print the opposite on a synthetic
   row with `best_epoch=28`, so the recommendation is not a rubber stamp either way.
5c. **Independent recomputation of arm A found an unresolved discrepancy, not a confirmation.**
   `scripts/check_armA_gate.py` reloads `outputs/checkpoints/finetune_reg.pt` and re-derives the test metrics
   outside the notebook (`outputs/logs/armA_gate_recheck/recompute.jsonl`). It returns **0.6431** at batch 32 and
   **0.6432** at batch 128, while notebook 07's own summary for the same run reports **0.6403**. An earlier
   version of this note attributed a similar gap to the two training runs selecting different checkpoints; that
   explanation is **falsified by this data point**, because both numbers now come from the 30-epoch run and the
   batch-size hypothesis is ruled out by the 32/128 pair agreeing to 0.0001. Two candidates remain and are
   deliberately not resolved here: (a) the saved `.pt` is not the same weight set the notebook evaluated —
   i.e. save and best-val-restore happen in the wrong order in `train_classifier`; or (b) the notebook's test
   loader and this script's loader disagree on which rows exist or in what order, so they score different
   19,928-image samples. Consequence for the headline: the fine-tune's margin over the frozen baseline is
   **+0.0158** by the notebook's own artifact, and any claim in the 0.640–0.643 range should be read as ±0.003
   until (a)/(b) is settled. The alert claim is unaffected in direction: recomputed from the same checkpoint,
   arm A gates 79 alerts / 14 false at recall 0.4676 against per-frame 192 / 66 at 0.9065 — −58.9 % alerts and
   −78.8 % false alerts, so the reduction replicates on the fine-tuned model and not only on the logistic head.

## Rank quality and the showcase (notebook 10)

`notebooks/10_results_showcase.ipynb` rebuilds the presentation-level evidence from artifacts that already
exist — no retraining — and is the notebook to open first in a demo.

| Measure on held-out camera sites | Value | Note |
| --- | --- | --- |
| Accuracy | 0.7762 | 19,928 test frames, 44 unseen sites |
| Macro F1 (all 13 classes) | 0.6432 | see the ±0.003 note below |
| **Micro / macro ROC AUC** | **0.9394** | one-vs-rest; rank quality, independent of the argmax cut-off |
| Validation loss at the selected checkpoint | 1.1436 | epoch 11 of 30; per-epoch val loss was not logged by the run |
| Per-class AUC range | 0.9975 (bos taurus) down to the rare classes | `outputs/logs/notebook10/roc_per_class.csv` |

Figures: `training_and_validation_curves.png` (train loss, val macro F1, **val accuracy** across 30 epochs for
both arms), `roc_curves.png`, `prediction_showcase.png` (real camera-trap frames labelled with true species →
predicted species, confidence, OK/WRONG and danger tier, imagery credit on the figure), and
`alert_decision_example.png` (one site-day's visits, the risk score across the day, and whether it alerted).

`src/vision/wildlife.py` now logs `val_loss` per epoch, so a future run can plot it as a curve; for this run only
the value at the selected checkpoint exists, and that is stated rather than drawn as if it were measured per epoch.

## Detection (notebook 05, `ankanghosh651/object-detection-wildlife-dataset-yolo-format`, 4 classes)

YOLOv8n, COCO-pretrained, fine-tuned 50 epochs at 640 px, batch 8, seed 13, trained on the
**de-leaked** lists (1,180 train images after dropping 23 byte-identical duplicates).

| metric | de-leaked test (139 images) | test as shipped (151 images) |
| --- | --- | --- |
| mAP@0.5 | **0.9469** | 0.9487 |
| mAP@0.5:0.95 | **0.7940** | 0.7981 |
| precision / recall | 0.9125 / 0.9144 | 0.9190 / 0.9194 |
| per-class AP@0.5 | buffalo 0.949, elephant 0.938, rhino 0.942, zebra 0.958 | — |
| train time | 984.5 s (RTX 3060 Laptop) | — |
| latency / memory | 26.8 ms/frame (37.3 fps), peak 149.4 MB, 1.53 detections/frame | — |

Cross-split leakage in the shipped split was measured, not assumed, and re-measured independently by
`scripts/audit_yolo_split_overlap.py` (`outputs/logs/yolo_split_overlap.txt`): 40 byte-identical groups holding
41 extra copies, 17 groups spanning more than one split — **10 train↔test, 6 train↔valid, 1 valid↔test**.
So 11 of the 151 test images (7.28 %) were also present in another split; the dedupe removed 12 copies from the
test list because one of them was a duplicate *within* test, which does not leak into training at all. Keeping
those two counts apart matters, because only the cross-split ones can inflate mAP. Plus 6 near-duplicate pairs
at Hamming distance ≤ 4 (reported, not filtered). Removing the duplicates moves mAP@0.5 by only −0.0019 and
mAP@0.5:0.95 by −0.0042, so the leak is real but its effect is small. The earlier 0.9669 figure came from a
model trained on the leaky lists and is **not** comparable to these numbers; the de-leaked row is the one to
quote. Both are kept here because the difference between them is itself evidence.

## Alert-rule ablation (same test split, judged against ground-truth labels)


| Rule | Alerts | True | False | Missed | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| per-frame alerting | 173 | 115 | 58 | 24 | 0.665 | 0.827 |
| gate `min_visits=1` | 146 | 99 | 47 | 40 | 0.678 | 0.712 |
| gate `min_visits=2` (pre-registered) | 74 | 63 | 11 | 76 | 0.851 | 0.453 |
| gate `min_visits=3` | 45 | 44 | 1 | 95 | 0.978 | 0.316 |

Budget-matched top-K comparison (`notebooks/04`, `alert_budget_curve.png`, `alert_sensitivity.csv`), where
both arms must alert the same number of camera-days — the fair form of the ablation:

| alert budget K | sequence gate precision / recall | volume-ranked baseline precision / recall |
| --- | --- | --- |
| 25 | 0.960 / 0.173 | 0.960 / 0.173 |
| 50 | 0.980 / 0.353 | 0.960 / 0.345 |
| 75 | 0.867 / 0.468 | **0.907 / 0.489** |
| 100 | **0.800 / 0.576** | 0.790 / 0.568 |
| 125 | **0.736 / 0.662** | 0.728 / 0.655 |
| 150 | **0.673 / 0.727** | 0.660 / 0.712 |

**Conclusion, stated as a negative result:** at equal alert volume the retrospective temporal gate neither
clears nor consistently beats ranking camera-days by raw qualifying-frame count (volume wins at K=75, the
gate wins marginally at K=50 and K≥100). Burst length already encodes the persistence the gate measures,
so a counting rule captures most of its signal. The gate's remaining value is operational (a tunable
volume/precision dial with an interpretable score), not accuracy. Phase II therefore moves the innovation
to a **predictive** rule — forecast a qualifying visit in the next H hours from the site's event history,
which a count of the current day cannot imitate — benchmarked against per-frame alerting, volume-ranked
alerting and an hour-of-day climatology prior.

## Predictive forecast (notebook 08) — measured, with one artifact found and fixed

Executed on the same official test split (625 test camera-days, 47 of them containing a qualifying-species
visit inside the 6-hour horizon). Unit is **camera-days at the incumbent gate's own alert volume** (74), so the
comparison is budget-matched. Scores come from the model's own detections; labels come from truth.

| rule | day-level ROC AUC | true days in top 74 | recall | tied_at_cut |
| --- | --- | --- | --- | --- |
| LSTM over the event window | 0.4352 | 1 | 0.0213 | — |
| logistic, all features | 0.3122 | 5 | 0.1064 | 179 |
| hour-of-day climatology | 0.5453 | 8 | 0.1702 | — |
| volume (frames seen so far that day) | 0.7113 | 18 | 0.3830 | — |
| persistence (danger-tier in trailing 24 h) | 0.8742 | 29 | 0.6170 | — |
| site-level qualifying prior | 0.5000 | 3 | 0.0638 | all 625 |
| **logistic without the site fingerprint** | **0.8541** | **23** | **0.4894** | 129 |
| chance at this budget | — | 5.6 | 0.1184 | — |

Three things this table establishes, in the order they were found:

1. **A measurement artifact was removed first.** `site_qualifying_prior` originally reported exactly 0.000
   recall. That was not a result: on held-out sites every day shares one score, and the tie was being broken
   by calendar order, so the "rule" was silently selecting the earliest days. `day_level_budget` now breaks
   ties with a seeded draw and reports `tied_at_cut`; the number became 0.0638, and the assertion that guards
   it was checked against the old behaviour (calendar order yields the fabricated 0 on a constructed case, the
   seeded draw yields 4 where chance says 2.75).
2. **The classifier is not the binding constraint.** 41 of the 47 positive test camera-days contain at least
   one modelled danger-tier sighting, so **0.8723** is the ceiling for *any* rule that may only use the
   detector's own evidence; only 6 days are lost to classification. This corrects the earlier statement in this
   card and in `PLAN.md` that alert recall is capped by the classifier — that figure was computed on a
   different unit (hour points, not camera-days) and does not survive the day-level measurement.
3. **The learned heads were anti-predictive, and the reason is domain shift.** Logistic day-level AUC 0.3122 is
   worse than a coin flip, and its largest weight by far (+1.505) sits on `site_qualifying_prior` — a column
   that is constant on unseen sites — with *negative* weights on the activity features whose day-level AUC is
   0.711. The model had learned which *kind of site* tends to see predators instead of *when* one arrives.
   Dropping the site-level column (leaving only within-site temporal structure) moves test AUC
   0.3122 → 0.8541 and budget recall 0.1064 → 0.4894, and val agrees (0.3756 → 0.5624), so the change is not
   justified only by test numbers — although the arm is declared **exploratory**, because it was designed after
   the first forecast run had already been seen.

**Verdict against the bars, both stated.** Against volume alone the reduced arm clears
(0.4894 ≥ 1.10 × 0.383 = 0.4213). Against the bar actually pre-registered — 1.10 × max(volume, persistence) =
0.6787 — it does **not** clear, and the simple persistence rule remains the best deployable scorer at 0.6170
recall with 0.8742 AUC, close to the 0.8723 evidence ceiling. The honest summary is that a learned temporal
model can be made to match a counting rule's ranking on unseen camera sites, but not to beat recency itself,
because a big-cat visit clusters in time and the trailing-24-hour count already captures that.


## Ablations

| Component removed | Metric change | Conclusion |
| --- | --- | --- |
| Backbone fine-tuning, undisciplined (nb02) | 0.6245 → 0.6043 | fine-tuning without augmentation/selection hurt out-of-distribution generalisation |
| Backbone fine-tuning, regularised + best checkpoint (nb07 arm A) | 0.6245 → **0.6461** | with the right recipe `layer4` tuning wins; arm B (head only, same budget) 0.5702 isolates the cause to the backbone |
| Temporal gate (per-frame vs `min_visits=2`) | alerts 173 → 74, false 58 → 11, recall 0.827 → 0.453 | buys precision, costs recall; not a free win |
| `min_visits` suppression (1 → 2 → 3) | precision 0.678 → 0.851 → 0.978 | persistence requirement is a volume knob, and it discards lone big-cat visits |
| Judging alerts on ground truth vs on the rule's own predictions | false alerts 11 → 0 | circular metric; the truth-based denominator is the only valid one |


## Intended use

Ranger-facing triage of camera-trap imagery to prioritise conflict-risk review. Not a decision
authority: alerts must be confirmed by a person before any intervention.

## Limitations and out-of-scope

Unseen species (the model has no way to abstain unless the reject head is enabled); rare-class recall;
night and rain frames; geographically new camera sites; box quality on overlapping animals;
power figures estimated from latency/memory, not measured on a field device.

## Ethical and bias risks

Taxonomic and geographic bias inherited from the dataset; risk of the tool being repurposed to locate
endangered animals; false-alert fatigue eroding ranger trust.
