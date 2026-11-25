# Technical Report — Phase I
## Project 11: Wildlife Species Identification and Conflict Alert System

Course CSAI3027P Deep Learning · Two-phase hackathon · Individual
Student: Shivam _(roll no.)_ · Date: 2026-10-24 · Language: Python (see `docs/SOURCE_LOG.md` for the
faculty ruling on the compendium's C/C++/Java wording)

> Written for a reader who has not seen the code. Every number is from a run recorded in
> `configs/experiments.yaml` and reproducible with the commands in `README.md`. Cells marked
> All numbers below are from executed artifacts: the 30-epoch pass (notebook 07, both arms, `patience=0`), the predictive alert rule (notebook 08) and the error analysis (notebook 09). `scripts/verify_claims.py` checks each one against the file that produced it.
> version was frozen — they are left blank rather than estimated.

## 1. Problem and user

Protected areas photograph wildlife with camera traps and review the images weeks later. When a predator
or an elephant appears at the forest edge near livestock or cropland, the discovery arrives after the
incident, so both prevention and compensation are late. The target user is a field ranger or
conflict-mitigation officer operating a camera-trap network, mostly offline, on low-power hardware.

The product does four things: identify the species in a frame, locate animals with boxes in a separate
detection stream, turn a site's recent sightings into a risk decision, and present the result with a
confidence and an explanation. Success is measured in species accuracy on **unseen camera sites** and in
**alerts per camera-day**, not in overall accuracy, because 34.2 % of the corpus contains no animal at all
— a model that always answers "empty" would score a third of the benchmark on accuracy alone.

## 2. Data

iWildCam v2.0 (WILDS camera-trap benchmark, Oxford WildCam / Snapshot Safari imagery), mirror
`thanhquang71/iwildcam-v2-0-2020-wilds-dataset`, 12.0 GB, 203,029 labelled images, 182 species present,
323 camera sites, dates 2013-01-01 to 2015-11-12. Two licence layers apply, and both were read from the
source pages rather than assumed: the WILDS release is **CDLA-Permissive-1.0**, and the underlying
photographs are **CC BY 4.0** per the Snapshot Safari organisation page. That means attribution to Beery,
Cole and Gjoka (2020) *and* to Snapshot Safari wherever imagery is reproduced, with no non-commercial and
no share-alike restriction, and CDLA §3.4 leaves the trained model itself unencumbered. Verbatim quotes,
heading boundaries and the corrected misreading are in `docs/LICENCE_EVIDENCE.md`. Full details and every
measured statistic are in `docs/DATASET_CARD.md`.

Three properties drove the design:

1. **Bursts.** A visit produces median 5, maximum 10 near-identical frames, so splits must group whole
   visits. The benchmark's own train/val/test are disjoint on camera site and on date, which is a stronger
   guarantee than a random split, so they are used as-is (199 / 32 / 44 sites).
2. **Empty frames.** 34.2 % of images, and 191 of the 633 test camera-days contain no animal — the
   false-alert failure mode is measurable rather than hypothetical.
3. **No boxes.** Species labels only. Detection therefore uses a second corpus
   (`ankanghosh651/...-yolo-format`, MIT, buffalo/elephant/rhino/zebra), and that switch is declared.

Preprocessing: images resized to 224 px for classification (source files are 448 px high, measured),
ImageNet normalisation, train-only augmentation (random resized crop 0.5-1.0, horizontal flip, colour
jitter, random grayscale, Gaussian blur). Seed 13 everywhere; manifests carry SHA-256 prefixes recorded in
`configs/experiments.yaml`.

## 3. Method

* **Baseline** — frozen ImageNet ResNet-50 features + class-balanced logistic regression. No deep
  training at all, which makes it a fair reference for "what does the pretraining already give me".
* **Proposed species model** — the same backbone with the head replaced by `Linear(2048, 13)` and
  `layer4` unfrozen, cross-entropy with inverse-frequency class weights and label smoothing, AdamW at
  1e-4 for the head and 1e-5 for the backbone, batch 32, checkpoint selected on validation macro F1.
* **Detector** — YOLOv8n fine-tuned on the de-leaked lists, 50 epochs at 640 px.
* **Alert rule** — `src/sequence/alert.py` scores a camera-day from tier-weighted, exponentially
  recency-decayed, night-biased visit evidence; `src/sequence/forecast.py` is the forward-looking version,
  predicting a qualifying visit in the next 6 hours from the site's own history.
* **Fundamentals module** — `src/from_scratch/`: numpy tensors, broadcasting, SVD, numerical gradient
  check, McCulloch-Pitts and sigmoid neurons, an MLP with hand-derived backpropagation, batch GD / SGD /
  momentum / Nesterov, Hebbian, modified Hebbian, competitive and error-correction learning, a Boltzmann
  machine trained by contrastive divergence, and initialisation / normalisation / regularisation ablations.
  This is what the course list demands beyond calling a pretrained model.

## 4. Results

| Model | val macro F1 | test macro F1 (13 classes) | test macro F1 (classes with ≥50 test images) | test accuracy |
| --- | --- | --- | --- | --- |
| majority class / random | 0.029 / 0.061 | 0.020 / 0.065 | — | 0.150 / 0.073 |
| Baseline: frozen features + logistic | 0.5400 | 0.6245 | — | 0.7623 |
| Fine-tune, 8 epochs, no selection (nb 02) | 0.5520 | 0.6043 | — | 0.7320 |
| Fine-tune + augmentation + label smoothing + best checkpoint (nb 07 arm A, patience-stopped at ep 19) | 0.5828 | **0.6461** | 0.7000 | 0.7752 |
| Frozen backbone, trainable head, same budget (nb 07 arm B) | 0.4979 | 0.5702 | 0.6178 | 0.7136 |
| 30 full epochs, no early stopping (nb 07 rerun) | 0.5828 | 0.6403 | 0.6937 | 0.7751 |

Rank quality, from `notebooks/10_results_showcase.ipynb` (rebuilt from existing artifacts, no retraining):
accuracy **0.7762**, micro/macro ROC AUC **0.9394** on the same held-out sites, validation loss 1.1436 at the
selected epoch 11. Per-epoch validation *loss* was not logged by the 30-epoch run, so it is reported at the
checkpoint only; `src/vision/wildlife.py` now records it per epoch for any future run. The notebook also renders
the labelled-frame showcase (true → predicted species, confidence, danger tier) and one site-day alert decision.

Detection on the de-leaked lists: **mAP@0.5 0.9469**, mAP@0.5:0.95 0.7940, precision 0.9125,
recall 0.9144; per-class AP@0.5 buffalo 0.949, elephant 0.938, rhino 0.942, zebra 0.958. Edge proxy:
26.8 ms/frame (37.3 fps), peak GPU memory 149.4 MB, 1.53 detections/frame on an RTX 3060 Laptop.

Alert rule on 633 test camera-days (139 with a qualifying species present), judged against the true
labels rather than the rule's own predictions: per-frame alerting fires 173 alerts with 58 false
(precision 0.665, recall 0.827); the retrospective gate at the pre-registered settings fires 74 with 11
false (precision 0.851, recall 0.453) — 57.2 % fewer alerts, 81 % fewer false alerts, and a recall cost
that failed the pre-registered 0.90 bar. Notebook 08's forward-looking version, compared at the same
**camera-day** budget, is measured in `notebooks/08_predictive_alert.ipynb` on its own denominators — 625
test camera-days of which **47** contain a qualifying visit inside the 6-hour horizon, alerted at the gate's
own volume of 74 days. At that budget: persistence (danger-tier in the trailing 24 h) 29 true days /
recall 0.6170 / day-level AUC 0.8742; volume 18 / 0.3830 / 0.7113; hour-of-day climatology 8 / 0.1702 /
0.5453; the LSTM 1 / 0.0213 / 0.4352; logistic on the full feature set 5 / 0.1064 / **0.3122, i.e. worse than
the 0.1184 chance expectation**. The evidence ceiling for any rule restricted to the model's own detections is
**0.8723** (41 of the 47 positive days carry a modelled danger-tier sighting; 6 are lost upstream to
classification), so the classifier is not what caps alert recall. The logistic arm's anti-prediction is
attributable and fixable: its largest weight is on `site_qualifying_prior`, constant on unseen sites, with
negative weights on activity — dropping that one column gives 23 true days / 0.4894 / AUC 0.8541, with val
agreeing (0.3756 → 0.5624). That clears the volume bar (0.4894 ≥ 1.10 × 0.383 = 0.4213) and **does not** clear
the pre-registered bar on max(volume, persistence) = 0.6787; the arm is declared exploratory because it was
designed after the first forecast run had been seen. Full table, the tie-break artifact it corrected, and the
reasoning are in `docs/MODEL_CARD.md` ("Predictive forecast") and `PLAN.md` §3.

## 5. Analysis of what failed, and what it taught

1. **The first fine-tune lost to the baseline** (0.6043 vs 0.6245) while improving on validation: it
   adapted to the camera sites it had seen. Fixing the recipe — augmentation, label smoothing, and selecting
   the checkpoint on validation instead of keeping the last epoch — turned it into a +0.0158 win, and the
   frozen-head control (arm B, 0.5702) shows the gain comes from updating `layer4`.
2. **A metric can be wrong in its unit.** The forecast-vs-counting comparison was first run over hour-points
   (top 75 of 7,970), where every rule including chance looks near-zero, and it printed a passing verdict.
   That result was withdrawn and restated in camera-days, the unit an alert queue actually holds.
3. **A metric can be circular.** Defining "false alert" against the rule's own predictions made the gate's
   false-alert count zero by construction, since it requires two qualifying detections to fire. The
   denominator is now the true labels, and the module's self-check contains a case that fails if it is
   changed back.
4. **Rare classes distort the headline.** `tayassu pecari` has 2 test images and scores F1 0.000, costing
   about 0.077 of all-class macro F1, so both variants are reported.
5. **A limitation I asserted before measuring it.** This report claimed the classifier caps alert recall
   because "24 qualifying camera-days produced no big-cat detection". Measured properly at camera-day
   granularity under the 6-hour horizon, 41 of the 47 positive test days *do* carry a modelled danger-tier
   sighting, so the ceiling for any evidence-only rule is 0.8723 and only 6 days are lost upstream. The 24-day
   figure counted hour-granularity points under the retrospective gate's own qualifying-day definition — a
   different question — and it was quoted as if it bounded the forecast. Lesson recorded: state the unit with
   the number, or the number becomes a claim about something else.
6. **A suspiciously good number was checked, not celebrated.** The detector's 0.966 mAP came from a split
   with byte-identical images shared between train and test; hashing found 40 duplicate groups and 17 spanning
   split boundaries (10 train↔test, 6 train↔valid, 1 valid↔test), i.e. 11 of 151 test images, and after
   de-leaking and retraining the honest figure is 0.9469. The leak's measured effect is small (−0.0019 mAP@0.5),
   which is the point: it was corrected because it was wrong, not because it was costing accuracy.
7. **Worse-than-random is a diagnosis, not a difficulty.** The logistic forecast scored 0.3122 day-level AUC
   on held-out sites, below its own 0.1184 chance line. That is not a hard dataset, it is an inverted
   dependency: the biggest weight sat on a site-level prior that is constant once every test site is unseen,
   and the activity weights were negative. Removing that one column took it to 0.8541 AUC / 0.4894 recall, and
   validation moved the same way. Site identity is the thing a camera-trap model most readily memorises and
   the thing that transfers least — which is the whole premise of this benchmark, now demonstrated on our own
   head rather than only cited.
8. **A degenerate rule can print a clean-looking zero.** `site_qualifying_prior` first reported exactly 0.000
   recall. All 625 days shared one score and the tie was broken by calendar order, so the "rule" had selected
   the earliest days. Tie-breaking is now a seeded draw and the table prints `tied_at_cut`; the value became
   0.0638, and the guard was validated against the old behaviour on a constructed all-tied case where the
   calendar order yields 0 and the seeded order yields 4 against a 2.75 chance expectation.
9. **An independent recomputation contradicted my own explanation, so the explanation was withdrawn.**
   Reloading arm A's saved checkpoint outside its notebook and re-deriving the test metric gives 0.6431 at batch
   32 and 0.6432 at batch 128, while notebook 07's own artifact for the same run says 0.6403. I had written that
   a similar gap came from two training runs selecting different checkpoints; that is falsified here, because
   both numbers come from the same 30-epoch run and the batch-size pair agree to 0.0001. The two live
   candidates — the saved `.pt` not matching the weights the notebook evaluated (save/restore ordering in
   `train_classifier`), or the two loaders scoring different row sets — are recorded as an open item rather than
   guessed at. Until it is settled the fine-tune's margin is quoted as the notebook's own **+0.0158** over the
   frozen baseline, with anything in 0.640–0.643 read as ±0.003. Evidence:
   `outputs/logs/armA_gate_recheck/recompute.jsonl`, `scripts/check_armA_gate.py`.

## 6. Responsible use

Intended: triage of camera-trap imagery to prioritise conflict-risk review by a human. Not intended:
locating or approaching endangered animals, autonomous deterrence, or any action without a ranger's
judgement. Known limitations: geographic and taxonomic bias of one camera network, daylight bias with 32 %
night frames handled as a separate error-analysis axis, no ground-truth conflict labels anywhere (the risk
score is a declared, justified function and is evaluated by counted false and missed alerts, never claimed
as validated risk prediction), and power figures are a laptop latency/memory proxy rather than a field
measurement. Incidental images of people are excluded from demonstrations and from any alert path.

## 7. Reproduction

```
.venv/Scripts/python.exe DL_PBL/src/vision/dataset_audit.py
.venv/Scripts/python.exe DL_PBL/src/vision/make_splits.py
.venv/Scripts/python.exe DL_PBL/tests/test_from_scratch.py
.venv/Scripts/python.exe DL_PBL/tests/test_app.py
.venv/Scripts/python.exe DL_PBL/scripts/check_notebooks.py
bash DL_PBL/scripts/local_full_run.sh            # full-corpus training
bash DL_PBL/scripts/phase_i_snapshot.sh phase_i  # refuses stale or errored notebooks
```
