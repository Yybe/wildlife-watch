# Project details — Wildlife Species Identification & Conflict Alert System

This file is the deeper companion to the README: what was built, why each choice was
made, what the numbers mean, and where the system fails.

## 1. Problem and user

Camera traps capture hundreds of thousands of unlabelled photos in wildlife reserves.
Rangers cannot review them all, and the operationally urgent question is not "what is
this animal" but **"is a dangerous animal near a settlement right now, or soon?"**

- **Input:** a camera-trap frame (and, for the alert, a site's recent history of frames).
- **Output:** species + confidence + a danger tier, and a per-site-day **risk score** with
  an alert decision.
- **Success criterion:** accurate species ID on *unseen camera sites*, plus an alert rule
  whose behaviour (alerts, false alerts, missed days) is measured fairly.

## 2. Dataset

**iWildCam v2.0** (WILDS benchmark), built on **Snapshot Safari** imagery.

- 203,029 labelled images; cut to **12 species + `empty` = 13 classes**, capped by whole
  sequence (a camera burst is one visit).
- Splits are **by camera site**, never random: train 36,060 / val 10,234 / **test 19,928
  images across 44 sites the model never sees**. This is the WILDS out-of-distribution
  protocol (Koh et al., 2021).
- Reality: 34.2% `empty` frames, ~32% of animal frames at night (infra-red), motion blur,
  extreme class imbalance, fixed height 448 / width 560–796.
- **Audit finding:** the release ships every split's images in one `train/` folder keyed by
  filename UUID — a naive loader would silently train on test data. Caught in notebook 01.
- Licence: CDLA-Permissive-1.0 (release) over CC BY 4.0 (photos). See `docs/DATASET_CARD.md`
  and `docs/LICENCE_EVIDENCE.md`.

## 3. Architecture

A **ResNet-50** backbone (ImageNet-pretrained) with the 1000-class head replaced by a
13-class linear head. Chosen because:

1. **Residual connections** let 50 layers train without vanishing gradients.
2. **Transfer learning** — pretrained features already encode edges/textures/shapes, which
   survive the night-infrared domain shift; even *frozen* they reach macro F1 0.6245.
3. **Compute** — fits an RTX 3060 at batch 32 / 224² inputs; larger backbones couldn't be
   honestly justified on 12 species.
4. **Comparability** — ResNet-50 is the reference backbone in the iWildCam/WILDS papers.

`notebooks/06_architecture_trace.ipynb` draws the layer diagram and records **tensor
shapes** from forward hooks on the trained model (stem → layer1–4 → avgpool → fc), as the
rubric requires.

A separate **YOLOv8n** detector localizes animals (`notebooks/05`), and the alert rule
(`src/sequence/alert.py`) is the decision layer.

## 4. Training

| Setting | Value | Why |
|---|---|---|
| Optimizer | **AdamW** | per-parameter adaptive steps handle the new head vs. pretrained backbone; decoupled weight decay regularizes without distorting the adaptive rate |
| Weight decay | 1e-4 | keeps weights small, resists overfitting 12 classes |
| Learning rate | 1e-4 head / 1e-5 backbone | head learns fast; backbone nudged, not rewritten |
| Batch size | 32 | stability vs. update count on the 3060 |
| Epochs | 30 | validation peaked at epoch 11; ran all 30 as visible evidence |
| Loss | cross-entropy + **label smoothing 0.1** | never tells the model its answer is 100% certain → less overconfident on unseen sites |
| Seed | 13 | bit-reproducible (epoch 1–11 reproduced exactly) |
| Selection | best-validation checkpoint restored | the naive "keep last epoch" run scored 0.6043 — worse than the baseline; this is what the restore prevents |

**Label smoothing** in one line: the target is 0.9 for the true class and 0.1 spread over
the rest, so the model stays appropriately humble on sites it has never seen.

## 5. The innovation — a predictive conflict alert

The classifier is a *photo diary*; the innovation turns it into a *warning system*.

- Each species has a **danger tier** (jaguar 4, puma/ocelot 3, livestock/nuisance 1–2,
  empty 0).
- Per camera site-day, a **risk score** combines: tier × model confidence × night bias
  (×1.5 for 18:00–05:59) × exponential recency (half-life 6 h), merged with a **noisy-OR**
  so repeated sightings add evidence without saturating.
- **Alert** fires when score ≥ 0.5 **and** ≥ 2 qualifying visits — "a lone sighting is not
  an incident" is the false-alert protection.
- The rule is a **declared function, not learned**, because no corpus has conflict ground
  truth. What is measured is its behaviour: alerts, false alerts, and missed days per
  camera-day.

### Measured at equal budget (notebook 08, `results/notebook08/`)

Rangers have finite patrol capacity, so every rule is judged at the **same 74-alert budget**
over 625 test camera-days (47 truly dangerous):

| Rule | Recall of dangerous days |
|---|---|
| Recency-weighted rule (shipped) | **0.617** |
| Learned head after site-fingerprint ablation | 0.489 |
| Volume (count of dangerous sightings) | 0.383 |
| Climatology (time-of-day habit) | 0.170 |
| Random / chance | 0.118 |
| Learned head before fix | 0.106 |
| LSTM forecaster | 0.021 |

The learned LSTM was trained on each site-day's 24-hour event sequence to predict danger in
the next 6 hours; with only ~244 positive examples it memorized training sites and
collapsed — reported honestly. A pre-registered success bar (≥1.10× the best baseline) was
**cleared against volume but not against recency**, and that verdict is stated in the report
and model card rather than spun.

## 6. Limitations and fixes

| Problem | Root cause | Fix / honest status |
|---|---|---|
| Naive fine-tune fell below baseline (0.6043) | last-epoch weights lost OOD robustness | augmentation + label smoothing + best-val restore → 0.6403 |
| Detection mirror leaked test images | 11 of 151 test frames byte-identical to train (7.28%) | byte-hash audit + de-leaked lists → mAP 0.9469 (leak inflated it by only 0.002) |
| Learned alert head scored below chance | largest weight on a *constant* site-prior feature | dropped the column → AUC 0.31→0.85; declared exploratory |
| No conflict ground truth | none exists in any camera-trap corpus | alert is a declared rule; report counts, never claim validated risk accuracy |
| Rare classes / classifier misses cap alert recall | 41 of 47 dangerous days carry a danger-tier sighting | evidence ceiling 0.8723 stated; macro F1 reported over all classes and over classes with ≥50 test images |

## 7. Reproducibility

- Single seed (13) across data splits, training, and tie-breaks; `scripts/verify_claims.py`
  checks that every number quoted in the docs exists in a committed artifact (39/39 pass),
  and `scripts/check_notebooks.py` confirms all notebooks compile and ran clean.
- `tests/` includes a hand-derived gradient check for the from-scratch network, alert-rule
  behaviour assertions, and a tie-break guard that fails if the measurement artifact is
  reintroduced.

## 8. Sources & AI assistance

Datasets, pretrained weights, papers (Beery et al. 2020; Koh et al. 2021), and all AI
coding assistance are disclosed prompt-wise in `docs/SOURCE_LOG.md`. The design,
experiments, analysis, and every defended number are the author's own.
