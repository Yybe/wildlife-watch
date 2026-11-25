# Individual Contribution Record — Project 11

Name: Shivam _(roll no.)_ · Course: CSAI3027P Deep Learning · Phase: I · Date: 2026-10-24

The hackathon rules make this an individual submission and require every external contribution to be
disclosed. This file states what was built, in what order, and what I must be able to explain without
looking at the code. Tool and AI assistance is logged prompt-wise in `docs/SOURCE_LOG.md`; the code and
measurements below are the work being submitted.

## What I designed and decided

1. **Problem framing and success metric.** Chose alerts per camera-day and macro F1 on unseen camera sites
   as the criteria, after measuring that 34.2 % of the corpus contains no animal, which makes plain accuracy
   meaningless. Wrote the acceptance bars before running the comparisons they judge.
2. **Dataset choice and audit.** Selected iWildCam v2.0 from the candidates by inspecting real file
   contents, rejected a mislabelled mirror (`silviamatoke/serengeti-dataset`, titled Serengeti but
   containing *Agouti* folders), traced the licence, and produced `notebooks/01` plus
   `docs/DATASET_CARD.md` from the label files rather than the dataset description.
3. **Split policy.** Used the benchmark's official splits after verifying with filename matching that all
   splits' images are present on the mirror, and kept a recarved holdout variant as a robustness check.
   Class capping is per whole sequence, so no burst of frames straddles a boundary.
4. **Classifier design and the correction that mattered.** Baseline of frozen features plus logistic
   regression, then a fine-tune that first *lost* to it out of distribution; diagnosed the cause as site
   overfitting and fixed the recipe (augmentation, label smoothing, validation-selected checkpoint), which
   produced 0.6461 versus 0.6245. The head-only control isolates the reason.
5. **Detection arm on a second corpus, and its de-leaking.** Found 17 cross-split byte-identical duplicate
   groups, rebuilt the splits, retrained, and reported the lower honest number (0.9469) alongside the
   inflated one.
6. **The alert rule as a designed function.** Danger tiers, recency decay, night weighting and a
   two-visit persistence requirement, written down as a specification with a pre-registered target; then
   measured, found to tie a counting baseline at equal alert volume, and the innovation re-scoped to a
   forecast.
7. **Fundamentals module.** The numpy tensor/SVD/gradient-check sandbox, McCulloch-Pitts and sigmoid
   neurons, MLP with hand-derived backpropagation, four optimisers, Hebbian / modified Hebbian /
   competitive / error-correction demonstrations, a contrastive-divergence Boltzmann machine, and the
   initialisation / normalisation / regularisation ablations, each with a failing-test check in
   `tests/test_from_scratch.py`.
8. **Measurement discipline in the alert module.** Budget ties are broken with a seeded draw and every
   ranking table prints `tied_at_cut`, so a rule whose ordering is decided by ties cannot be reported as a
   result; day-level ROC AUC is computed alongside budget-matched recall specifically to separate "the features
   carry no signal" from "the aggregation loses it". Both were added after they caught something real.
9. **Prototype.** `src/app/predict.py`: input validation, species prediction with confidence and runner-up,
   danger tier, and a risk decision from site history; `tests/test_app.py` covers the rejection paths.

## What I measured (all reproducible)

| Claim | Evidence |
| --- | --- |
| Baseline 0.6245 / fine-tune 0.6043 / regularised fine-tune 0.6461 test macro F1 | `notebooks/02`, `07`; `outputs/logs/notebook07/summary.csv` |
| Detector mAP@0.5 0.9469 de-leaked, 0.9487 as shipped | `notebooks/05`; `outputs/logs/notebook05/experiment_log.jsonl` |
| 26.8 ms/frame, 149 MB peak on laptop GPU | `notebooks/05` |
| Gate: 74 alerts / 11 false / recall 0.453 vs per-frame 173 / 58 / 0.827 | `notebooks/04` |
| Gradient check max abs error 3.5e-10 (test harness) | `tests/test_from_scratch.py` |
| Optimisers on the wildlife tabular problem: batch GD 1.2104, SGD 1.2115, momentum 0.2804, Nesterov 0.2949, with batch GD seeing 324,600 samples against 12,513 for the rest | `notebooks/03` |
| Optimisers on the test harness problem: 1.356 / 1.357 / 0.393 / 0.419 — a different dataset, quoted only here | `tests/test_from_scratch.py` |
| Hebbian norm grows 0.146 to 1.26e4 unbounded; modified Hebbian saturates at 1.416 | `notebooks/03` |
| 21 fundamentals checks + 7 app checks pass | `python DL_PBL/tests/test_from_scratch.py`, `test_app.py` |

## Negative results I am reporting rather than hiding

* The first fine-tune was worse than doing no deep training at all out of distribution.
* The pre-registered 0.90 alert-recall bar was unreachable on the gate's own denominator (per-frame alerting
  recovers 0.827 of its qualifying camera-days). I also asserted that the classifier caps *forecast* recall at
  24 days — wrong: at camera-day granularity under the 6-hour horizon the ceiling is 0.8723 (41 of 47 positive
  days carry a modelled danger-tier sighting), so the forecast's failure is not explained by the classifier.
* The retrospective temporal gate does not beat a rule that merely counts detection frames when both are
  given the same alert budget.
* A false-alert metric defined on the rule's own predictions reported zero false alerts by construction.
* A "passed" forecast benchmark was an artefact of budgeting in hour-points instead of camera-days and was
  withdrawn.
* The trained forecast **lost** to the pre-registered bar (0.4894 against 0.6787 required). It beats the
  volume counter and loses to recency, because qualifying visits cluster in time and the trailing-24-hour
  count already carries that signal. Reported as the innovation's measured limit, not smoothed over.
* My own forecast head was worse than random (day-level AUC 0.3122) because it had memorised *site identity*
  through a site-level prior whose value is constant on unseen sites. Removing that feature is a measured
  improvement (0.8541 AUC, val-agreeing) but is declared exploratory, since it was designed after seeing the
  first forecast run.
* A rule that reported exactly 0.000 recall was not measuring anything: every test day shared one score and
  the tie was broken by calendar order. Tie-breaking is now seeded and `tied_at_cut` is printed next to every
  recall, so a tie-decided ranking cannot be quoted as a result.

## Assistance used

An AI coding assistant helped locate datasets, write and debug code, run experiments, and draft
documentation; the dataset findings, both metric bugs, the leak in the detection set, the fine-tune
failure, and the re-scoping of the innovation are recorded with the evidence that produced them in
`docs/SOURCE_LOG.md`, including the prompt-wise table.

## Viva checklist — each item must be answerable from memory

1. Why is accuracy useless here, and what replaces it? (34.2 % empty frames; macro F1 and alerts/day)
2. Why group splits by sequence and camera site, and what does the test split actually measure?
3. Forward shape trace: 224×224×3 → stem /2 → stages /32 → 7×7×2048 → pool → 13.
4. Why the frozen-feature baseline beat the first fine-tune, and what exactly fixed it.
5. Derivative of the softmax cross-entropy output layer; why `label_smoothing` was added.
6. Difference between Hebbian, competitive and error-correction learning; what signal drives each.
7. What SVD was used for and the rank that captured 95 % of variance.
8. mAP versus IoU; why a byte-identical duplicate across splits inflates mAP, and by how much here.
9. The alert rule's four terms, its two failure directions, and why 12 lone-sighting suppressions matter.
10. Which numbers are proxies, not measurements: latency/memory standing in for edge power, and the
    absence of ground-truth conflict labels.
11. What the forecast bar was pre-registered as, which counter it cleared and which it lost to, and why
    recency is the hard baseline on clustered wildlife visits.
12. Why my logistic head scored *below* chance on held-out sites, which feature caused it, and what removing
    it did to test and validation AUC. Expect "why is a site-level prior useless here" — the test split is 44
    camera sites absent from train, so the column is constant by construction.
13. What `tied_at_cut` means, which rule it exposed (the site prior's fabricated 0.000 → 0.0638), and how the
    guard was validated against the pre-fix behaviour.
