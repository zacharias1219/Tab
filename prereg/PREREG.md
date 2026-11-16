# Tab pre-registration (PREREG.md)

Committed before any benchmark result exists in git history (HF-6, checked by `ci/prereg_order.py`).
Version 1, 2026-11-16. Any later change to this file is an amendment: it gets a dated entry in the
"Amendments" section at the bottom, and results produced before the amendment are labelled as such.

## 1. Headline metric and aggregation
- Skill score S = 1 - (model log loss / log loss of the training-fold class-prior predictor).
  Log loss uses probabilities clipped at 1e-15 and the full dataset's class list (evalkit, `evalkit.metrics`).
- Per dataset: S averaged over the 5 outer folds.
- Headline per baseline B: mean over the 20 frozen datasets of (S_Tab - S_B), with a 95% percentile
  bootstrap over datasets (10,000 resamples, seed 20261116).
- Raw log loss and AUC (binary only) are shown beside the headline and cannot override it (HF-4).
- Per-dataset verdict: win / loss if the 95% percentile bootstrap interval (10,000 resamples, seed
  20261116) of the mean paired fold difference in S excludes zero, otherwise tie.
- A failed run (error, timeout, oom) on a fold is kept in the table. For the headline it scores as the
  class-prior predictor on that fold (S = 0), the most a system that produced nothing could claim.

## 2. Primary plot
Skill score versus actual seconds (log x axis), one point per (system, dataset), mean over folds.

## 3. Systems and budget
- Tab (this repo), FLAML 2.8.0, AutoGluon 1.6.3 (`presets="medium_quality"` default, no other tuning).
- Floors: sklearn default HistGradientBoostingClassifier and default LogisticRegression (with the same
  in-fold preprocessing as Tab), run to completion, actual seconds recorded.
- Class-prior stub (skill 0 by construction), used to validate the harness.
- Every system: same ceiling (60 s for tier 1), same CPU count, same machine, one subprocess per
  (system, dataset, fold). Compared on actual seconds measured by the runner, never on nominal budget.
- Kill at 1.5 x ceiling; status `timeout` is recorded and kept.

## 4. Datasets
- Tier 1: the 20 frozen CC18 tasks in `bench/selected_datasets.json`, folds 0-4 of the OpenML task split.
  The sha256 of that file is locked in `bench/selected_datasets.sha256` and checked in CI (HF-2).
- Tier 2: the 8 long-budget tasks listed in the same file.
- No dataset is dropped or replaced. Crashes appear in the table as crashes.

## 5. Development pool (sets every tuned parameter)
- The CC18 tasks that are eligible but not drawn into tier 1 or tier 2 (34 tasks).
- Twelve of them, drawn by `bench.devpool.draw_dev_pool` with `numpy.random.RandomState(20261118)`
  over task ids sorted ascending, are the only data used to set:
  controller window W, controller threshold delta, the leak detector flag threshold, and the
  calibration choice. They never appear in reported results.
- `bench/tune_dev.py` writes the chosen values to `prereg/tuned_params.json`. That file must be
  committed before the first tier-1 result (same CI order check as this file).
- Until then the library defaults (W=8, delta=0.001, detector threshold 0.9, temperature scaling on)
  are provisional and labelled as such in any output.

## 6. Budget-controller study (G3)
- Arms: Tab with a fixed 300 s ceiling and the controller off, vs Tab with the controller on under the
  same ceiling. Tier 2 (8 datasets) x 5 folds, paired.
- Outcomes: compute saved = 1 - (actual s, controller) / (actual s, fixed), mean over datasets with a
  95% bootstrap interval over datasets; score change = S_controller - S_fixed, same aggregation.
- Success needs BOTH: compute saved >= 15% with the interval excluding zero, AND the lower bound of
  the score change above -m, where m is the non-inferiority margin.
- Margin procedure: m = 0.5 x median over the 12 development datasets of the fold-to-fold standard
  deviation of S for Tab (fixed arm, 60 s). Computed by `bench/tune_dev.py`, recorded in
  `prereg/tuned_params.json`.
- Otherwise reported as a null or partial result. No re-tuning after seeing tier-2 outcomes.

## 7. Leakage study (G4)
- 4 types x 3 severities on each of the 20 frozen datasets, one seed (20261116) = 240 cases.
  - T1 target copy: exact, affine (2y+3 on the label code), label-encoded copy (shuffled code mapping).
  - T2 near-copy: label code + Gaussian noise at sigma 0.1, 0.5, 1.0 x std(label code) for binary
    and multiclass alike; plus flip variants at p = 0.05, 0.2, 0.4 for categorical injections.
    The numeric variant is the reported T2 row; flip variants are reported beside it.
  - T3 row order: fully sorted by target, 80% sorted in blocks, shuffled blocks of 50.
  - T4 train/test overlap: 1%, 5%, 20% of test rows duplicated into train.
- Recall per type and severity with Wilson 95% intervals; pooled recall on the simple set
  {T1 exact, T1 affine, T3 fully sorted, T4 20%}.
- Kill criterion: pooled simple-set recall below 70% -> the detector is cut to a documented heuristic.
- False-positive rate from injecting 5 null columns (normal, uniform, categorical, integer id-like
  shuffled, boolean) into every dataset.
- Unmodified datasets are scanned too; flagged columns are listed for manual review, separate from
  the recall numbers.

## 8. Time-aware study
If electricity has a usable time ordering (checked by `bench/time_study.py --check`), compare random
5-fold vs forward-chaining 5-split evaluation for Tab and report the optimism gap
(S_random - S_forward) with a bootstrap interval over splits. Plus a synthetic drift set with a known
onset (`bench.time_study.synthetic_drift`). If electricity has no usable ordering, the study reports
that and runs on the synthetic set only.

## 9. Stacking ablation
Tab with stacking vs Tab without, tier 1, paired over datasets, same aggregation as the headline.

## 10. Guardrails (reported, not tuned)
- NFR-1: actual wall-clock <= 1.10 x ceiling on >= 95% of Tab runs. Overruns listed.
- Floor check: Tab's mean skill must beat default HistGradientBoosting with the interval excluding
  zero; if not, the report says tuning did not pay.

## 11. Disclosure
credit-g, GesturePhaseSegmentationProcessed and adult were seen in the feasibility spike. Spike numbers
are excluded from all analyses.

## Amendments
(none)
