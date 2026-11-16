# 3A Tab: PRD and Eval Spec (v0.1, 2026-11-16)
Status: draft for review. Passes the Phase 3 gate mechanically (every ID has a planned check, section 10).
Companion docs: "3A MVP State and Brief" (status, scope, kill criteria), "3A Evidence: Spike and Dataset Freeze" (numbers behind the decisions).

## 1. Problem
Tabular AutoML systems are ranked by leaderboard position. Readers rarely learn when a system loses, whether more search time actually helps, or whether the data was leaky to begin with. Tab is a small AutoML system for tabular classification whose product is trustworthy measurement. Every claim on its results page traces to a check and a number a stranger can regenerate.

## 2. Readers (portfolio equivalent of users)
| Reader | Must be able to do | Time they give it |
|---|---|---|
| Recruiter | Find the headline finding, the table of losses and a green repro badge | 2 min |
| Researcher | Inspect the pre-registration, rerun the smoke repro, check the statistics | 1 hour |
| Founder | Read cost and latency per dataset size, and a plain "when not to use it" section | 10 min |

## 3. Goals, done-definition, guardrails
Goals
- G1 Verifiable: one command regenerates a smoke version of the results and checks them against committed references.
- G2 Honest comparison against FLAML and AutoGluon on the frozen 20 datasets, losses shown.
- G3 One pre-registered finding: does plateau-based budget stopping save compute without a detectable score loss.
- G4 A leakage detector with recall measured per injected leak type.

Done means all of the following
- Every check in section 10 exists and is green, with zero skipped tests.
- The benchmark ran on all 20 datasets x 5 folds with no dataset dropped or replaced. Crashes appear in the table as crashes.
- The pre-registration file was committed before the first benchmark result (HF-6).
- The results page contains every publish-standard item: problem and architecture diagram, rejected design tradeoffs, benchmark with methodology and hardware, failure cases, cost and latency analysis, one-command repro.
- The smoke repro passes from a clean clone on a machine that is not the Omen.

No target for beating a baseline is set. Publishing the interval is the success, whichever way it falls.

Guardrails (must not regress)
- Frozen dataset list hash unchanged (HF-2).
- Tab wall-clock within 1.10 x the ceiling on at least 95% of benchmark runs (NFR-1).
- Floor check: Tab's cross-dataset skill score must beat default sklearn HistGradientBoosting at a 95% interval that excludes zero. If not, the report says tuning did not pay.

## 4. Non-goals
See the state doc. Plus: automatic time-column detection (the user names the time column), regression, serving, UI.

## 5. Stories (one outcome each, each sized for one build session)
- US-1 Fit and predict. Given a DataFrame (numeric, categorical, boolean, missing values; 2 to 10 classes; up to 50,000 rows x 200 columns) and a time ceiling, when fit runs, then predict_proba returns rows summing to 1 within 1e-9 with columns in class order, and fit reports actual wall-clock including refit.
- US-2 Preprocessing inside the fold. Given any training split, when the preprocessor is fitted, then no statistic (mean, level set, imputation value) depends on validation or test rows.
- US-3 Nested CV. Given outer 5-fold and inner 3-fold, when tuning runs, then no tuner, calibrator or stacker ever sees an outer test index.
- US-4 Zoo and leaderboard. Given the zoo (logistic, random forest, histogram gradient boosting, small MLP), when a run finishes, then leaderboard.json lists each model's mean and interval log loss and wall-clock.
- US-5 Calibration. Given a fitted ensemble, when calibration runs on inner folds, then ECE (15 equal-width bins) is reported before and after, and the outer fold is untouched.
- US-6 Permutation importance. Given a fitted model and held-out data, when importance runs, then a feature that alone determines the label ranks first.
- US-7 Budget controller. Given a ceiling, a window W and threshold delta, when the best inner log loss improves by less than delta (relative) over the last W trials, then search stops, the best model so far is refitted, and the stop reason is logged. The ceiling always wins.
- US-8 Leakage detector. Given a training table and target, when detect runs, then it returns scored column flags covering four types (target copy, near-copy, row-order or ID leak, train/test overlap) and states its false-positive rate on null columns.
- US-9 Time-aware splits. Given a named time column, when the splitter runs, then every validation row's timestamp is at or after the latest training timestamp, with an optional gap.
- US-10 Stacking. Given at least two base models, when stacking fits, then the meta-learner trains only on out-of-fold predictions for rows the base model did not see.
- US-11 Benchmark harness. Given the frozen list, when it runs a system, then predicted probabilities are saved to disk and scored by evalkit alone, and results.json records actual seconds, nominal budget, CPU count and versions.
- US-12 Cost and memory. Given a run, when it ends, then peak memory across the whole process tree and wall-clock are recorded.
- US-13 Repro. Given a clean clone, when make repro runs, then a smoke benchmark (2 datasets, 1 fold, 20 s ceiling, three systems) finishes and its scores sit inside the committed reference intervals. make repro-full regenerates everything.
- US-14 Results page. Given results.json, when the page builds, then it fails if any publish-standard section lacks data.

## 6. Non-functional requirements (numbers)
- NFR-1 Budget compliance: actual wall-clock at most 1.10 x ceiling on at least 95% of runs. Overruns are reported, never hidden.
- NFR-2 Determinism: in trial-count mode with a fixed seed, two runs give bitwise-identical predictions on one machine. Time mode is for the benchmark only.
- NFR-3 Memory: peak at most 8 GB on the largest frozen dataset (machine has 16.8 GB).
- NFR-4 Speed of feedback: unit and canary tests under 3 min on the Omen, full CI under 10 min.
- NFR-5 Envelope: inputs beyond 50,000 rows, 200 columns or 10 classes raise a clear error, never degrade silently.
- NFR-6 Independence: Tab's package imports neither AutoGluon nor FLAML. They live only in the benchmark folder.
- NFR-7 Smoke repro under 15 min.

## 7. Hard-fail cases (must never happen)
- HF-1 Any outer-fold row influences preprocessing, tuning, calibration or stacking.
- HF-2 The frozen dataset list changes after freeze.
- HF-3 A system is scored with its own metric instead of evalkit on saved probabilities.
- HF-4 A headline claim rests on AUC or accuracy when log loss does not support it.
- HF-5 A table reports only nominal budget without actual seconds.
- HF-6 Results exist in git history before the pre-registration commit.
- HF-7 Tab imports a competitor.

## 8. Known failure modes (to show, not hide)
Time-bounded baselines are not repeatable run to run. Small test sets (credit-g has 100 test rows per fold of its own split) make per-dataset verdicts noisy. Ceiling datasets produce ties. electricity is probably time-ordered, so random folds inflate every system there. The leakage detector can flag legitimately strong features. Calibration can hurt on tiny data. All five get a section on the results page.

## 9. Eval spec (about 50 cases, written before the code)
Deterministic checks only. No LLM judge is involved anywhere in this project.
- Leakage cases (30): 4 types x 3 severities x 2 base tables (one synthetic, one real development table) = 24, plus 6 clean negatives (random-noise columns).
- Controller curves (8): plateau early, plateau late, noisy plateau, steady descent, rising (overfit), flat from trial 1, sawtooth, single trial.
- Splitter cases (6): tied timestamps, non-monotone index, missing timestamps, gap larger than data, single-class validation fold, duplicate timestamps.
- Data robustness (10): unseen class in a fold, constant column, all-missing column, categorical with over 1,000 levels, boolean dtype, mixed dtypes, duplicate column names, 10 classes, extreme values (1e30), NaN in target (must raise).
- Full injected-leak study (240 cases): the 12 variants above on each of the 20 frozen datasets, one seed. Reports recall per type and severity with Wilson intervals, plus false-positive rate from a null-column injection on every dataset. Unmodified real datasets are scanned and flagged items listed for manual review, separate from the recall numbers.

Leak variants: T1 target copy (exact, affine transform, label-encoded copy). T2 near-copy (target plus noise at sigma 0.1, 0.5, 1.0 times target std; flip probability 0.05, 0.2, 0.4 for categorical targets). T3 row-order leak (fully sorted, 80% sorted in blocks, shuffled blocks of 50). T4 train/test overlap (1%, 5%, 20% of test rows duplicated in train).

"Simple" set for the 70% kill criterion: T1 exact, T1 affine, T3 fully sorted, T4 20%.

## 10. Requirement to check table
| Req ID | Check type | Where it will live |
|---|---|---|
| US-1 | contract test on synthetic and tiny real data | tests/contract/test_fit_predict.py |
| US-2 | unit test plus canary (leaky variant must fail the test) | tests/canary/test_preproc_leak.py |
| US-3 | spy test: recorded tuner indices disjoint from outer test | tests/unit/test_nested_cv.py, tests/canary |
| US-4 | schema test plus each model fits tiny data | tests/contract/test_leaderboard.py |
| US-5 | ECE vs hand-computed values; property: calibrated ECE not worse on a synthetic miscalibrated model; outer fold untouched | tests/unit/test_calibration.py |
| US-6 | planted-feature test, 20 seeds, top rank in at least 19 | tests/eval/test_importance.py |
| US-7 | the 8 synthetic curves, ceiling never exceeded, best-so-far returned | tests/eval/test_controller.py |
| US-8 | 30 leakage cases; kill criterion on the simple set | tests/eval/test_leak_cases.py, bench/leak_study.py |
| US-9 | property test over the 6 splitter cases | tests/unit/test_time_split.py |
| US-10 | OOF provenance test plus canary; ablation on tier 1 | tests/canary/test_stack_leak.py, bench/ablation.py |
| US-11 | stub system (class prior) end to end on 2 tiny datasets; scorer vs sklearn log_loss and roc_auc_score; skill-score formula test | tests/contract/test_harness.py, tests/unit/test_evalkit.py |
| US-12 | subprocess that allocates 500 MB, measured within 15% | tests/unit/test_cost.py |
| US-13 | CI job on a clean checkout | ci/repro-smoke |
| US-14 | build fails on missing section data | tests/contract/test_page.py |
| NFR-1 | harness asserts ratio on every run, flags in results.json | bench/ and tests/contract |
| NFR-2 | two trial-count runs compared byte for byte | tests/unit/test_determinism.py |
| NFR-3 | peak memory logged on the largest dataset, threshold check in the report build | bench/ |
| NFR-4 | CI timing gate | ci/ |
| NFR-5 | oversize inputs raise | tests/eval (data robustness) |
| NFR-6 | import test over the tab package | tests/unit/test_no_competitor_imports.py |
| NFR-7 | CI timing gate on smoke | ci/repro-smoke |
| HF-1 | the canary suite (every stage has a deliberately leaky variant) | tests/canary/ |
| HF-2 | CI hash check of selected_datasets.json | ci/ |
| HF-3 | harness scores only from saved probabilities; import test bans metric calls inside systems | tests/contract/test_harness.py |
| HF-4 | report generator test: AUC win with log-loss tie must print "tie" | tests/contract/test_headline.py |
| HF-5 | results schema requires actual_wall_s | tests/contract/test_schema.py |
| HF-6 | script compares commit order of PREREG.md and first results | ci/prereg-order |
| HF-7 | same as NFR-6 | tests/unit/test_no_competitor_imports.py |

## 11. Pre-registration file (prereg/PREREG.md, committed before any benchmark result)
- Headline metric and aggregation. Skill score S = 1 minus (model log loss divided by log loss of the training-fold class-prior predictor). Per dataset, average over the 5 folds. Headline per baseline: mean over the 20 datasets of (S_Tab minus S_baseline) with a 95% percentile bootstrap over datasets (10,000 resamples, fixed seed). Raw log loss and AUC are shown beside it and cannot override it. Per-dataset win, tie or loss from the 95% bootstrap interval over the 5 fold differences.
- Primary plot: skill score versus actual seconds.
- Reference rows: sklearn default HistGradientBoosting and logistic regression, seconds each, as floors.
- Budget-controller study: Tab with a fixed 300 s ceiling versus Tab with the controller under the same ceiling, tier 2 (8 datasets) x 5 folds. Outcomes: compute saved (ratio of actual seconds) and score change (S difference). Success needs at least 15% compute saved with the interval excluding zero AND the lower bound of the score change above the non-inferiority margin. Margin procedure: half the median, across development datasets, of the fold-to-fold standard deviation of S. Otherwise it is reported as a null or partial result.
- Development pool: the 34 eligible but undrawn CC18 tasks. Twelve of them, drawn with RandomState(20261118) over task ids sorted ascending, are the only data used to set W, delta, the detector's flag threshold and the calibration choice. They never appear in reported results.
- Leakage study design from section 9, with the simple-set definition.
- Time-aware study: if electricity has a usable time ordering, compare random folds against forward-chaining folds (5 splits) for Tab and report the optimism gap with an interval. Plus a synthetic drift set with a known onset.
- Stacking ablation on tier 1, paired, with intervals.
- Disclosure: credit-g, GesturePhaseSegmentationProcessed and adult were seen in the spike. Spike numbers are excluded from all analyses.

## 12. Glossary
Ceiling: maximum wall-clock allowed, a limit and not a target. Actual seconds: measured time including refit. Tier 1: the 20 frozen datasets. Tier 2: the 8 long-budget datasets. Skill score: see section 11. Simple set: see section 9. OOF: out-of-fold prediction. Canary: a deliberately leaky implementation that a test must catch.

## 13. Defaults I chose that you can veto
1. Development pool of 12 undrawn datasets sets every tuned parameter, so none are tuned on the 20.
2. Two cheap floor baselines (default HistGradientBoosting, logistic regression) added to the benchmark.
3. Non-inferiority margin set by a procedure on development data, not a number picked now.
4. The user names the time column; no auto-detection in v1.
5. Preprocessing leak is covered by the canary tests, not by the data-level detector.
6. No LLM judge anywhere.
