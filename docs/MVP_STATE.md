# MVP State: 3A Tabular AutoML (call it "Tab")
Updated: 2026-11-16
Current phase: 3 done as a draft (PRD and eval spec written, awaiting review). Next: Phase 4, architecture and contracts.
Companion docs: "3A PRD and Eval Spec" (stories, NFRs, checks, pre-registration contents), "3A Evidence: Spike and Dataset Freeze" (spike numbers, frozen dataset list, plan changes).

## One-line problem
Tabular AutoML tools are judged on leaderboard rank, but almost nobody shows when they lose, how much compute a stopping rule really saves, or whether the data was leaky to begin with. Tab is a small AutoML system whose proof is honest measurement, not rank.

## Target readers
Recruiter (skims in 2 minutes), researcher (checks pre-registration, statistics, repro), founder (cost and when-not-to-use). Archetypes, chosen 2026-11-16. Real names get added when outreach starts.

## Kill criteria (written before building)
- Spike shows a 1-minute-budget run of the baselines cannot finish 10 CC18 datasets in under ~30 min on the Omen 16. STATUS: not triggered (about 22 min for one fold).
- Budget controller saves under 15% compute at non-inferior score across datasets with the interval excluding zero: report it as a null result, ship anyway, do not tune until it "works".
- Leakage detector pooled recall below 70% on the pre-registered simple set (T1 exact, T1 affine, T3 fully sorted, T4 20%): cut it to a documented heuristic and say so.

## Scope (in)
Preprocessing inside the fold. Boring model zoo (logistic, random forest, histogram gradient boosting or LightGBM, small MLP). Nested CV tuning (outer 5, inner 3), random search first. Calibration with ECE. Permutation importance. Leaderboard. Advanced, build order: (1) budget controller, (2) leakage detector, (3) time-aware splits, (4) stacking (first to slip to v1.1, with a note). Benchmark: frozen 20 CC18 datasets vs AutoGluon and FLAML, same machine, same ceiling, compared on score versus actual seconds, plus two cheap floor baselines. TabArena-lite slice: default no, revisit at benchmark exit.

## Non-goals
Regression, imbalance special handling, text/image columns, GPU, distributed search, a UI, custom boosting implementation, NAS, beating AutoGluon overall, deep feature engineering, serving, automatic time-column detection.

## Decisions log
- 2026-11-16 Reviewer archetypes: recruiter, researcher, founder.
- 2026-11-16 Native Windows on the Omen 16.
- 2026-11-16 Stacking in v1.
- 2026-11-16 Headline metric: log loss. AUC secondary on binary only. Cross-dataset headline on a scale-free skill score (1 minus model log loss over class-prior log loss), raw log loss shown beside it.
- 2026-11-16 Dataset list frozen (20 + 8). Development pool of undrawn datasets sets all tuned parameters.
- 2026-11-16 Leakage detector also scanned over the 20 real datasets, reported separately from injected-leak recall. Injected types: target copy, near-copy, row-order/ID, train/test overlap. Preprocessing leak covered by canary tests.
- 2026-11-16 Electricity: verify time ordering, use as the time-aware case if it holds.

## Gate log
| Phase | Gate | Result | Date |
|---|---|---|---|
| 1 | 3 reviewer archetypes, kill criteria written | passed (archetypes, not named people) | 2026-11-16 |
| 2 | One-page brief with spike result | passed | 2026-11-16 |
| 3 | Every requirement ID has a planned check | draft passes mechanically, user review pending | 2026-11-16 |

## Open questions
1. User review of the PRD defaults (PRD section 13).
2. Verify electricity has a usable time ordering (check the date column).
3. Phase 4 interface decisions: fit/predict API, predictions file format, results.json schema, repo layout, whether evalkit lives in this repo or in the shared monorepo from 1A.

## Next action
Review the PRD. Then Phase 4: ADRs, contracts (results.json schema, predictions format, Tab API), one-page diagram, threat model, walking skeleton definition.
