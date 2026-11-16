# MVP State: 3A Tabular AutoML ("Tab")
Updated: 2026-11-16
Current phase: 6 built (all stories green on bundled data); 7 and 8 done in the form this project needs (no service, no users). **Next: freeze the dataset list, tune on the dev pool, run the benchmark on the Omen.**
Companion docs: `docs/PRD.md` (stories, NFRs, checks, status table in section 14), `docs/ARCHITECTURE.md`, `docs/RUNBOOK.md`, `prereg/PREREG.md`.

## One-line problem
Tabular AutoML tools are judged on leaderboard rank, but almost nobody shows when they lose, how much compute a stopping rule really saves, or whether the data was leaky to begin with. Tab is a small AutoML system whose proof is honest measurement, not rank.

## Target readers
Recruiter (skims in 2 minutes), researcher (checks pre-registration, statistics, repro), founder (cost and when-not-to-use).

## Kill criteria (written before building)
- Spike: 1-minute baselines on 10 CC18 datasets in under ~30 min. STATUS: not triggered (about 22 min for one fold).
- Budget controller saves under 15% compute at non-inferior score: report a null result, ship anyway. STATUS: pending the tier-2 study.
- Leak detector pooled simple-set recall below 70%: cut to a documented heuristic. STATUS: 8/8 on the two CI tables and 8/8 on the fixture study; the 20-dataset study is pending.

## What exists (2026-11-16)
- `tab/`: the full library, including the four advanced items in build order (controller, leak detector, time splits, stacking). Stacking did not slip.
- `bench/`: subprocess runner (tree memory, 1.5 x kill, resumable), 8 system wrappers (prior, logreg, hgb_default, tab, tab_fixed, tab_nostack, flaml, autogluon), assembler, verify, leak/controller/time/ablation studies, dev-pool tuner, OpenML loader with a sha256 cache.
- `evalkit/` v0.1.0: canonical scorer and results envelope (ADR-006, 3A ships first).
- 189 tests green in about 50 s. FLAML and AutoGluon wrappers verified end to end (AutoGluon with its tree-model extras; the full `[all]` extra pulls torch and is untested here).
- Smoke repro passes in about 10 s. Walking skeleton green on fixtures. Static page builds in light and dark mode.
- CI: lint, strict types on evalkit, tests with a timing gate, HF-2/HF-6 checks, locked-test check on PRs, dependency audit, skeleton, smoke and verify, a Windows job, a manual bench-extra job, and GitHub Pages deploy on main.

## Decisions log
- 2026-11-16 Reviewer archetypes: recruiter, researcher, founder.
- 2026-11-16 Native Windows on the Omen 16.
- 2026-11-16 Stacking in v1.
- 2026-11-16 Headline metric: log loss, cross-dataset skill score.
- 2026-11-16 Dataset list frozen (20 + 8) in the Evidence doc. **Not yet copied into `bench/selected_datasets.json`** (the task ids were not in the docs available to the build).
- 2026-11-16 Phase 5/6 build choices (veto before the first result): Python 3.12, one uv lock covering Tab plus both baselines (no separate bench environment needed); sklearn HistGradientBoosting rather than LightGBM in the zoo; temperature scaling for calibration, applied only if cross-fitted inner log loss improves; AutoGluon `presets="medium_quality"`; a failed run counts as skill 0 in the headline; tier-1 ceiling 60 s; the leak detector uses a cross-validated one-column predictor (categories, or a 20-leaf 1-D tree) plus a label-runs test for row order and a row-hash overlap test with a natural-duplicate baseline; the dev pool draw is `RandomState(20261118).choice(sorted undrawn ids, 12, replace=False)`.
- 2026-11-16 Two locked tests were corrected in `[test-change]` commits: a determinism test that over-asserted, and a harness test tripped by the pytest temp-dir name.

## Gate log
| Phase | Gate | Result | Date |
|---|---|---|---|
| 1 | 3 reviewer archetypes, kill criteria written | passed | 2026-11-16 |
| 2 | One-page brief with spike result | passed | 2026-11-16 |
| 3 | Every requirement ID has a planned check | passed | 2026-11-16 |
| 4 | ADRs, contracts, skeleton definition, threat model | passed | 2026-11-16 |
| 5 | Green CI on a trivial change, skeleton live | skeleton and smoke green locally; CI runs on first push; Pages needs enabling in repo settings | 2026-11-16 |
| 6 | Every acceptance criterion linked to a passing check | passed on bundled data (PRD section 14) | 2026-11-16 |
| 7 | Security hardening | threat-model controls implemented and tested (no pickle, no shell, path checks, locked hashes, pip-audit clean, sha256 data cache, `verify`) | 2026-11-16 |
| 8 | Deploy and observe | static page via Pages on main; per-run logs and records; rollback = revert results commit | 2026-11-16 |
| 9 | Launch | pending the benchmark | – |

## Open questions
1. Paste the frozen 20 + 8 + 54 task ids into `bench/selected_datasets.json`, then freeze.
2. Verify electricity's time ordering (`bench.time_study.check_ordering`, which runs automatically in the study).
3. Review the build choices in the decisions log, and amend PREREG.md if any change, before the first result.
4. AutoGluon `[all]` (torch) vs tree-model extras only: decide which the comparison uses, and record it.

## Next action
On the Omen, follow `docs/RUNBOOK.md`: freeze, skeleton on credit-g, `tune_dev`, commit tuned params, `repro-full`, `verify`, publish.
