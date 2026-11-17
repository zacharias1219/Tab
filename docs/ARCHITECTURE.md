# 3A Tab: Architecture and Contracts (Phase 4, v0.1, 2026-11-16)
Companion to the PRD. Everything here exists to make the PRD's checks cheap to write and impossible to fake.

## 1. Shape in one paragraph
A plain Python library (`tab/`) that fits and predicts. A separate benchmark harness (`bench/`) that runs every system, Tab included, as one subprocess per run, measured from outside. Runs communicate only through files: probabilities in, nothing else. A small scoring package (`evalkit/`) turns those files into scores and intervals. Results land in one schema-validated `results.json`, and a static report page is built from it. No server, no database, no network at test time.

## 2. Containers
| Container | Job | Rule that keeps it honest |
|---|---|---|
| `tab/` | The AutoML library: preprocess, zoo, nested CV, controller, calibration, importance, leak detector, time splits, stacking | Imports no competitor and no scorer from the harness (NFR-6, HF-3) |
| `bench/` | Dataset loading, runner, system wrappers, leak study, ablations | The runner, not the system, measures wall-clock and memory (HF-5, US-12) |
| `evalkit/` | Scoring, skill score, bootstrap, verdicts, schema validation | Only code allowed to produce a score (HF-3). Written to be lifted into the shared monorepo |
| `contracts/` | JSON Schemas for results and prediction metadata | Changing a schema needs a test change in the same commit |
| `runs/` | One directory per run: `proba.npy`, `meta.json`, `system_log.json` | Append-only, hashed, never edited |
| `report/` | Builds the static results page from `results.json` | Build fails if a publish-standard section has no data (US-14) |

## 3. Architecture decision records

### ADR-001 Stack
Context: solo build on a Windows laptop with 20 logical CPUs and 16.8 GB RAM, must reproduce from a clean clone. Choice: Python (3.11 or 3.12, whichever AutoGluon 1.6.3 supports, confirmed in Phase 5), numpy, pandas, scikit-learn, LightGBM, pytest, hypothesis, jsonschema, pyarrow. `uv` with a lockfile. Baselines installed through an optional `bench` extra. Rejected: conda (hard to lock, how the spike picked up stray ray and mlflow), Poetry (extra tool, no gain), a monorepo-wide toolchain before 1A exists. Consequences: if the AutoGluon pin conflicts with Tab's scikit-learn range, the bench extra moves into its own environment. ADR-003 already makes that cheap.

### ADR-002 No LLM anywhere
Context: the playbook's LLM-layer items (model abstraction, prompt files, token caps, structured model outputs) assume a model call. Choice: Tab makes none, and no LLM judge is used in evaluation. Rejected: an LLM-written report or an LLM judge for leak explanations, because every claim here must be checkable by deterministic code. Consequences: the model-provider ADR and prompt versioning do not apply. Cost analysis is CPU time and memory only.

### ADR-003 One subprocess per run, files as the contract
Context: we need honest wall-clock including startup and refit, peak memory across worker processes, clean kills at timeout, and room for baselines with conflicting dependencies. Choice: the runner spawns `python -m bench.systems.<name>` for each (system, task, fold), watches the whole process tree with psutil, and assembles the run record itself. The system sees training data and unlabeled test features, writes `proba.npy` and `meta.json`, and nothing else counts. Rejected: calling systems in-process (cannot measure memory or enforce timeouts cleanly, and a crash takes the harness down), a job queue or container per run (more machinery than the question needs). Consequences: slightly slower startup per run, counted in actual seconds for every system equally. A system cannot misreport its own time or score.

### ADR-004 Plain files, no database
Context: a run produces a few arrays and a small JSON. Choice: directories under `runs/`, content-hashed; `results.json` assembled from them by one script. Rejected: SQLite or DuckDB (fine tools, but nothing here needs queries beyond pandas), MLflow (the spike showed what an unrelated tracker drags into the environment). Consequences: resumable overnight batches fall out for free, since a run directory either has valid `meta.json` or gets redone.

### ADR-005 Static report, hosted by the portfolio site
Context: the master plan has one portfolio site that renders every project's `results.json`. Choice: `report/` outputs a static bundle and the validated `results.json`; GitHub Pages is the fallback host. Rejected: a running service or notebook server. Consequences: the repro badge points at the CI job, and the page can be diffed between releases.

### ADR-006 evalkit lives here first, as a standalone package
Context: the master plan says 1A creates evalkit v0 and the results envelope, but 3A is being built first. Choice: `evalkit/` is a self-contained package inside this repo with its own tests and no imports from `tab/` or `bench/`; the envelope fields in `results.schema.json` (schema_version, project, git_sha, hardware, versions, runs, analyses) are project-neutral. Whichever project ships first owns the canonical copy; the other adopts it. Rejected: blocking 3A on 1A, or building a monorepo skeleton first. Consequences: the master plan needs one line changed (see section 7).

### ADR-007 Pure functions at the seams
Context: the controller, leak detector, splitter and scorer carry most of the claims. Choice: each is a function over arrays and plain config, no hidden state, no I/O, so tests feed curves and tables directly. Rejected: classes that own fitting loops and also decide when to stop, which cannot be tested without running a search. Consequences: the eval spec's 8 controller curves and 30 leak cases run in milliseconds.

## 4. Contracts

### 4.1 Tab API (Python)
```python
class Tab:
    def __init__(self, *, ceiling_s: float | None = None, max_trials: int | None = None,
                 seed: int = 0, n_jobs: int = 1, controller: ControllerConfig | None = None,
                 time_col: str | None = None, stacking: bool = True, calibrate: bool = True)
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Tab"
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray   # (n, k) float64, rows sum to 1, class order = classes_
    classes_: np.ndarray
    report_: FitReport   # actual_wall_s, stop_reason, n_trials, leaderboard, ece_before, ece_after, leak_flags
```
Exactly one of `ceiling_s` (time mode, benchmark) or `max_trials` (trial mode, tests and NFR-2 determinism) must be set, otherwise ValueError. Inputs outside 50,000 rows, 200 columns or 10 classes raise ValueError (NFR-5). A missing or NaN target raises.

### 4.2 System runner contract
`python -m bench.systems.<name> --split-dir DIR --budget-s B --seed S --cpus N --out RUNDIR`
- `DIR/train.parquet` has features and the target column `__y`. `DIR/test_X.parquet` has features only. Test labels live elsewhere and are never passed to a system.
- The system writes `RUNDIR/proba.npy` (float64, n_test x k, columns in `meta.classes` order), `RUNDIR/meta.json` (validated by `predictions_meta.schema.json`), and optionally `RUNDIR/system_log.json` with `stop_reason` and `n_trials`.
- The runner records `actual_wall_s` (from spawn to exit, refit included), `peak_rss_gb_tree`, `n_cpus`, exit status. A timeout is the ceiling times 1.5, then a kill, status `timeout`.
- evalkit reads `proba.npy`, `meta.json` and the withheld labels, and writes the `scores` block with `scored_by: "evalkit@<version>"`.

### 4.3 results.json
`contracts/results.schema.json`, enforced by 14 passing tests in `tests/contract/test_results_schema.py`. The rules that carry PRD claims: `actual_wall_s` required and positive (HF-5); `scores` must name evalkit (HF-3); a failed run needs an `error` and is kept, never dropped; `headline.metric` can only be `skill_score_log_loss` (HF-4); unknown fields are rejected so typos fail loudly. Verdict-versus-interval consistency is not expressible in JSON Schema and gets its own test in Phase 6 (`tests/contract/test_headline.py`).

### 4.4 Scoring definitions (evalkit)
Log loss: probabilities clipped at 1e-15, labels fixed from the full dataset's class list. Skill: 1 minus (model log loss divided by the log loss of the training-fold class-prior predictor). AUC: binary datasets only, one-vs-rest is not computed. ECE: 15 equal-width bins on the max-class probability. Intervals: percentile bootstrap, 10,000 resamples, fixed seed. Verdict: win or loss if the 95% interval of the mean paired fold difference excludes zero, else tie.

## 5. Walking skeleton (Phase 5 builds this first)
The thinnest slice that touches every layer. Systems are the class-prior stub and default logistic regression. Data is credit-g fold 0 locally, and a 200-row bundled fixture in CI so CI needs no network.
1. `bench/datasets.py` loads the fold and writes `train.parquet`, `test_X.parquet`, and withheld labels.
2. The runner spawns each system as a subprocess, measures time and memory.
3. The system writes `proba.npy` and `meta.json`.
4. evalkit scores, writes the run record, assembles `results.json`.
5. `results.json` validates against the schema.
6. `report/` builds one page with one table row and a headline section that says n = 1 and is not yet a claim.
7. CI runs steps 1 to 6 from a clean checkout in under 5 minutes and uploads the page.
Exit test: `make skeleton` is green locally and in CI, and breaking any one contract (delete `actual_wall_s`, edit `scored_by`) turns it red.

## 6. Threat model (30 minutes, proportionate)
Who can send input: the author, plus data downloaded from OpenML and pip packages. No network service, no users, no secrets, no LLM.
| Risk | What could happen | Control |
|---|---|---|
| Model files loaded from disk | Pickle deserialization runs arbitrary code | Never unpickle across runs. The only artifact read back is `proba.npy` loaded with `allow_pickle=False` |
| Subprocess launch | Argument injection through dataset or system names | Argument lists only, never `shell=True`; system names validated against the schema pattern |
| Path handling | A crafted name writes outside `runs/` | Run IDs generated from validated fields; resolve and check the path stays under the runs root |
| Dependency supply chain | A compromised or drifted package changes results | Lockfile with hashes, versions recorded in `results.json`, CI installs from the lock only |
| Downloaded data drift | OpenML changes a dataset silently | Cache with sha256 per task, and `dataset_list_sha256` in results |
| Result tampering | A number edited by hand | Every prediction file hashed in the run record; `make verify` recomputes scores from `runs/` and compares |
| Resource exhaustion | A runaway system eats memory or time | Ceiling x 1.5 kill, memory watch on the tree, status `oom` or `timeout` recorded |
| Leakage into scoring | Labels reach a system | Labels never passed to the system process; contract test asserts the CLI args and split directory hold no label file |

## 7. Master plan updates this implies
- 3A advanced line: add the pre-registered controller study, the development pool, floor baselines, skill-score headline, electricity as the time-aware case, the four injected leak types.
- Shared spine: "evalkit v0 and the results.json envelope are created by whichever of 1A or 3A ships first" (currently says 1A). 3A is first, so its `evalkit/` and envelope become canonical, and 1A adopts them.
- Replace "equal wall-clock" with "same ceiling, compared on actual seconds".
- Note the 3A benchmark as a stratified draw of 20 from the 54 eligible CC18 datasets.
