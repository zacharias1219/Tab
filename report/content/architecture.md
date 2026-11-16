### Problem
Tabular AutoML systems are ranked by leaderboard position. Readers rarely learn when a system loses, whether more search time actually helps, or whether the data was leaky to begin with. Tab is a small AutoML system for tabular classification whose product is trustworthy measurement: every number on this page traces to a check and to a file a stranger can regenerate.

### Architecture
- `tab/` is a plain Python library that fits and predicts: in-fold preprocessing, a boring zoo (logistic, random forest, histogram gradient boosting, small MLP), random search over 3 inner folds, a plateau budget controller, OOF stacking, temperature calibration, permutation importance, a leak detector and forward-chaining time splits.
- `bench/` runs every system, Tab included, as **one subprocess per run**, measured from outside: wall-clock from spawn to exit, peak memory summed over the whole process tree.
- Systems see training data and unlabeled test features, and write `proba.npy` plus `meta.json`. Nothing else counts. A system cannot report its own score or its own time.
- `evalkit/` is the only code allowed to produce a score. It reads the saved probabilities and the withheld labels.
- `results.json` is schema-validated; this page is built from it and refuses to build if a section has no data.
