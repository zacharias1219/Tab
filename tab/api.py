"""Public API (ARCHITECTURE 4.1)."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tab import config as C
from tab._metrics import ece, log_loss_rows
from tab.calibration import apply_temperature, cross_fit_calibration
from tab.config import ControllerConfig
from tab.controller import should_stop
from tab.leak import DEFAULT_THRESHOLD, detect_leaks
from tab.nested import inner_folds
from tab.preprocess import Preprocessor, dedupe_columns, normalize_frame
from tab.stacking import Stacker
from tab.time_split import forward_chaining_splits
from tab.zoo import (
    DEFAULT_ORDER,
    FAMILIES,
    build_model,
    default_config,
    fit_quiet,
    full_proba,
    sample_config,
)

INNER_FOLDS = 3
STACK_MAX_BASE = 4
# Rough relative trial cost of an unseen family vs the cheapest seen one (time mode only).
FAMILY_PRIOR = {"logistic": 1.0, "hist_gb": 2.0, "random_forest": 3.0, "mlp": 3.0}
TIME_SAFETY = 0.88
LEAK_SCAN_ROWS = 5000  # fraction of the ceiling the planner aims to finish inside


def validate_inputs(X: pd.DataFrame, y: pd.Series) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
    if len(X) > C.MAX_ROWS:
        raise ValueError(f"Tab supports at most 50,000 rows; got {len(X):,} (NFR-5)")
    if X.shape[1] > C.MAX_COLS:
        raise ValueError(f"Tab supports at most 200 columns; got {X.shape[1]} (NFR-5)")
    if X.shape[1] == 0:
        raise ValueError("X has no columns")
    ys = pd.Series(y)
    if ys.isna().any():
        raise ValueError("target has missing values (NaN); drop or label those rows first")
    k = ys.nunique()
    if k < 2 or k > C.MAX_CLASSES:
        raise ValueError(f"Tab supports between 2 and 10 classes; got {k} classes (NFR-5)")
    if len(X) < 2 * INNER_FOLDS:
        raise ValueError(f"need at least {2 * INNER_FOLDS} rows")


def _sorted_classes(y: pd.Series) -> np.ndarray:
    vals = pd.Series(y).to_numpy()
    try:
        return np.unique(vals)
    except TypeError:
        return np.array(sorted(pd.unique(vals), key=str), dtype=object)


@dataclass
class Trial:
    idx: int
    family: str
    params: dict[str, Any]
    status: str = "ok"
    error: str | None = None
    mean_loss: float = math.inf
    fold_losses: list[float] = field(default_factory=list)
    ci: tuple[float, float] = (math.inf, math.inf)
    wall_s: float = 0.0
    oof: np.ndarray | None = None
    fold_models: list[Any] | None = None

    @property
    def name(self) -> str:
        return f"t{self.idx:03d}_{self.family}"

    @property
    def cfg(self) -> dict[str, Any]:
        return {"family": self.family, "params": self.params}


@dataclass
class FitReport:
    actual_wall_s: float
    refit_s: float
    stop_reason: str
    n_trials: int
    n_failed_trials: int
    leaderboard: list[dict[str, Any]]
    ece_before: float | None
    ece_after: float | None
    calibrated: bool
    temperature: float | None
    chosen: str
    stack_considered: bool
    best_single_loss: float | None
    stack_loss: float | None
    leak_flags: list[dict[str, Any]]
    refit_mode: str
    provisional_params: bool
    ceiling_s: float | None
    max_trials: int | None
    timings: dict[str, float]
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Tab:
    def __init__(
        self,
        *,
        ceiling_s: float | None = None,
        max_trials: int | None = None,
        seed: int = 0,
        n_jobs: int = 1,
        controller: ControllerConfig | None = None,
        time_col: str | None = None,
        stacking: bool = True,
        calibrate: bool = True,
        leak_threshold: float | None = None,
        _observer: Any = None,
    ):
        if (ceiling_s is None) == (max_trials is None):
            raise ValueError("set exactly one of ceiling_s (time mode) or max_trials (trial mode)")
        if ceiling_s is not None and not ceiling_s > 0:
            raise ValueError("ceiling_s must be positive")
        if max_trials is not None and max_trials < 1:
            raise ValueError("max_trials must be >= 1")
        self.ceiling_s = ceiling_s
        self.max_trials = max_trials
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.controller = controller or ControllerConfig()
        self.time_col = time_col
        self.stacking = stacking
        self.calibrate = calibrate
        self.leak_threshold = leak_threshold
        self._observer = _observer
        self._fitted = False

    # ---------------------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Tab:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        validate_inputs(X, y)
        y = pd.Series(y)
        X = normalize_frame(X)
        ts = None
        if self.time_col is not None:
            tc = dedupe_columns([self.time_col])[0]
            if tc not in X.columns:
                raise ValueError(f"time column {self.time_col!r} not in X")
            ts = X[tc]
            X = X.drop(columns=[tc])
        self.classes_ = _sorted_classes(y)
        cls_index = {c: i for i, c in enumerate(self.classes_.tolist())}
        yc = np.array([cls_index[v] for v in y.to_numpy().tolist()], dtype=np.int64)
        k = self.classes_.size
        n = len(X)
        self.columns_ = list(X.columns)
        index = X.index.to_numpy()
        obs = self._observer
        time_mode = self.ceiling_s is not None
        ceiling = float(self.ceiling_s or 0.0)
        deadline = t0 + ceiling

        # leak scan (cheap, flags only)
        ts_l = time.perf_counter()
        try:
            leak = detect_leaks(
                X,
                y.to_numpy(),
                threshold=self.leak_threshold or DEFAULT_THRESHOLD,
                null_columns=0,
                seed=self.seed,
                max_rows=LEAK_SCAN_ROWS,
            )
            leak_flags = [f for f in leak.flags if f["flagged"]]
        except Exception as e:  # never let a diagnostic break fitting
            leak_flags = [
                {
                    "column": None,
                    "type": "detector_error",
                    "score": 0.0,
                    "flagged": False,
                    "error": str(e),
                }
            ]
        timings["leak_scan_s"] = time.perf_counter() - ts_l

        # inner folds
        if ts is not None:
            folds = forward_chaining_splits(ts, n_splits=INNER_FOLDS)
        else:
            folds = inner_folds(yc, INNER_FOLDS, self.seed)
        fold_id = np.full(n, -1)
        for f, (_, va) in enumerate(folds):
            fold_id[va] = f
        mask = fold_id >= 0

        cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

        def fold_data(mode: str, f: int):
            key = (mode, f)
            if key not in cache:
                tr, va = folds[f]
                pre = Preprocessor(mode=mode, observer=obs).fit(X.iloc[tr])
                cache[key] = (pre.transform(X.iloc[tr]), pre.transform(X.iloc[va]))
            return cache[key]

        def run_trial(i: int, cfg: dict[str, Any]) -> Trial:
            tr0 = time.perf_counter()
            t = Trial(i, cfg["family"], cfg["params"])
            oof = np.full((n, k), np.nan)
            models = []
            try:
                mode = FAMILIES[t.family].preproc_mode
                for f, (tr, va) in enumerate(folds):
                    Xtr, Xva = fold_data(mode, f)
                    m = fit_quiet(build_model(cfg, self.seed + i, self.n_jobs), Xtr, yc[tr])
                    if obs is not None:
                        obs.record("model", index[tr])
                        obs.record_oof(index[tr], index[va])
                    oof[va] = full_proba(m, Xva, k)
                    models.append(m)
                rows = log_loss_rows(yc[mask], oof[mask])
                t.mean_loss = float(rows.mean())
                half = (
                    1.96 * float(rows.std(ddof=1)) / math.sqrt(rows.size) if rows.size > 1 else 0.0
                )
                t.ci = (t.mean_loss - half, t.mean_loss + half)
                t.fold_losses = [float(log_loss_rows(yc[va], oof[va]).mean()) for _, va in folds]
                t.oof = oof
                t.fold_models = models
            except Exception as e:  # a failed trial is recorded, never fatal
                t.status = "failed"
                t.error = f"{type(e).__name__}: {e}"[:300]
                t.mean_loss = math.nan
            t.wall_s = time.perf_counter() - tr0
            return t

        rng = np.random.RandomState(self.seed)
        trials: list[Trial] = []
        unit: dict[str, float] = {}
        stop_reason = "max_trials"
        ts_s = time.perf_counter()

        def est_trial(cfg) -> float:
            fam = cfg["family"]
            cost = FAMILIES[fam].cost(cfg["params"])
            if fam in unit:
                return unit[fam] * cost
            if not unit:
                return 0.0
            base = min(u / FAMILY_PRIOR[f] for f, u in unit.items())
            return base * FAMILY_PRIOR[fam] * cost * 1.5

        def est_finalize() -> float:
            ok = [t for t in trials if t.status == "ok"]
            if not ok:
                return 0.0
            best_by_fam: dict[str, Trial] = {}
            for t in sorted(ok, key=lambda t: t.mean_loss):
                best_by_fam.setdefault(t.family, t)
            members = list(best_by_fam.values())[: (STACK_MAX_BASE if self.stacking else 1)]
            return sum(0.6 * t.wall_s for t in members) + 0.02 * (self.ceiling_s or 0) + 0.3

        i = 0
        while True:
            if self.max_trials is not None and i >= self.max_trials:
                stop_reason = "max_trials"
                break
            cfg = (
                default_config(DEFAULT_ORDER[i], n, X.shape[1])
                if i < len(DEFAULT_ORDER)
                else sample_config(rng, n, X.shape[1])
            )
            if time_mode:
                now = time.perf_counter()
                budget_end = t0 + TIME_SAFETY * ceiling
                fits = now + 1.3 * est_trial(cfg) + est_finalize() <= budget_end
                tries = 0
                while not fits and tries < 6:
                    cfg = sample_config(rng, n, X.shape[1])
                    fits = now + 1.3 * est_trial(cfg) + est_finalize() <= budget_end
                    tries += 1
                if not fits and trials:
                    stop_reason = "ceiling"
                    break
            t = run_trial(i, cfg)
            trials.append(t)
            if t.status == "ok":
                cost = FAMILIES[t.family].cost(t.params)
                u = t.wall_s / max(cost, 1e-6)
                unit[t.family] = max(unit.get(t.family, 0.0), u) if t.family in unit else u
            self._prune_fold_models(trials)
            i += 1
            r = should_stop([tt.mean_loss for tt in trials], self.controller)
            if r:
                stop_reason = r
                break
            if time_mode and time.perf_counter() >= deadline:
                stop_reason = "ceiling"
                break
        timings["search_s"] = time.perf_counter() - ts_s

        # ---- selection: best single, optional stack, optional calibration ------------------
        tsel = time.perf_counter()
        ok = sorted([t for t in trials if t.status == "ok"], key=lambda t: (t.mean_loss, t.idx))
        prior = np.bincount(yc, minlength=k).astype(float) + 1e-3
        self._prior = prior / prior.sum()
        self._plan: dict[str, Any] = {"kind": "prior"}
        best_single_loss = stack_loss = None
        stack_considered = False
        chosen_oof = None
        lb_extra = []
        if ok:
            best = ok[0]
            best_single_loss = best.mean_loss
            chosen_oof = best.oof
            self._plan = {"kind": "single", "members": [best]}
            best_by_fam: dict[str, Trial] = {}
            for t in ok:
                best_by_fam.setdefault(t.family, t)
            bases = list(best_by_fam.values())[:STACK_MAX_BASE]
            base_oof: list[np.ndarray] = [b.oof for b in bases if b.oof is not None]
            if self.stacking and len(bases) >= 2:
                stack_considered = True
                stack_oof = np.full((n, k), np.nan)
                for f in range(len(folds)):
                    trm = mask & (fold_id != f)
                    tem = fold_id == f
                    if obs is not None:
                        obs.record("stacker", index[trm])
                    st = Stacker(seed=self.seed).fit([o[trm] for o in base_oof], yc[trm], k)
                    stack_oof[tem] = st.predict_proba([o[tem] for o in base_oof])
                rows = log_loss_rows(yc[mask], stack_oof[mask])
                stack_loss = float(rows.mean())
                half = (
                    1.96 * float(rows.std(ddof=1)) / math.sqrt(rows.size) if rows.size > 1 else 0.0
                )
                lb_extra.append(
                    {
                        "model": "stack(" + ",".join(b.name for b in bases) + ")",
                        "family": "stack",
                        "params": {"members": [b.name for b in bases], "meta": "logistic C=1"},
                        "mean_log_loss": stack_loss,
                        "ci_low": stack_loss - half,
                        "ci_high": stack_loss + half,
                        "fold_log_loss": [
                            float(log_loss_rows(yc[fold_id == f], stack_oof[fold_id == f]).mean())
                            for f in range(len(folds))
                        ],
                        "wall_s": 0.0,
                    }
                )
                if stack_loss < best.mean_loss - 1e-4:
                    if obs is not None:
                        obs.record("stacker", index[mask])
                    self._stacker = Stacker(seed=self.seed).fit(
                        [o[mask] for o in base_oof], yc[mask], k
                    )
                    self._plan = {"kind": "stack", "members": bases}
                    chosen_oof = stack_oof

        ece_before = ece_after = None
        self._temperature = None
        calibrated = False
        if chosen_oof is not None:
            ece_before = ece(yc[mask], chosen_oof[mask])
            ece_after = ece_before
            if self.calibrate:
                res = cross_fit_calibration(
                    chosen_oof[mask], yc[mask], fold_id[mask], observer=obs, index=index[mask]
                )
                ece_after = res.ece_after if res.use else ece_before
                if res.use:
                    self._temperature = res.temperature
                    calibrated = True
        timings["select_s"] = time.perf_counter() - tsel

        # ---- refit on all training rows ----------------------------------------------------
        tr0 = time.perf_counter()
        refit_mode = "none"
        if self._plan["kind"] != "prior":
            members: list[Trial] = self._plan["members"]
            est = sum(0.6 * m.wall_s for m in members)
            short = time_mode and (time.perf_counter() + est > deadline - 0.02 * ceiling)
            if short and all(m.fold_models for m in members):
                refit_mode = "fold_bag"
                self._fitted_members: list[tuple[str, Any, Any]] = [
                    (m.family, None, m.fold_models) for m in members
                ]
                self._fold_preproc: dict[str, list[Preprocessor]] = {}
                for m in members:
                    mode = FAMILIES[m.family].preproc_mode
                    if mode not in self._fold_preproc:
                        self._fold_preproc[mode] = [
                            Preprocessor(mode=mode, observer=obs).fit(X.iloc[tr]) for tr, _ in folds
                        ]
            else:
                refit_mode = "full"
                pres: dict[str, Preprocessor] = {}
                fitted: list[tuple[str, Any, Any]] = []
                for m in members:
                    mode = FAMILIES[m.family].preproc_mode
                    if mode not in pres:
                        pres[mode] = Preprocessor(mode=mode, observer=obs).fit(X)
                    model = fit_quiet(
                        build_model(m.cfg, self.seed + m.idx, self.n_jobs),
                        pres[mode].transform(X),
                        yc,
                    )
                    if obs is not None:
                        obs.record("model", index)
                    fitted.append((m.family, model, None))
                self._pre = pres
                self._fitted_members = fitted
        refit_s = time.perf_counter() - tr0
        for t in trials:  # free memory
            t.oof = None
            t.fold_models = None

        lb = [
            {
                "model": t.name,
                "family": t.family,
                "params": t.params,
                "mean_log_loss": t.mean_loss,
                "ci_low": t.ci[0],
                "ci_high": t.ci[1],
                "fold_log_loss": t.fold_losses,
                "wall_s": t.wall_s,
            }
            for t in ok
        ] + lb_extra
        if not lb:
            rows = log_loss_rows(yc[mask], np.tile(self._prior, (int(mask.sum()), 1)))
            lb = [
                {
                    "model": "prior",
                    "family": "prior",
                    "params": {},
                    "mean_log_loss": float(rows.mean()),
                    "ci_low": float(rows.mean()),
                    "ci_high": float(rows.mean()),
                    "fold_log_loss": [float(rows.mean())],
                    "wall_s": 0.0,
                }
            ]
        lb.sort(key=lambda e: (e["mean_log_loss"], e["model"]))
        for rank, entry in enumerate(lb, 1):
            entry["rank"] = rank
        chosen = {"prior": "prior", "single": ok[0].name if ok else "prior", "stack": "stack"}[
            self._plan["kind"]
        ]
        self._k = k
        self._fitted = True
        self.report_ = FitReport(
            actual_wall_s=time.perf_counter() - t0,
            refit_s=max(refit_s, 1e-9),
            stop_reason=stop_reason,
            n_trials=len(trials),
            n_failed_trials=sum(t.status != "ok" for t in trials),
            leaderboard=lb,
            ece_before=ece_before,
            ece_after=ece_after,
            calibrated=calibrated,
            temperature=self._temperature,
            chosen=chosen,
            stack_considered=stack_considered,
            best_single_loss=best_single_loss,
            stack_loss=stack_loss,
            leak_flags=leak_flags,
            refit_mode=refit_mode,
            provisional_params=C.PROVISIONAL,
            ceiling_s=self.ceiling_s,
            max_trials=self.max_trials,
            timings=timings,
            history=[
                {
                    "trial": t.idx,
                    "family": t.family,
                    "loss": (None if not math.isfinite(t.mean_loss) else t.mean_loss),
                    "wall_s": t.wall_s,
                    "status": t.status,
                }
                for t in trials
            ],
        )
        return self

    @staticmethod
    def _prune_fold_models(trials: list[Trial]) -> None:
        """Keep fold models only for the current best trial of each family (memory bound)."""
        best: dict[str, Trial] = {}
        for t in trials:
            if t.status == "ok" and (
                t.family not in best or t.mean_loss < best[t.family].mean_loss
            ):
                best[t.family] = t
        keep = {id(t) for t in best.values()}
        for t in trials:
            if id(t) not in keep:
                t.fold_models = None
                t.oof = None

    # ---------------------------------------------------------------------------------------
    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        X = normalize_frame(X)
        if self.time_col is not None:
            tc = dedupe_columns([self.time_col])[0]
            if tc in X.columns:
                X = X.drop(columns=[tc])
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"columns missing at predict time: {missing[:5]}")
        return X[self.columns_]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit before predict_proba")
        X = self._align(X)
        n = len(X)
        if self._plan["kind"] == "prior":
            p = np.tile(self._prior, (n, 1))
        else:
            probas = []
            for family, model, fold_models in self._fitted_members:
                mode = FAMILIES[family].preproc_mode
                if fold_models is None:
                    probas.append(full_proba(model, self._pre[mode].transform(X), self._k))
                else:
                    ps = [
                        full_proba(m, pre.transform(X), self._k)
                        for m, pre in zip(fold_models, self._fold_preproc[mode], strict=True)
                    ]
                    probas.append(np.mean(ps, axis=0))
            p = self._stacker.predict_proba(probas) if self._plan["kind"] == "stack" else probas[0]
            if self._temperature is not None:
                p = apply_temperature(p, self._temperature)
        p = np.asarray(p, dtype=np.float64)
        p = np.clip(p, 0.0, None)
        return p / p.sum(axis=1, keepdims=True)

    # ---------------------------------------------------------------------------------------
    def leaderboard_json(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("not fitted")
        entries = []
        for e in self.report_.leaderboard:
            entries.append(
                {
                    k: e[k]
                    for k in (
                        "rank",
                        "model",
                        "family",
                        "params",
                        "mean_log_loss",
                        "ci_low",
                        "ci_high",
                        "fold_log_loss",
                        "wall_s",
                    )
                }
            )
        return {"schema_version": "1", "metric": "inner_cv_log_loss", "entries": entries}

    def write_leaderboard(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.leaderboard_json(), indent=2), encoding="utf-8")
        return path
