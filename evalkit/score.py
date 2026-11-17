"""Turn one run's saved probabilities and withheld labels into the `scores` block."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from evalkit import SCORED_BY, metrics


def encode(labels: NDArray[Any], all_classes: Sequence[str]) -> NDArray[np.int64]:
    index = {c: i for i, c in enumerate(all_classes)}
    try:
        return np.array([index[str(v)] for v in labels], dtype=np.int64)
    except KeyError as e:  # pragma: no cover - defensive
        raise ValueError(f"label {e} not in class list") from e


def score_predictions(
    proba: NDArray[np.floating],
    proba_classes: Sequence[str],
    y_test: NDArray[Any],
    y_train: NDArray[Any],
    all_classes: Sequence[str],
) -> dict[str, Any]:
    metrics.check_proba(proba)
    p = metrics.align_proba(proba, [str(c) for c in proba_classes], list(all_classes))
    yt = encode(y_test, all_classes)
    if p.shape[0] != yt.size:
        raise ValueError(f"{p.shape[0]} predictions for {yt.size} test rows")
    ytr = encode(y_train, all_classes)
    k = len(all_classes)
    ll = metrics.log_loss(yt, p)
    prior_ll = metrics.log_loss(yt, metrics.prior_proba(ytr, k, yt.size))
    auc = metrics.roc_auc_binary(yt, p[:, 1]) if k == 2 else None
    return {
        "scored_by": SCORED_BY,
        "log_loss": ll,
        "prior_log_loss": prior_ll,
        "skill": metrics.skill(ll, prior_ll),
        "auc": auc,
        "ece": metrics.ece(yt, p),
        "n_test": int(yt.size),
    }
