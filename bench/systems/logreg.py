"""Floor baseline: default LogisticRegression with Tab's in-fold one-hot preprocessing."""

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

from bench.systems._common import run_system
from tab.preprocess import Preprocessor


def fit_predict(train, y, test, ctx):
    classes = sorted(y.unique())
    pre = Preprocessor(mode="onehot").fit(train)
    model = LogisticRegression(max_iter=1000, random_state=ctx.seed)
    model.fit(pre.transform(train), y.to_numpy())
    p = model.predict_proba(pre.transform(test))
    order = [list(model.classes_).index(c) for c in classes]
    return np.asarray(p[:, order], dtype=np.float64), classes, {"stop_reason": None, "n_trials": 1}


if __name__ == "__main__":
    run_system("logreg", sklearn.__version__, fit_predict)
