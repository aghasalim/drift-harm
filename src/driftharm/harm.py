"""Measuring harm, which is the only thing this benchmark actually cares about.

Harm is the drop in out-of-sample ROC AUC between the reference window and the
current window, scored by a model that was fitted on strictly earlier rows and
has seen neither window.

The number on its own is meaningless. Two random windows from an unchanged pool
produce a non-zero AUC gap simply because AUC is estimated from a finite number
of positives. So the harm *label* is defined against that measured null, not
against a round number someone liked.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC, or NaN when a window happens to contain a single class."""
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def auc_drop(bundle, ref_X, ref_y, cur_X, cur_y) -> dict:
    a_ref = safe_auc(ref_y, bundle.predict(ref_X))
    a_cur = safe_auc(cur_y, bundle.predict(cur_X))
    return {"auc_ref": a_ref, "auc_cur": a_cur, "auc_drop": a_ref - a_cur}


def segment_auc_drop(bundle, ref_X, ref_y, cur_X, cur_y, rows) -> float:
    """AUC drop restricted to a row subset, for the dilution archetype."""
    if rows is None or len(rows) == 0:
        return float("nan")
    a_ref = safe_auc(ref_y[rows], bundle.predict(ref_X[rows]))
    a_cur = safe_auc(cur_y[rows], bundle.predict(cur_X[rows]))
    return a_ref - a_cur
