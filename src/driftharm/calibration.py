"""Measure every detector's null before believing anything it says.

A drift score has no meaning until you know what it does when nothing has
happened. This module draws repeated pairs of disjoint, exchangeable windows
from the held-out pool, records each detector's score and the AUC gap, and turns
those empirical distributions into thresholds at a stated false-alarm rate.

Every threshold in the benchmark comes from here. None was chosen by eye.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .detectors import DETECTORS
from .harm import auc_drop, segment_auc_drop


def draw_pair(rng, n_pool: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Two disjoint random index sets of size `window` from the pool."""
    idx = rng.choice(n_pool, size=2 * window, replace=False)
    return idx[:window], idx[window:]


def null_run(bundle, n_reps: int = 100, window: int = 20_000, seed: int = 0,
             detectors=None) -> pd.DataFrame:
    """Score `n_reps` true-null window pairs with every detector."""
    detectors = detectors or DETECTORS
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(n_reps):
        i_ref, i_cur = draw_pair(rng, len(bundle.pool_X), window)
        ref_X, cur_X = bundle.pool_X[i_ref], bundle.pool_X[i_cur]
        ref_y, cur_y = bundle.pool_y[i_ref], bundle.pool_y[i_cur]
        rec = {"rep": rep}
        rec.update(auc_drop(bundle, ref_X, ref_y, cur_X, cur_y))
        # Same segment rule the dilution archetypes use, measured under the null.
        # Without this there is no way to say whether a segment-level AUC drop is
        # large -- and the aggregate threshold is far too tight for a 3% slice.
        k = max(1, int(len(cur_X) * 0.03))
        seg = np.argsort(-bundle.predict(cur_X))[:k]
        rec["segment_auc_drop"] = segment_auc_drop(
            bundle, ref_X, ref_y, cur_X, cur_y, seg
        )
        for name, det in detectors.items():
            rec[name] = det(ref_X, cur_X)
        rows.append(rec)
    return pd.DataFrame(rows)


def thresholds(null_df: pd.DataFrame, alpha: float = 0.05,
               detectors=None) -> dict[str, float]:
    """Per-detector alarm threshold at a target false-alarm rate `alpha`.

    The (1 - alpha) empirical quantile of the null score. Using the same alpha
    for every detector is the whole point: it removes "this one was tuned
    tighter" as an explanation for any difference in harm alignment.
    """
    detectors = detectors or DETECTORS
    return {n: float(np.quantile(null_df[n], 1 - alpha)) for n in detectors}


def harm_threshold(null_df: pd.DataFrame, alpha: float = 0.05,
                   col: str = "auc_drop") -> float:
    """AUC-drop threshold above which a window counts as harmed.

    Calibrated against the *same* null: this is how far apart two windows'
    AUCs get when nothing at all has changed. One-sided, because a window that
    scores better than reference is not harm. `col` selects the aggregate drop
    or the 3%-segment drop, which have very different noise floors.
    """
    return float(np.quantile(null_df[col].dropna(), 1 - alpha))


def summarise_null(null_df: pd.DataFrame, alpha: float = 0.05,
                   detectors=None, eval_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Null summary. If `eval_df` is given, thresholds come from `null_df` and
    the realised false-alarm rate is measured on `eval_df` -- an honest
    out-of-sample figure rather than the quantile agreeing with itself."""
    detectors = detectors or DETECTORS
    taus = thresholds(null_df, alpha, detectors)
    held = eval_df if eval_df is not None else null_df
    out = []
    for n in detectors:
        v = held[n].to_numpy()
        out.append({
            "detector": n,
            "null_mean": float(v.mean()),
            "null_sd": float(v.std(ddof=1)),
            "null_p50": float(np.quantile(v, 0.5)),
            "null_p95": float(np.quantile(v, 0.95)),
            "null_max": float(v.max()),
            "threshold": taus[n],
            "realised_far": float((v > taus[n]).mean()),
            "far_is_out_of_sample": eval_df is not None,
        })
    return pd.DataFrame(out)
