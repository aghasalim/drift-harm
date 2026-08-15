"""Run the archetype suite: draw windows, apply archetypes, score everything."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import draw_pair
from .detectors import DETECTORS
from .harm import auc_drop, segment_auc_drop
from .scenarios import ARCHETYPES


def _context(bundle, ref_X) -> dict:
    return {
        "importance": bundle.importance,
        "zero_idx": bundle.zero_idx,
        "ref_median": np.nanmedian(ref_X, axis=0),
        "scores": None,  # filled per-window
    }


def run_suite(bundle, n_reps: int = 20, window: int = 20_000, seed: int = 0,
              archetypes=None, detectors=None) -> pd.DataFrame:
    """One record per (archetype, replicate) with every detector's raw score.

    Alarm labels are *not* assigned here -- thresholds come from the null run and
    are applied afterwards, so a threshold change never requires re-running the
    expensive part.
    """
    archetypes = archetypes or ARCHETYPES
    detectors = detectors or DETECTORS
    rng = np.random.default_rng(seed)
    rows = []

    for rep in range(n_reps):
        i_ref, i_cur = draw_pair(rng, len(bundle.pool_X), window)
        ref_X, ref_y = bundle.pool_X[i_ref], bundle.pool_y[i_ref]
        base_X, base_y = bundle.pool_X[i_cur], bundle.pool_y[i_cur]

        ctx = _context(bundle, ref_X)
        ctx["scores"] = bundle.predict(base_X)

        for name, arch in archetypes.items():
            w = arch.fn(rng, base_X, base_y, ctx)
            rec = {
                "archetype": name,
                "rep": rep,
                "expected_harm": arch.expected_harm,
                "expected_alarm": arch.expected_alarm,
            }
            rec.update(auc_drop(bundle, ref_X, ref_y, w.X_model, w.y))
            seg = w.meta.get("seg_rows")
            rec["segment_auc_drop"] = segment_auc_drop(
                bundle, ref_X, ref_y, w.X_model, w.y, seg
            )
            rec["segment_positive_share"] = w.meta.get("seg_positive_share", np.nan)
            for dname, det in detectors.items():
                rec[dname] = det(ref_X, w.X_monitor)
            rows.append(rec)
    return pd.DataFrame(rows)


def label_alarms(trials: pd.DataFrame, taus: dict, harm_tau: float,
                 seg_harm_tau: float | None = None) -> pd.DataFrame:
    """Apply the calibrated thresholds. The benchmark's harm label is the
    aggregate one -- a single consistent rule across every archetype. The
    segment label is recorded alongside it but not used for ranking, because
    only the dilution archetypes define a segment; see the README on why the
    aggregate rule scores dilution as harmless."""
    out = trials.copy()
    out["harm"] = (out["auc_drop"] > harm_tau).astype(int)
    if seg_harm_tau is not None:
        out["segment_harm"] = (out["segment_auc_drop"] > seg_harm_tau).astype(
            "Int64"
        )
    for d, t in taus.items():
        out[f"alarm_{d}"] = (out[d] > t).astype(int)
    return out


def gradual_curve(bundle, n_reps: int = 10, window: int = 20_000, seed: int = 0,
                  n_batches: int = 8, strength: float = 1.0,
                  detectors=None) -> pd.DataFrame:
    """Same total shift, delivered abruptly vs. accrued over `n_batches`.

    The sudden arm receives the full corruption in batch 1 and holds it; the
    gradual arm receives b/n_batches of it at batch b. Both end at the same
    place, which is what makes detection delay comparable.
    """
    from .scenarios import _graded_shift

    detectors = detectors or DETECTORS
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(n_reps):
        i_ref, i_cur = draw_pair(rng, len(bundle.pool_X), window)
        ref_X, ref_y = bundle.pool_X[i_ref], bundle.pool_y[i_ref]
        base_X, base_y = bundle.pool_X[i_cur], bundle.pool_y[i_cur]
        ctx = _context(bundle, ref_X)
        ctx["scores"] = bundle.predict(base_X)

        for b in range(1, n_batches + 1):
            for mode, frac in (("sudden", 1.0), ("gradual", b / n_batches)):
                w = _graded_shift(rng, base_X, base_y, ctx, frac=frac,
                                  strength=strength, n_cols=10)
                rec = {"mode": mode, "batch": b, "rep": rep, "frac": frac}
                rec.update(auc_drop(bundle, ref_X, ref_y, w.X_model, w.y))
                for dname, det in detectors.items():
                    rec[dname] = det(ref_X, w.X_monitor)
                rows.append(rec)
    return pd.DataFrame(rows)
