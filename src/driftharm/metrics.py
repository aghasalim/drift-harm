"""Harm alignment: scoring a detector against downstream damage, not against
whether a distribution moved.

A trial is one (scenario, replicate) pair. It has a binary measured-harm label
and, per detector, a binary alarm. Cross-tabulating those gives the four
quantities that matter:

    harm-recall      of the trials that hurt the model, how many were alerted
    harm-precision   of the alerts raised, how many corresponded to real damage
    harm-F1          their harmonic mean
    MCC              the same table as a correlation, which unlike F1 responds
                     to true negatives and so cannot be gamed by a detector that
                     simply alarms more often

MCC is the headline ranking because the suite deliberately contains quiet cases
that a trigger-happy detector should be punished for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def confusion(harm: np.ndarray, alarm: np.ndarray) -> dict:
    harm = np.asarray(harm).astype(bool)
    alarm = np.asarray(alarm).astype(bool)
    return {
        "tp": int((harm & alarm).sum()),
        "fp": int((~harm & alarm).sum()),
        "fn": int((harm & ~alarm).sum()),
        "tn": int((~harm & ~alarm).sum()),
    }


def alignment(harm: np.ndarray, alarm: np.ndarray) -> dict:
    c = confusion(harm, alarm)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (
        2 * prec * rec / (prec + rec)
        if tp + fp and tp + fn and (prec + rec) > 0
        else 0.0
    )
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {
        **c,
        "harm_precision": prec,
        "harm_recall": rec,
        "harm_f1": f1,
        "specificity": spec,
        "balanced_accuracy": np.nanmean([rec, spec]),
        "mcc": mcc,
    }


def rank_detectors(trials: pd.DataFrame, detector_names) -> pd.DataFrame:
    """One row per detector, sorted by MCC descending."""
    rows = []
    for d in detector_names:
        r = alignment(trials["harm"].to_numpy(), trials[f"alarm_{d}"].to_numpy())
        r["detector"] = d
        rows.append(r)
    cols = [
        "detector", "mcc", "harm_f1", "harm_precision", "harm_recall",
        "specificity", "balanced_accuracy", "tp", "fp", "fn", "tn",
    ]
    return pd.DataFrame(rows)[cols].sort_values("mcc", ascending=False).reset_index(drop=True)


def per_archetype(trials: pd.DataFrame, detector_names) -> pd.DataFrame:
    """Alarm rate per detector per archetype, next to the measured harm rate."""
    rows = []
    for arch, g in trials.groupby("archetype", sort=False):
        rec = {
            "archetype": arch,
            "n": len(g),
            "expected_harm": bool(g["expected_harm"].iloc[0]),
            "measured_harm_rate": float(g["harm"].mean()),
            "mean_auc_drop": float(g["auc_drop"].mean()),
        }
        for d in detector_names:
            rec[d] = float(g[f"alarm_{d}"].mean())
        rows.append(rec)
    return pd.DataFrame(rows)
