"""The benchmark run: calibrate every detector against a measured null, then
score the archetype suite and rank detectors by harm alignment.

Order matters here and is not negotiable. The null is measured first, on window
pairs where nothing has happened, and the thresholds are frozen before a single
archetype is scored. Half the null replicates are held back so the realised
false-alarm rate is an out-of-sample number rather than a quantile agreeing with
itself.

    python experiments/02_run_benchmark.py real
    python experiments/02_run_benchmark.py synthetic
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from driftharm.calibration import harm_threshold, null_run, summarise_null, thresholds
from driftharm.detectors import DETECTOR_NAMES, DETECTORS
from driftharm.metrics import per_archetype, rank_detectors
from driftharm.suite import gradual_curve, label_alarms, run_suite

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
ALPHA = 0.05

CFG = {
    "real": dict(window=20_000, null_reps=300, suite_reps=20, grad_reps=6),
    "synthetic": dict(window=20_000, null_reps=300, suite_reps=20, grad_reps=6),
}


def load(which: str):
    if which == "real":
        return joblib.load(ROOT / "artifacts" / "real_bundle.pkl")
    from driftharm.data import build_synthetic_bundle

    return build_synthetic_bundle(n=250_000, d=60, n_informative=40, seed=0)


def main(which: str) -> None:
    cfg = CFG[which]
    REPORTS.mkdir(exist_ok=True)
    t0 = time.time()
    bundle = load(which)
    print(f"[{which}] bundle: {json.dumps(bundle.meta)[:400]}", flush=True)

    # ---- 1. the null, before anything else -------------------------------
    print(f"[{which}] null run, {cfg['null_reps']} reps ...", flush=True)
    null = null_run(bundle, n_reps=cfg["null_reps"], window=cfg["window"], seed=1)
    null.to_csv(REPORTS / f"{which}_null_scores.csv", index=False)

    half = len(null) // 2
    cal, val = null.iloc[:half], null.iloc[half:]
    taus = thresholds(cal, alpha=ALPHA)
    harm_tau = harm_threshold(cal, alpha=ALPHA)
    seg_harm_tau = harm_threshold(cal, alpha=ALPHA, col="segment_auc_drop")
    null_summary = summarise_null(cal, alpha=ALPHA, eval_df=val)
    null_summary.to_csv(REPORTS / f"{which}_null_summary.csv", index=False)
    print(null_summary.to_string(index=False), flush=True)
    print(f"[{which}] harm threshold (aggregate AUC drop) = {harm_tau:.5f}", flush=True)
    print(f"[{which}] harm threshold (3% segment AUC drop) = {seg_harm_tau:.5f}", flush=True)
    print(f"[{which}] null elapsed {time.time() - t0:.0f}s", flush=True)

    # ---- 2. the archetype suite ------------------------------------------
    print(f"[{which}] suite, {cfg['suite_reps']} reps x archetypes ...", flush=True)
    trials = run_suite(bundle, n_reps=cfg["suite_reps"], window=cfg["window"], seed=2)
    trials = label_alarms(trials, taus, harm_tau, seg_harm_tau)
    trials.to_csv(REPORTS / f"{which}_trials.csv", index=False)

    # The true-null archetype is part of the suite so that false alarms are
    # scored against measured quiet cases, not only against archetypes that
    # were designed to be quiet.
    ranking = rank_detectors(trials, DETECTOR_NAMES)
    ranking.to_csv(REPORTS / f"{which}_ranking.csv", index=False)
    print("\n=== harm-alignment ranking ===", flush=True)
    print(ranking.to_string(index=False), flush=True)

    by_arch = per_archetype(trials, DETECTOR_NAMES)
    by_arch.to_csv(REPORTS / f"{which}_by_archetype.csv", index=False)
    print("\n=== alarm rate per archetype ===", flush=True)
    print(by_arch.to_string(index=False), flush=True)

    # ---- 3. gradual vs sudden --------------------------------------------
    print(f"\n[{which}] gradual vs sudden ...", flush=True)
    grad = gradual_curve(bundle, n_reps=cfg["grad_reps"], window=cfg["window"], seed=3)
    for d, t in taus.items():
        grad[f"alarm_{d}"] = (grad[d] > t).astype(int)
    grad["harm"] = (grad["auc_drop"] > harm_tau).astype(int)
    grad.to_csv(REPORTS / f"{which}_gradual.csv", index=False)
    gsum = (
        grad.groupby(["mode", "batch"])
        .agg({"auc_drop": "mean", "harm": "mean",
              **{f"alarm_{d}": "mean" for d in DETECTOR_NAMES}})
        .reset_index()
    )
    gsum.to_csv(REPORTS / f"{which}_gradual_summary.csv", index=False)
    print(gsum.to_string(index=False), flush=True)

    meta = {
        "dataset": which,
        "alpha": ALPHA,
        "window": cfg["window"],
        "null_reps_total": cfg["null_reps"],
        "null_reps_calibration": half,
        "null_reps_validation": len(null) - half,
        "suite_reps": cfg["suite_reps"],
        "thresholds": taus,
        "harm_threshold_auc_drop": harm_tau,
        "harm_threshold_segment_auc_drop": seg_harm_tau,
        "null_segment_auc_drop_mean": float(null["segment_auc_drop"].mean()),
        "null_auc_drop_mean": float(null["auc_drop"].mean()),
        "null_auc_drop_sd": float(null["auc_drop"].std(ddof=1)),
        "bundle": {k: v for k, v in bundle.meta.items() if not isinstance(v, list)},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (REPORTS / f"{which}_run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[{which}] done in {meta['runtime_seconds']}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "real")
