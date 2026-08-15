"""How many null replicates does a threshold need before it holds?

During development I calibrated on 60 null replicates and KS came out with a
16.7% realised false-alarm rate against a 5% target -- the threshold was fitted
to the top three order statistics of a noisy sample and did not transfer. Rather
than quietly raise the replicate count, this sweeps it and measures the cost.

Runs on the saved null scores, so it is free and needs no re-simulation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from driftharm.detectors import DETECTOR_NAMES

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
ALPHA = 0.05
N_VAL = 100
N_DRAWS = 400


def sweep(null: pd.DataFrame, sizes=(20, 40, 60, 100, 150)) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = len(null)
    rows = []
    for n_cal in sizes:
        if n_cal + N_VAL > n:
            continue
        far = {d: [] for d in DETECTOR_NAMES}
        for _ in range(N_DRAWS):
            idx = rng.permutation(n)
            cal, val = null.iloc[idx[:n_cal]], null.iloc[idx[n_cal:n_cal + N_VAL]]
            for d in DETECTOR_NAMES:
                tau = np.quantile(cal[d], 1 - ALPHA)
                far[d].append(float((val[d] > tau).mean()))
        for d in DETECTOR_NAMES:
            v = np.array(far[d])
            rows.append({
                "n_calibration_reps": n_cal,
                "detector": d,
                "mean_far": v.mean(),
                "p90_far": np.quantile(v, 0.9),
                "target_alpha": ALPHA,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for which in ("real", "synthetic"):
        f = REPORTS / f"{which}_null_scores.csv"
        if not f.exists():
            continue
        out = sweep(pd.read_csv(f))
        out.to_csv(REPORTS / f"{which}_calibration_size.csv", index=False)
        print(f"\n=== {which}: realised false-alarm rate vs calibration size "
              f"(target {ALPHA}) ===")
        print(out.pivot(index="n_calibration_reps", columns="detector",
                        values="mean_far").round(4).to_string())
