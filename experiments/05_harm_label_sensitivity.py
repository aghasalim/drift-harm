"""How much weight will the headline ranking carry? Two ways of asking.

1. Change the harm definition. The headline ranking labels a trial harmful when
   the *aggregate* AUC drop clears its null-calibrated threshold. That rule
   scores the dilution archetypes as harmless, because damage confined to 3% of
   rows barely moves an AUC computed over the whole window -- while the segment
   AUC drop on those same trials is large and clears its own separately
   calibrated threshold in every replicate. Re-scoring under
   `harm = aggregate OR segment` gives a second, equally defensible ranking.

2. Resample the trials. 240 trials is not many. A percentile bootstrap over
   whole trials puts an interval on each MCC, which is the only honest way to
   read a gap of 0.1 between two detectors.

Both run on the saved CSVs, so they are free and need no re-simulation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from driftharm.detectors import DETECTOR_NAMES
from driftharm.metrics import alignment, bootstrap_mcc_draws, rank_detectors

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
N_BOOT = 2000


def bootstrap_mcc(trials: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Flat trial bootstrap. Experiment 06 shows this scheme is the optimistic
    one -- it treats 240 trials as 240 independent facts when the alarm outcome
    is fixed by the archetype in most cells. Kept here because it is what the
    headline interval has always been; 06 reports the other two schemes."""
    draws = bootstrap_mcc_draws(trials, DETECTOR_NAMES, "iid_trial", n_boot, seed)
    harm = trials["harm"].to_numpy()
    rows = []
    for k, d in enumerate(DETECTOR_NAMES):
        v = draws[:, k]
        rows.append({
            "detector": d,
            "mcc": alignment(harm, trials[f"alarm_{d}"].to_numpy())["mcc"],
            "mcc_lo95": float(np.quantile(v, 0.025)),
            "mcc_hi95": float(np.quantile(v, 0.975)),
            "p_mcc_above_zero": float((v > 0).mean()),
        })
    return pd.DataFrame(rows).sort_values("mcc", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    for which in ("real", "synthetic"):
        f = REPORTS / f"{which}_trials.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)

        boot = bootstrap_mcc(t)
        boot.to_csv(REPORTS / f"{which}_ranking_bootstrap.csv", index=False)
        print(f"\n=== {which}: headline MCC with {N_BOOT}-draw bootstrap interval ===")
        print(boot.round(3).to_string(index=False))

        t2 = t.copy()
        t2["harm"] = ((t2["harm"] == 1) | (t2["segment_harm"].fillna(0) == 1)).astype(int)
        out = rank_detectors(t2, DETECTOR_NAMES)
        out.to_csv(REPORTS / f"{which}_ranking_segment_aware.csv", index=False)
        print(f"\n=== {which}: ranking under harm = aggregate OR 3%-segment "
              f"(base rate {t2['harm'].mean():.3f}) ===")
        print(out.round(3).to_string(index=False))
