"""Why do the real and synthetic rankings disagree? Diagnosis, not tuning.

The README used to report the disagreement (Spearman -0.43) as an unexplained
weakness. This experiment asks what the disagreement is evidence *of*, and the
answer turns out to be: nothing beyond the instability the ranking already has
inside a single dataset.

Six measurements, all off the saved CSVs so this is free to re-run:

1. Three bootstrap schemes. The headline interval resamples trials as if all 240
   were independent. They are not: in most (detector, archetype) cells the alarm
   rate is exactly 0 or 1, so the 240 trials carry closer to 12 independent facts
   per detector. Resampling archetypes as clusters is the version of the question
   that matches the README's own limitation ("the 12 archetypes are my taxonomy,
   not an exhaustive one").

2. How often does the observed disagreement happen *within* one dataset? Draw two
   independent resamples of the same trials, rank both, correlate. That is the
   reference distribution the observed -0.43 has to be read against.

3. Leave one archetype out, twelve times, and see which detector wins.

4. Per-archetype harm-rate and alarm-rate differences between the two datasets --
   the same twelve archetype names do not produce the same twelve experiments.

5. A 2x2: real alarms scored against synthetic harm labels and vice versa, which
   separates "the detectors behaved differently" from "the labels were different".

6. For the archetype the two datasets disagree about most, the raw detector score
   against each dataset's own null -- an alarm rate of 0/20 versus 20/20 can come
   from an identical footprint sitting over two different noise floors.

Optionally, if artifacts/real_bundle.pkl is present, a per-column diagnostic of
what sets the max-KS null floor on real data. That is what makes measurement 6
come out the way it does.

    python experiments/06_ranking_stability.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from driftharm.detectors import DETECTOR_NAMES
from driftharm.metrics import (
    BOOTSTRAP_SCHEMES,
    alignment,
    bootstrap_mcc_draws,
    rank_stability,
    resample_index,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
N_BOOT = 2000
DATASETS = ("real", "synthetic")


def mcc_vector(harm: np.ndarray, alarms: np.ndarray) -> np.ndarray:
    return np.array([alignment(harm, alarms[:, k])["mcc"] for k in range(alarms.shape[1])])


def load(which: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS / f"{which}_trials.csv")


def alarm_matrix(t: pd.DataFrame) -> np.ndarray:
    return np.column_stack([t[f"alarm_{d}"].to_numpy() for d in DETECTOR_NAMES])


# -- 1. how stable is the ranking under each resampling scheme -----------------


def rank_stability_table(t: pd.DataFrame) -> pd.DataFrame:
    point = mcc_vector(t["harm"].to_numpy(), alarm_matrix(t))
    out = []
    for scheme in BOOTSTRAP_SCHEMES:
        draws = bootstrap_mcc_draws(t, DETECTOR_NAMES, scheme, N_BOOT, seed=0)
        tab = rank_stability(draws, DETECTOR_NAMES, point)
        tab.insert(0, "scheme", scheme)
        out.append(tab)
    return pd.concat(out, ignore_index=True)


# -- 2. does one dataset even agree with itself? -------------------------------


def self_agreement(t: pd.DataFrame, scheme: str, observed: float,
                   n_boot: int = N_BOOT, seed: int = 11) -> dict:
    """Spearman between the rankings from two independent resamples of the same
    trials. If the cross-dataset value sits inside this, the disagreement between
    datasets is not evidence of anything about the datasets."""
    rng = np.random.default_rng(seed)
    harm, alarms, groups = t["harm"].to_numpy(), alarm_matrix(t), t["archetype"].to_numpy()
    rho = np.empty(n_boot)
    for b in range(n_boot):
        i1, i2 = resample_index(rng, groups, scheme), resample_index(rng, groups, scheme)
        rho[b] = stats.spearmanr(
            mcc_vector(harm[i1], alarms[i1]), mcc_vector(harm[i2], alarms[i2])
        ).statistic
    ok = np.isfinite(rho)
    v = rho[ok]
    return {
        "scheme": scheme,
        "n_draws": n_boot,
        "n_degenerate": int((~ok).sum()),
        "self_spearman_mean": float(v.mean()),
        "self_spearman_p05": float(np.quantile(v, 0.05)),
        "self_spearman_p50": float(np.median(v)),
        "p_self_spearman_at_or_below_observed": float((v <= observed + 1e-9).mean()),
    }


# -- 3. how much does one archetype move the ranking? --------------------------


def leave_one_archetype_out(t: pd.DataFrame) -> pd.DataFrame:
    full = mcc_vector(t["harm"].to_numpy(), alarm_matrix(t))
    rows = []
    for a in dict.fromkeys(t["archetype"]):
        u = t[t["archetype"] != a]
        m = mcc_vector(u["harm"].to_numpy(), alarm_matrix(u))
        rows.append({
            "dropped_archetype": a,
            "winner": DETECTOR_NAMES[int(np.argmax(m))],
            "spearman_vs_full_suite": float(stats.spearmanr(full, m).statistic),
            **{d: float(m[k]) for k, d in enumerate(DETECTOR_NAMES)},
        })
    return pd.DataFrame(rows)


# -- 4. the same archetype names are not the same experiments ------------------


def archetype_disagreement() -> pd.DataFrame:
    a = pd.read_csv(REPORTS / "real_by_archetype.csv").set_index("archetype")
    b = pd.read_csv(REPORTS / "synthetic_by_archetype.csv").set_index("archetype")
    out = pd.DataFrame({
        "harm_rate_real": a["measured_harm_rate"],
        "harm_rate_synthetic": b["measured_harm_rate"],
        "harm_rate_diff": b["measured_harm_rate"] - a["measured_harm_rate"],
        "mean_abs_alarm_rate_diff": (b[list(DETECTOR_NAMES)] - a[list(DETECTOR_NAMES)])
        .abs().mean(axis=1),
    })
    for d in DETECTOR_NAMES:
        out[f"alarm_diff_{d}"] = b[d] - a[d]
    return out.reset_index().sort_values(
        "mean_abs_alarm_rate_diff", ascending=False, ignore_index=True
    )


# -- 5. labels or alarms? ------------------------------------------------------


def swap_decomposition(real: pd.DataFrame, syn: pd.DataFrame,
                       n_perm: int = 400, seed: int = 0) -> pd.DataFrame:
    """Score one dataset's alarms against the other's harm labels.

    Replicates inside an archetype are exchangeable, so a trial's alarm can be
    paired with any replicate's harm label from the same archetype in the other
    dataset. Averaged over random pairings so the answer is not an artifact of
    which replicate met which.
    """
    rng = np.random.default_rng(seed)
    arch = list(dict.fromkeys(real["archetype"]))
    assert list(dict.fromkeys(syn["archetype"])) == arch

    def hybrid(alarm_df, harm_df):
        ga, gh = alarm_df["archetype"].to_numpy(), harm_df["archetype"].to_numpy()
        A, h = alarm_matrix(alarm_df), harm_df["harm"].to_numpy()
        acc = np.zeros((n_perm, len(DETECTOR_NAMES)))
        for i in range(n_perm):
            h2 = np.empty(len(A), dtype=int)
            for a in arch:
                ia, ib = np.flatnonzero(ga == a), np.flatnonzero(gh == a)
                h2[ia] = h[rng.permutation(ib)]
            acc[i] = mcc_vector(h2, A)
        return acc.mean(axis=0)

    cells = {
        ("real", "real"): mcc_vector(real["harm"].to_numpy(), alarm_matrix(real)),
        ("real", "synthetic"): hybrid(real, syn),
        ("synthetic", "real"): hybrid(syn, real),
        ("synthetic", "synthetic"): mcc_vector(syn["harm"].to_numpy(), alarm_matrix(syn)),
    }
    base_r, base_s = cells[("real", "real")], cells[("synthetic", "synthetic")]
    rows = []
    for (alarms_from, labels_from), v in cells.items():
        rows.append({
            "alarms_from": alarms_from,
            "harm_labels_from": labels_from,
            "winner": DETECTOR_NAMES[int(np.argmax(v))],
            "spearman_vs_real_ranking": float(stats.spearmanr(base_r, v).statistic),
            "spearman_vs_synthetic_ranking": float(stats.spearmanr(base_s, v).statistic),
            **{d: float(v[k]) for k, d in enumerate(DETECTOR_NAMES)},
        })
    return pd.DataFrame(rows)


# -- 6. the archetype the two datasets disagree about most ---------------------


def masked_null_footprint() -> pd.DataFrame:
    """`imputation_masked_null` is the single largest alarm-behaviour difference
    between the datasets. Put its raw scores next to each dataset's own null, so
    the comparison is between the footprint the archetype leaves and the floor it
    has to clear -- not between two alarm rates."""
    import json

    rows = []
    for w in DATASETS:
        t = load(w)
        null = pd.read_csv(REPORTS / f"{w}_null_scores.csv")
        tau = json.loads((REPORTS / f"{w}_run_meta.json").read_text())["thresholds"]
        sub = t[t["archetype"] == "imputation_masked_null"]
        for d in DETECTOR_NAMES:
            rows.append({
                "dataset": w,
                "detector": d,
                "masked_null_score_mean": float(sub[d].mean()),
                "null_score_mean": float(null[d].mean()),
                "threshold": float(tau[d]),
                "score_over_threshold": float(sub[d].mean() / tau[d]),
                "alarm_rate": float(sub[f"alarm_{d}"].mean()),
            })
    return pd.DataFrame(rows)


# -- optional: what sets the max-KS null floor on real data --------------------


def real_null_floor_by_column() -> pd.DataFrame | None:
    """The max-over-columns KS null is whatever the noisiest column produces.
    On IEEE-CIS that is a column which is already mostly NaN, so its effective
    sample size -- and therefore its null KS -- is the same one that masking 90%
    of a dense column creates. Needs the cached real bundle."""
    pkl = ROOT / "artifacts" / "real_bundle.pkl"
    if not pkl.exists():
        return None
    import joblib

    from driftharm.detectors import _ks_col

    b = joblib.load(pkl)
    X = b.pool_X
    nan_rate = np.isnan(X).mean(axis=0)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), 40_000, replace=False)
    ref, cur = X[idx[:20_000]], X[idx[20_000:]]
    masked = set(int(j) for j in np.argsort(-b.importance)[:6])
    rows = []
    for j in range(X.shape[1]):
        rows.append({
            "feature": b.features[j],
            "nan_rate_in_pool": float(nan_rate[j]),
            "effective_n_per_window": int(round((1 - nan_rate[j]) * 20_000)),
            "null_ks": _ks_col(ref[:, j], cur[:, j]),
            "gain_importance": float(b.importance[j]),
            "masked_by_imputation_archetype": j in masked,
        })
    return pd.DataFrame(rows).sort_values("null_ks", ascending=False, ignore_index=True)


# -----------------------------------------------------------------------------


def main() -> None:
    trials = {w: load(w) for w in DATASETS}
    point = {w: mcc_vector(trials[w]["harm"].to_numpy(), alarm_matrix(trials[w]))
             for w in DATASETS}
    observed = float(stats.spearmanr(point["real"], point["synthetic"]).statistic)

    print("=== 0. the disagreement being explained ===")
    print(pd.DataFrame({"detector": DETECTOR_NAMES,
                        "mcc_real": point["real"],
                        "mcc_synthetic": point["synthetic"]}).round(3).to_string(index=False))
    print(f"cross-dataset Spearman = {observed:+.3f}\n")

    for w in DATASETS:
        tab = rank_stability_table(trials[w])
        tab.to_csv(REPORTS / f"{w}_rank_stability.csv", index=False)
        print(f"=== 1. {w}: MCC interval and P(rank) by resampling scheme ===")
        print(tab[["scheme", "detector", "mcc", "mcc_lo95", "mcc_hi95", "ci_width",
                   "p_mcc_above_zero", "p_rank1"]].round(3).to_string(index=False))
        print()

    agree = []
    for w in DATASETS:
        for scheme in BOOTSTRAP_SCHEMES:
            rec = self_agreement(trials[w], scheme, observed)
            rec["dataset"] = w
            rec["observed_cross_dataset_spearman"] = observed
            agree.append(rec)
    agree = pd.DataFrame(agree)[
        ["dataset", "scheme", "n_draws", "n_degenerate", "self_spearman_mean",
         "self_spearman_p05", "self_spearman_p50",
         "observed_cross_dataset_spearman", "p_self_spearman_at_or_below_observed"]
    ]
    agree.to_csv(REPORTS / "ranking_agreement.csv", index=False)
    print("=== 2. Spearman between two resamples of the SAME dataset ===")
    print(agree.round(3).to_string(index=False))
    print()

    for w in DATASETS:
        loo = leave_one_archetype_out(trials[w])
        loo.to_csv(REPORTS / f"{w}_leave_one_archetype_out.csv", index=False)
        print(f"=== 3. {w}: drop one archetype, re-rank ===")
        print(loo[["dropped_archetype", "winner", "spearman_vs_full_suite"]]
              .round(3).to_string(index=False))
        print()

    dis = archetype_disagreement()
    dis.to_csv(REPORTS / "archetype_disagreement.csv", index=False)
    print("=== 4. the same archetype, measured on the two datasets ===")
    print(dis[["archetype", "harm_rate_real", "harm_rate_synthetic", "harm_rate_diff",
               "mean_abs_alarm_rate_diff"]].round(3).to_string(index=False))
    print(f"mean |harm-rate difference| over 12 archetypes = "
          f"{dis['harm_rate_diff'].abs().mean():.3f}\n")

    swap = swap_decomposition(trials["real"], trials["synthetic"])
    swap.to_csv(REPORTS / "ranking_swap_decomposition.csv", index=False)
    print("=== 5. alarms from one dataset, harm labels from the other ===")
    print(swap.round(3).to_string(index=False))
    print()

    fp = masked_null_footprint()
    fp.to_csv(REPORTS / "masked_null_footprint.csv", index=False)
    print("=== 6. imputation_masked_null: the footprint against each dataset's own null ===")
    print(fp.round(5).to_string(index=False))
    print()

    floor = real_null_floor_by_column()
    if floor is None:
        print("=== 7. skipped: artifacts/real_bundle.pkl not present ===")
    else:
        floor.to_csv(REPORTS / "real_null_floor_by_column.csv", index=False)
        print("=== 7. real data: what sets the max-KS null floor (top 8 columns) ===")
        print(floor.head(8).round(4).to_string(index=False))
        print(f"monitored columns with NaN rate > 0.8: "
              f"{int((floor['nan_rate_in_pool'] > 0.8).sum())} of {len(floor)}")


if __name__ == "__main__":
    main()
