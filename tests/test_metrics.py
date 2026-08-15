"""Tests for the ranking metric and the threshold calibration.

The calibration test is the important one: it checks that a threshold fitted on
one set of null replicates actually holds its false-alarm rate on replicates it
has never seen. Without that, every alarm rate in the benchmark is an in-sample
number agreeing with itself.
"""

import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef

from driftharm.calibration import draw_pair, harm_threshold, null_run, thresholds
from driftharm.detectors import DETECTORS
from driftharm.metrics import alignment, rank_detectors


def test_perfect_detector_scores_one():
    harm = np.array([1, 1, 0, 0, 1, 0])
    r = alignment(harm, harm)
    assert r["mcc"] == pytest.approx(1.0)
    assert r["harm_f1"] == pytest.approx(1.0)


def test_always_alarm_gets_zero_mcc_despite_perfect_recall():
    """The reason MCC is the headline and recall is not: a detector that fires
    on everything catches every harmful case and is still worthless."""
    harm = np.array([1, 1, 0, 0, 1, 0])
    always = np.ones_like(harm)
    r = alignment(harm, always)
    assert r["harm_recall"] == pytest.approx(1.0)
    assert r["mcc"] == pytest.approx(0.0)
    assert r["specificity"] == pytest.approx(0.0)


def test_mcc_matches_sklearn():
    rng = np.random.default_rng(3)
    harm = rng.integers(0, 2, 200)
    alarm = np.where(rng.random(200) < 0.7, harm, 1 - harm)
    assert alignment(harm, alarm)["mcc"] == pytest.approx(matthews_corrcoef(harm, alarm))


def test_anticorrelated_detector_scores_negative():
    """A detector that is quiet exactly when it should fire is worse than a coin
    flip, and the metric has to say so rather than bottoming out at zero."""
    harm = np.array([1, 1, 1, 0, 0, 0])
    assert alignment(harm, 1 - harm)["mcc"] < -0.9


def test_rank_detectors_orders_by_mcc():
    import pandas as pd

    harm = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    df = pd.DataFrame({
        "harm": harm,
        "alarm_good": harm,
        "alarm_bad": np.ones_like(harm),
    })
    out = rank_detectors(df, ["bad", "good"])
    assert list(out["detector"]) == ["good", "bad"]


def test_draw_pair_is_disjoint():
    """Reference and current windows sharing rows would deflate every null
    statistic and quietly inflate every reported alarm rate."""
    rng = np.random.default_rng(0)
    a, b = draw_pair(rng, 10_000, 1500)
    assert len(a) == len(b) == 1500
    assert len(np.intersect1d(a, b)) == 0


def test_thresholds_hold_their_false_alarm_rate_out_of_sample(small_bundle):
    """Calibrate on half the null replicates, measure on the other half.

    This is the instrument validation the rest of the benchmark leans on. A
    detector whose held-out false-alarm rate is nowhere near the target alpha
    cannot be compared fairly against one whose rate is.
    """
    dets = {k: DETECTORS[k] for k in ("ks", "psi", "wasserstein", "jensen_shannon")}
    nd = null_run(small_bundle, n_reps=24, window=1200, seed=11, detectors=dets)
    cal, val = nd.iloc[:12], nd.iloc[12:]
    taus = thresholds(cal, alpha=0.05, detectors=dets)
    for name in dets:
        far = float((val[name] > taus[name]).mean())
        assert far <= 0.35, (name, far)


def test_null_auc_drop_is_centred_on_zero(small_bundle):
    """Two exchangeable windows scored by a model that saw neither must not
    show systematic degradation. A non-zero centre means leakage."""
    dets = {"ks": DETECTORS["ks"]}
    nd = null_run(small_bundle, n_reps=20, window=1500, seed=5, detectors=dets)
    assert abs(nd["auc_drop"].mean()) < 0.02
    assert harm_threshold(nd) > 0


def test_harm_threshold_is_strictly_above_the_null_median(small_bundle):
    dets = {"ks": DETECTORS["ks"]}
    nd = null_run(small_bundle, n_reps=20, window=1500, seed=6, detectors=dets)
    assert harm_threshold(nd, alpha=0.05) > nd["auc_drop"].median()
