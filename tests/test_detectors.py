"""Tests for the instrument itself.

These are not smoke tests. Each one pins a property that, if it broke silently,
would invalidate a published number: the null behaviour, monotonicity in shift
size, scale invariance, and the NaN policy that one whole archetype depends on.
"""

import numpy as np
import pytest

from driftharm.detectors import (
    DETECTORS,
    _js_distance,
    _ks_col,
    _psi_col,
    _wasserstein_col,
    c2st,
    mmd,
)

UNIVARIATE = ["ks", "psi", "wasserstein", "jensen_shannon"]


def _gauss(rng, n, d, shift=0.0):
    return (rng.normal(size=(n, d)) + shift).astype("float32")


@pytest.mark.parametrize("name", list(DETECTORS))
def test_identical_input_scores_near_zero(name, rng):
    """A detector handed the same matrix twice must not claim drift."""
    X = _gauss(rng, 3000, 6)
    assert DETECTORS[name](X, X.copy()) < 0.08


@pytest.mark.parametrize("name", UNIVARIATE)
def test_monotone_in_shift_size(name, rng):
    """Bigger shift, bigger score. A detector that is not monotone in the
    effect it claims to measure cannot be thresholded meaningfully."""
    det = DETECTORS[name]
    ref = _gauss(rng, 4000, 4)
    scores = [det(ref, _gauss(rng, 4000, 4, shift=s)) for s in (0.0, 0.25, 0.75, 2.0)]
    assert scores == sorted(scores), scores
    assert scores[-1] > scores[0] * 3


def test_wasserstein_is_scale_invariant(rng):
    """Scaled by the reference IQR, so a column in cents and the same column in
    dollars contribute equally to the max. Without this the aggregate is just
    whichever feature has the biggest units."""
    a = rng.normal(size=5000)
    b = rng.normal(loc=0.5, size=5000)
    plain = _wasserstein_col(a, b)
    scaled = _wasserstein_col(a * 1000, b * 1000)
    assert plain == pytest.approx(scaled, rel=1e-6)


def test_js_distance_is_bounded_and_saturates(rng):
    """Jensen-Shannon distance lives in [0, 1]; a value outside means the
    divergence/distance conversion is wrong.

    It does not reach 1.0 even for completely disjoint samples, and that is a
    property of every binned detector here, not an error: the outermost
    reference bins are open-ended, so a distribution shifted far away still
    lands inside the top bin and shares 1/n_bins of the reference mass with it.
    Binned detectors saturate; the rank-based ones (KS, Wasserstein) do not.
    """
    a = rng.normal(size=3000)
    far = rng.normal(loc=50, size=3000)
    assert 0.0 <= _js_distance(a, a.copy()) <= 1.0
    d = _js_distance(a, far)
    assert 0.85 <= d <= 1.0
    # more bins -> less mass trapped in the open top bin -> closer to the max
    assert _js_distance(a, far, n_bins=100) > d


def test_psi_is_zero_for_the_reference_against_itself(rng):
    a = rng.normal(size=5000)
    assert _psi_col(a, a.copy()) == pytest.approx(0.0, abs=1e-9)


def test_univariate_detectors_are_blind_to_dropped_nans(rng):
    """The mechanism behind the imputation archetype, pinned as a test.

    Punching NaN holes in a column leaves the surviving values distributed
    exactly as before. Every detector that drops non-finite values therefore
    compares two identical distributions and reports almost nothing -- even
    though 60% of the column's information is gone.
    """
    a = rng.normal(size=20000)
    b = a.copy()
    b[rng.random(len(b)) < 0.6] = np.nan
    assert _ks_col(a, b) < 0.03
    assert _psi_col(a, b) < 0.01
    assert _js_distance(a, b) < 0.06


def test_median_imputation_is_loudly_visible(rng):
    """The same feed failure, imputed instead of left as NaN, is not subtle.

    This is the other half of the archetype: harm is identical, but the value
    the monitor reads decides whether anything is seen at all.
    """
    a = rng.normal(size=20000)
    b = a.copy()
    m = rng.random(len(b)) < 0.6
    b[m] = np.median(a)
    assert _ks_col(a, b) > 0.2
    assert _psi_col(a, b) > 0.3


def test_mmd_separates_a_mean_shift_from_the_null(rng):
    ref = _gauss(rng, 3000, 8)
    null = mmd(ref, _gauss(rng, 3000, 8))
    shifted = mmd(ref, _gauss(rng, 3000, 8, shift=0.4))
    assert shifted > null
    assert shifted > 0.02


def test_c2st_is_centred_on_zero_under_the_null(rng):
    """Returned as AUC - 0.5, so an honest discriminator on exchangeable
    samples should land near zero, not near 0.5."""
    ref = _gauss(rng, 4000, 8)
    assert abs(c2st(ref, _gauss(rng, 4000, 8))) < 0.05
    assert c2st(ref, _gauss(rng, 4000, 8, shift=0.7)) > 0.2


def test_detectors_tolerate_a_constant_column(rng):
    """Degenerate columns appear in real tables. Nothing may raise or return NaN."""
    X = _gauss(rng, 1500, 4)
    X[:, 2] = 3.0
    Y = _gauss(rng, 1500, 4)
    Y[:, 2] = 3.0
    for name, det in DETECTORS.items():
        v = det(X, Y)
        assert np.isfinite(v), name


def test_detectors_tolerate_an_all_nan_column(rng):
    """A column that arrives completely empty must not crash the run. It also
    must not be silently reported as fine -- which is exactly what the
    drop-NaN policy does, so this test documents the behaviour rather than
    endorsing it."""
    X = _gauss(rng, 1500, 4)
    Y = _gauss(rng, 1500, 4)
    Y[:, 1] = np.nan
    for name in UNIVARIATE:
        v = DETECTORS[name](X, Y)
        assert np.isfinite(v)
    assert _ks_col(X[:, 1], Y[:, 1]) == 0.0  # the blind spot, in one line
