"""Tests for the scenario generators.

Each archetype makes a structural claim -- "P(x) is untouched", "only
zero-importance columns move", "only 3% of rows change". If a generator quietly
violates its own claim, every number derived from it is wrong and nothing else
in the harness would notice.
"""

import numpy as np
import pytest

from driftharm.scenarios import ARCHETYPES


def _run(name, rng, bundle, ctx, **kw):
    X = bundle.pool_X[4000:8000]
    y = bundle.pool_y[4000:8000]
    return ARCHETYPES[name].fn(rng, X, y, {**ctx, "scores": bundle.predict(X)}, **kw), X, y


def test_true_null_changes_nothing(rng, small_bundle, ctx):
    w, X, y = _run("true_null", rng, small_bundle, ctx)
    assert np.array_equal(w.X_model, X)
    assert np.array_equal(w.X_monitor, X)
    assert np.array_equal(w.y, y)


def test_concept_drift_leaves_the_inputs_bit_identical(rng, small_bundle, ctx):
    """The defining property: P(x) is not perturbed at all, only P(y|x).

    If a single input value moved, an input-distribution detector could in
    principle catch this scenario, and the blind-spot claim would be unearned.
    """
    w, X, y = _run("concept_drift_no_covariate_shift", rng, small_bundle, ctx, flip=0.3)
    assert np.array_equal(w.X_model, X)
    assert np.array_equal(w.X_monitor, X)
    assert 0.2 < (w.y != y).mean() < 0.4


def test_covariate_shift_only_resamples_real_rows(rng, small_bundle, ctx):
    """Rows are drawn whole, so every (x, y) pair in the output is a pair that
    really occurred. That is what makes P(y|x) exactly preserved rather than
    approximately preserved, which is the difference between a covariate-shift
    scenario and a mislabelled concept-drift one."""
    w, X, y = _run("covariate_shift_moderate", rng, small_bundle, ctx)
    keys = {tuple(r) + (int(t),) for r, t in zip(X, y)}
    for r, t in zip(w.X_model[:400], w.y[:400]):
        assert tuple(r) + (int(t),) in keys


def test_covariate_shift_actually_moves_the_distribution(rng, small_bundle, ctx):
    """A 'false alarm' scenario is only interesting if there is a real shift to
    alarm on."""
    from driftharm.detectors import DETECTORS

    ref = small_bundle.pool_X[:4000]
    w, _, _ = _run("covariate_shift_moderate", rng, small_bundle, ctx)
    assert DETECTORS["ks"](ref, w.X_monitor) > 0.1


def test_irrelevant_feature_drift_touches_only_zero_importance_columns(
    rng, small_bundle, ctx
):
    """The harm claim rests entirely on these columns having weight zero in the
    true model; if the generator strays into an informative column the
    'zero harm by construction' label is a lie."""
    w, X, _ = _run("irrelevant_feature_drift", rng, small_bundle, ctx)
    changed = np.where(~np.all(np.isclose(w.X_model, X, equal_nan=True), axis=0))[0]
    assert set(changed) == set(small_bundle.zero_idx.tolist())
    assert len(changed) > 0


def test_imputation_archetype_splits_the_two_views(rng, small_bundle, ctx):
    """The monitor sees NaN, the model sees the reference median. Same rows,
    same failure, different observability -- that gap is the measurement."""
    w, X, _ = _run("imputation_masked_null", rng, small_bundle, ctx, null_rate=0.5, n_cols=2)
    cols = w.meta["cols"]
    assert np.isnan(w.X_monitor[:, cols]).mean() == pytest.approx(0.5, abs=0.05)
    assert not np.isnan(w.X_model).any()
    untouched = [j for j in range(X.shape[1]) if j not in cols]
    assert np.array_equal(w.X_model[:, untouched], X[:, untouched])


def test_imputation_visible_shows_the_model_view_to_the_monitor(rng, small_bundle, ctx):
    w, _, _ = _run("imputation_visible", rng, small_bundle, ctx, null_rate=0.5, n_cols=2)
    assert np.array_equal(w.X_monitor, w.X_model)
    assert not np.isnan(w.X_monitor).any()


def test_dilution_shift_changes_only_the_segment(rng, small_bundle, ctx):
    """3% of volume means 3% of rows, not 3% of the effect. If the corruption
    leaked outside the segment the aggregate detectors would be being tested on
    the wrong thing."""
    w, X, _ = _run("dilution_shift", rng, small_bundle, ctx, seg_frac=0.03)
    moved = np.where(~np.all(np.isclose(w.X_model, X, equal_nan=True), axis=1))[0]
    assert len(moved) == pytest.approx(0.03 * len(X), rel=0.15)
    assert set(moved.tolist()) == set(w.meta["seg_rows"].tolist())


def test_dilution_shift_segment_concentrates_positives(rng, small_bundle, ctx):
    """The archetype is only meaningful when the small segment carries an
    outsized share of the label mass."""
    w, X, y = _run("dilution_shift", rng, small_bundle, ctx, seg_frac=0.03)
    assert w.meta["seg_positive_share"] > 3 * w.meta["seg_frac"]


def test_gradual_moves_less_than_sudden(rng, small_bundle, ctx):
    """Same corruption, partially delivered. If gradual were not strictly
    smaller, the detection-delay comparison would be meaningless."""
    g, X, _ = _run("gradual_shift", rng, small_bundle, ctx, frac=0.25, strength=1.0)
    s, _, _ = _run("sudden_shift", rng, small_bundle, ctx, strength=1.0)
    assert np.abs(g.X_model - X).mean() < np.abs(s.X_model - X).mean()


def test_every_archetype_preserves_row_count_and_shape(rng, small_bundle, ctx):
    for name in ARCHETYPES:
        w, X, y = _run(name, rng, small_bundle, ctx)
        assert w.X_model.shape == X.shape, name
        assert w.X_monitor.shape == X.shape, name
        assert len(w.y) == len(y), name


def test_dilution_permuted_preserves_every_marginal_exactly(rng, small_bundle, ctx):
    """Shuffling values inside the segment moves rows, not distributions.

    Sorting each column gives the identical vector before and after, so any
    detector working from marginals alone has literally nothing to see. This is
    what separates the two dilution variants: one is diluted, the other is
    invisible by arithmetic.
    """
    w, X, _ = _run("dilution_permuted", rng, small_bundle, ctx)
    for j in range(X.shape[1]):
        assert np.array_equal(np.sort(w.X_model[:, j]), np.sort(X[:, j])), j
    moved = np.where(~np.all(np.isclose(w.X_model, X, equal_nan=True), axis=1))[0]
    assert 0 < len(moved) <= len(w.meta["seg_rows"])


def test_covariate_shift_strength_ladder_is_monotone(rng, small_bundle, ctx):
    """The three rungs must actually differ, or the dose-response result in the
    README is measuring the same scenario three times."""
    from driftharm.detectors import DETECTORS

    ref = small_bundle.pool_X[:4000]
    scores = []
    for name in ("covariate_shift_mild", "covariate_shift_moderate",
                 "covariate_shift_strong"):
        w, _, _ = _run(name, rng, small_bundle, ctx)
        scores.append(DETECTORS["ks"](ref, w.X_monitor))
    assert scores == sorted(scores), scores
