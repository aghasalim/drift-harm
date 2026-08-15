"""The failure-case taxonomy.

Each archetype is a transform applied to a *current* window that was drawn from
the same held-out pool as the reference window. Because both windows start out
exchangeable, anything the detectors see afterwards is the archetype and nothing
else.

Two output views matter and they are not the same thing:

    X_model    what the deployed model scores
    X_monitor  what the drift detector sees

In production these are usually different -- monitoring reads the raw ingested
table, the model reads the post-imputation feature vector. One archetype exists
purely to measure what that gap costs.

`expected_harm` is the archetype's design intent. It is a hypothesis, not the
answer: harm is measured separately in `harm.py` and the two are compared. When
they disagree, the disagreement is the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Window:
    X_model: np.ndarray
    X_monitor: np.ndarray
    y: np.ndarray
    meta: dict = field(default_factory=dict)


@dataclass
class Archetype:
    name: str
    expected_harm: bool
    expected_alarm: str  # 'fire' | 'quiet' | 'unknown'
    description: str
    fn: Callable


ARCHETYPES: dict[str, Archetype] = {}


def archetype(name, expected_harm, expected_alarm, description):
    def deco(fn):
        ARCHETYPES[name] = Archetype(name, expected_harm, expected_alarm, description, fn)
        return fn

    return deco


# --------------------------------------------------------------------------


def _plain(X, y, **meta) -> Window:
    return Window(X_model=X, X_monitor=X.copy(), y=y, meta=meta)


@archetype(
    "true_null",
    expected_harm=False,
    expected_alarm="quiet",
    description=(
        "Two disjoint random windows from the same pool. Nothing has changed. "
        "Any alarm here is a false alarm and any measured harm here is noise; "
        "this is the archetype every threshold is calibrated against."
    ),
)
def true_null(rng, X, y, ctx):
    return _plain(X.copy(), y.copy())


def _covariate_shift(rng, X, y, ctx, strength):
    # Pick a feature with real but not dominant importance, so the shift is
    # correlated with things the model uses rather than an isolated poke.
    order = np.argsort(-ctx["importance"])
    j = int(order[min(12, len(order) - 1)])
    v = X[:, j].astype("float64")
    v = np.where(np.isfinite(v), v, np.nanmedian(v))
    z = (v - v.mean()) / (v.std() + 1e-9)
    w = np.exp(strength * np.clip(z, -3, 3))
    w = w / w.sum()
    idx = rng.choice(len(X), size=len(X), replace=True, p=w)
    return _plain(X[idx], y[idx], tilt_feature=int(j), strength=strength)


_COV_DOC = (
    "Genuine covariate shift produced by importance-weighted resampling of real "
    "rows on a mid-importance feature. Because whole (x, y) rows are drawn, "
    "P(y|x) is preserved exactly and only P(x) moves; many monitored features "
    "shift together the way they do in real traffic. Run at three strengths "
    "because whether covariate shift is benign turned out to be a question with "
    "a measured answer rather than an assumed one -- see the README."
)

for _s, _label, _hyp in (
    (0.5, "covariate_shift_mild", False),
    (1.0, "covariate_shift_moderate", False),
    (2.0, "covariate_shift_strong", False),
):
    archetype(_label, expected_harm=_hyp, expected_alarm="fire",
              description=f"{_COV_DOC} (tilt strength {_s})")(
        (lambda s: lambda rng, X, y, ctx, strength=s: _covariate_shift(rng, X, y, ctx, strength))(_s)
    )


@archetype(
    "concept_drift_no_covariate_shift",
    expected_harm=True,
    expected_alarm="quiet",
    description=(
        "P(y|x) changes while P(x) is bit-identical: the same feature rows, with "
        "a fraction of labels flipped. Every input-distribution detector is blind "
        "here by construction -- it is not a tuning failure, it is the definition "
        "of the method. Included to make the blind spot measurable rather than "
        "argued about."
    ),
)
def concept_drift_no_covariate_shift(rng, X, y, ctx, flip: float = 0.35):
    y2 = y.copy()
    m = rng.random(len(y2)) < flip
    y2[m] = 1 - y2[m]
    return _plain(X.copy(), y2, flip_rate=flip, n_flipped=int(m.sum()))


@archetype(
    "imputation_masked_null",
    expected_harm=True,
    expected_alarm="quiet",
    description=(
        "An upstream feed degrades: a high-importance column arrives NaN for a "
        "fraction of rows. Serving imputes the reference median, so the model "
        "loses information. Monitoring reads the raw column, where dropping "
        "non-finite values leaves the surviving values unchanged -- so the "
        "detector compares two identical distributions and sees nothing."
    ),
)
def imputation_masked_null(rng, X, y, ctx, null_rate: float = 0.9, n_cols: int = 6):
    cols = np.argsort(-ctx["importance"])[:n_cols]
    Xm = X.copy()  # what the monitor reads: raw, with holes
    Xs = X.copy()  # what the model scores: imputed
    med = ctx["ref_median"]
    for j in cols:
        m = rng.random(len(X)) < null_rate
        Xm[m, j] = np.nan
        Xs[m, j] = med[j]
    return Window(
        X_model=Xs, X_monitor=Xm, y=y.copy(),
        meta={"cols": [int(c) for c in cols], "null_rate": null_rate},
    )


@archetype(
    "imputation_visible",
    expected_harm=True,
    expected_alarm="unknown",
    description=(
        "The same feed failure, but monitoring is wired to the post-imputation "
        "feature vector the model actually consumes. Identical harm, different "
        "observability. The pair isolates the cost of monitoring the wrong table."
    ),
)
def imputation_visible(rng, X, y, ctx, null_rate: float = 0.9, n_cols: int = 6):
    w = imputation_masked_null(rng, X, y, ctx, null_rate=null_rate, n_cols=n_cols)
    return Window(X_model=w.X_model, X_monitor=w.X_model.copy(), y=w.y, meta=w.meta)


def _segment(ctx, X, seg_frac):
    """The top `seg_frac` of rows by the model's own predicted risk -- the
    high-value slice an operator would actually carve out. Small volume, large
    share of the positives."""
    k = max(1, int(len(X) * seg_frac))
    return np.argsort(-ctx["scores"])[:k]


def _seg_meta(y, seg, seg_frac):
    return dict(
        seg_frac=seg_frac,
        seg_size=int(len(seg)),
        seg_positive_share=float(y[seg].sum() / max(y.sum(), 1)),
        seg_rows=seg,
    )


@archetype(
    "dilution_shift",
    expected_harm=True,
    expected_alarm="quiet",
    description=(
        "Drift confined to a small, disproportionately risky segment: an "
        "out-of-distribution additive shift on 3% of rows that carry a large "
        "share of the positives. The aggregate marginals barely move because "
        "97% of the mass is untouched."
    ),
)
def dilution_shift(rng, X, y, ctx, seg_frac: float = 0.03, strength: float = 4.0,
                   n_cols: int = 10):
    seg = _segment(ctx, X, seg_frac)
    cols = np.argsort(-ctx["importance"])[:n_cols]
    Xc = X.copy()
    for j in cols:
        sd = np.nanstd(Xc[:, j])
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        Xc[seg, j] = Xc[seg, j] + strength * sd * rng.normal(size=len(seg))
    return _plain(Xc, y.copy(), **_seg_meta(y, seg, seg_frac))


@archetype(
    "dilution_permuted",
    expected_harm=True,
    expected_alarm="quiet",
    description=(
        "The same 3% segment, but its feature values are shuffled among the "
        "segment's own rows. Every marginal distribution is preserved exactly -- "
        "only the joint structure inside the segment is destroyed. Univariate "
        "detectors are blind by arithmetic; the multivariate ones have something "
        "to find, if 3% of the batch is enough for them to find it."
    ),
)
def dilution_permuted(rng, X, y, ctx, seg_frac: float = 0.03):
    seg = _segment(ctx, X, seg_frac)
    Xc = X.copy()
    for j in range(X.shape[1]):
        Xc[seg, j] = Xc[rng.permutation(seg), j]
    return _plain(Xc, y.copy(), **_seg_meta(y, seg, seg_frac))


@archetype(
    "irrelevant_feature_drift",
    expected_harm=False,
    expected_alarm="fire",
    description=(
        "A large shift in monitored columns whose gain importance is exactly "
        "zero. The model cannot be affected -- these columns never appear in a "
        "split -- so any alarm is a true false alarm."
    ),
)
def irrelevant_feature_drift(rng, X, y, ctx, shift_sd: float = 3.0):
    zero = ctx["zero_idx"]
    if len(zero) == 0:
        raise ValueError("no zero-importance columns available")
    Xc = X.copy()
    for j in zero:
        col = Xc[:, j]
        sd = np.nanstd(col)
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        Xc[:, j] = col + shift_sd * sd
    return _plain(Xc, y.copy(), n_cols=int(len(zero)), shift_sd=shift_sd)


@archetype(
    "sudden_shift",
    expected_harm=True,
    expected_alarm="fire",
    description=(
        "A harmful shift delivered in full, in one batch: the top features are "
        "corrupted with additive noise scaled to their own spread."
    ),
)
def sudden_shift(rng, X, y, ctx, strength: float = 1.0, n_cols: int = 10):
    return _graded_shift(rng, X, y, ctx, frac=1.0, strength=strength, n_cols=n_cols)


@archetype(
    "gradual_shift",
    expected_harm=True,
    expected_alarm="unknown",
    description=(
        "The same total corruption as sudden_shift, but only the fraction that "
        "has accrued by the batch under test. Reported as a per-batch curve, so "
        "detection delay is read off directly rather than asserted."
    ),
)
def gradual_shift(rng, X, y, ctx, strength: float = 1.0, n_cols: int = 10, frac: float = 0.25):
    return _graded_shift(rng, X, y, ctx, frac=frac, strength=strength, n_cols=n_cols)


def _graded_shift(rng, X, y, ctx, frac, strength, n_cols):
    cols = np.argsort(-ctx["importance"])[:n_cols]
    Xc = X.copy()
    for j in cols:
        col = Xc[:, j].astype("float64")
        sd = np.nanstd(col)
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        Xc[:, j] = col + frac * strength * sd * rng.normal(size=len(col))
    return _plain(Xc, y.copy(), frac=frac, strength=strength, n_cols=int(n_cols))


ARCHETYPE_NAMES = list(ARCHETYPES)
