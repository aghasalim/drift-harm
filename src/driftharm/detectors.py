"""Six drift detectors, all reduced to a single comparable scalar.

Every detector here maps (reference matrix, current matrix) to one number where
larger means "more drifted". No detector carries its own alarm threshold: the
thresholds are measured against a null in `calibration.py`, so a p-value-based
detector and an effect-size-based detector end up on the same footing.

NaN policy is a real design decision and is stated per detector, because in one
of the failure archetypes the NaN policy is the entire result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats

EPS = 1e-8


def _clean_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop non-finite values independently from each side.

    This is what scipy.stats.ks_2samp and essentially every off-the-shelf PSI
    implementation do. It is also the reason a column that arrives empty can
    score zero drift.
    """
    return a[np.isfinite(a)], b[np.isfinite(b)]


# --------------------------------------------------------------------------
# univariate statistics (per column, then aggregated)
# --------------------------------------------------------------------------


def _ks_col(ref: np.ndarray, cur: np.ndarray) -> float:
    ref, cur = _clean_pair(ref, cur)
    if len(ref) < 2 or len(cur) < 2:
        return 0.0
    return float(stats.ks_2samp(ref, cur).statistic)


def _bin_edges(ref: np.ndarray, n_bins: int) -> np.ndarray:
    q = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(ref, q))
    if len(edges) < 2:  # degenerate reference column
        edges = np.array([ref[0] - 1.0, ref[0] + 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _hist_pair(ref: np.ndarray, cur: np.ndarray, n_bins: int):
    edges = _bin_edges(ref, n_bins)
    p = np.histogram(ref, bins=edges)[0].astype(float)
    q = np.histogram(cur, bins=edges)[0].astype(float)
    p = p / max(p.sum(), 1.0)
    q = q / max(q.sum(), 1.0)
    return p, q


def _psi_col(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    ref, cur = _clean_pair(ref, cur)
    if len(ref) < 2 or len(cur) < 2:
        return 0.0
    p, q = _hist_pair(ref, cur, n_bins)
    p, q = np.clip(p, EPS, None), np.clip(q, EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def _js_distance(ref: np.ndarray, cur: np.ndarray, n_bins: int = 20) -> float:
    """Jensen-Shannon *distance* (sqrt of the divergence), in [0, 1]."""
    ref, cur = _clean_pair(ref, cur)
    if len(ref) < 2 or len(cur) < 2:
        return 0.0
    p, q = _hist_pair(ref, cur, n_bins)
    m = 0.5 * (p + q)
    div = 0.5 * stats.entropy(p, m, base=2) + 0.5 * stats.entropy(q, m, base=2)
    return float(np.sqrt(max(div, 0.0)))


def _wasserstein_col(ref: np.ndarray, cur: np.ndarray) -> float:
    """Wasserstein-1 scaled by the reference IQR so columns are comparable.

    Without scaling, the aggregate is dominated by whichever column happens to
    have the largest units (TransactionAmt, always).
    """
    ref, cur = _clean_pair(ref, cur)
    if len(ref) < 2 or len(cur) < 2:
        return 0.0
    scale = np.subtract(*np.percentile(ref, [75, 25]))
    if not np.isfinite(scale) or scale <= 0:
        scale = np.std(ref)
    if not np.isfinite(scale) or scale <= 0:
        return 0.0
    return float(stats.wasserstein_distance(ref, cur) / scale)


def _aggregate(ref: np.ndarray, cur: np.ndarray, col_fn: Callable) -> float:
    """Max over columns: the standard 'alert if any monitored feature drifts'.

    Max makes the multiple-comparison burden real rather than hidden, and the
    calibration step prices it in.
    """
    return max(col_fn(ref[:, j], cur[:, j]) for j in range(ref.shape[1]))


# --------------------------------------------------------------------------
# multivariate statistics
# --------------------------------------------------------------------------


def _impute_standardize(ref: np.ndarray, cur: np.ndarray):
    """Reference-median impute + reference standardize.

    Multivariate detectors cannot consume NaN. Imputing with the reference
    median is the same thing a serving pipeline does, which means these two
    detectors see post-imputation data while the univariate ones see raw data.
    That asymmetry is measured, not swept away -- see the imputation archetype.
    """
    med = np.nanmedian(ref, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    r = np.where(np.isfinite(ref), ref, med)
    c = np.where(np.isfinite(cur), cur, med)
    mu, sd = r.mean(0), r.std(0)
    sd = np.where(sd > EPS, sd, 1.0)
    return (r - mu) / sd, (c - mu) / sd


def mmd(ref: np.ndarray, cur: np.ndarray, n_sub: int = 1500, seed: int = 0) -> float:
    """Unbiased MMD^2 with an RBF kernel, median-heuristic bandwidth.

    Subsampled to `n_sub` per side: the kernel matrix is quadratic and the null
    calibration needs hundreds of repeats.
    """
    r, c = _impute_standardize(ref, cur)
    rng = np.random.default_rng(seed)
    if len(r) > n_sub:
        r = r[rng.choice(len(r), n_sub, replace=False)]
    if len(c) > n_sub:
        c = c[rng.choice(len(c), n_sub, replace=False)]

    z = np.vstack([r, c])
    d2 = np.maximum(
        (z * z).sum(1)[:, None] + (z * z).sum(1)[None, :] - 2 * z @ z.T, 0.0
    )
    n = len(r)
    med = np.median(d2[np.triu_indices_from(d2, k=1)])
    gamma = 1.0 / max(med, EPS)
    K = np.exp(-gamma * d2)

    Krr, Kcc, Krc = K[:n, :n], K[n:, n:], K[:n, n:]
    m = len(c)
    term_r = (Krr.sum() - np.trace(Krr)) / (n * (n - 1))
    term_c = (Kcc.sum() - np.trace(Kcc)) / (m * (m - 1))
    return float(term_r + term_c - 2 * Krc.mean())


def c2st(ref: np.ndarray, cur: np.ndarray, seed: int = 0, n_sub: int = 8000) -> float:
    """Classifier two-sample test: held-out AUC of a reference-vs-current discriminator.

    Returned as AUC - 0.5 so that "no drift" sits at zero like the other
    detectors. Uses a gradient-boosted tree, so it picks up interactions and
    marginal shifts alike.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    r, c = _impute_standardize(ref, cur)
    rng = np.random.default_rng(seed)
    if len(r) > n_sub:
        r = r[rng.choice(len(r), n_sub, replace=False)]
    if len(c) > n_sub:
        c = c[rng.choice(len(c), n_sub, replace=False)]

    X = np.vstack([r, c])
    y = np.r_[np.zeros(len(r)), np.ones(len(c))]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    clf = lgb.LGBMClassifier(
        n_estimators=120, learning_rate=0.1, num_leaves=31,
        min_child_samples=40, random_state=seed, n_jobs=-1, verbose=-1,
    )
    clf.fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]) - 0.5)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Detector:
    name: str
    fn: Callable[[np.ndarray, np.ndarray], float]
    multivariate: bool
    nan_policy: str
    note: str

    def __call__(self, ref: np.ndarray, cur: np.ndarray) -> float:
        return float(self.fn(ref, cur))


DETECTORS: dict[str, Detector] = {
    "ks": Detector(
        "ks", lambda r, c: _aggregate(r, c, _ks_col), False, "drop",
        "max two-sample Kolmogorov-Smirnov statistic over monitored columns",
    ),
    "psi": Detector(
        "psi", lambda r, c: _aggregate(r, c, _psi_col), False, "drop",
        "max Population Stability Index (10 reference-quantile bins)",
    ),
    "wasserstein": Detector(
        "wasserstein", lambda r, c: _aggregate(r, c, _wasserstein_col), False, "drop",
        "max IQR-scaled Wasserstein-1 distance",
    ),
    "jensen_shannon": Detector(
        "jensen_shannon", lambda r, c: _aggregate(r, c, _js_distance), False, "drop",
        "max Jensen-Shannon distance (20 reference-quantile bins)",
    ),
    "mmd": Detector(
        "mmd", mmd, True, "reference-median impute",
        "unbiased RBF-kernel MMD^2, median-heuristic bandwidth, 1500-row subsample",
    ),
    "c2st": Detector(
        "c2st", c2st, True, "reference-median impute",
        "classifier two-sample test, held-out discriminator AUC - 0.5",
    ),
}

DETECTOR_NAMES = list(DETECTORS)
