"""Datasets for DriftHarm: the real IEEE-CIS fraud stream and a synthetic control.

The single rule this module exists to enforce: the model is trained on rows that
never appear in any evaluation window. Every reference/current window is drawn
from a held-out pool that is strictly later in time than the training data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("/Users/salim/ieee-fraud-ml/data/raw")
ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"

# How many of the model's top features are monitored, plus how many
# near-zero-importance ones. Real monitoring watches more columns than the model
# actually leans on; the padding is what makes the irrelevant-feature archetype
# expressible at all.
N_TOP_FEATURES = 40
N_NOISE_FEATURES = 20


@dataclass
class Bundle:
    """A trained model plus a held-out pool it has never seen."""

    model: object
    pool_X: np.ndarray  # (n, n_monitored) float32, may contain NaN
    pool_y: np.ndarray  # (n,) int
    features: list[str]  # names of the monitored columns, len == pool_X.shape[1]
    importance: np.ndarray  # model gain importance per monitored column
    predict_cols: list[str]  # full ordered feature list the model expects
    predict_index: np.ndarray  # position of each monitored col inside predict_cols
    meta: dict = field(default_factory=dict)

    @property
    def top_idx(self) -> np.ndarray:
        """Indices of monitored columns the model actually relies on."""
        return np.argsort(-self.importance)[:N_TOP_FEATURES]

    @property
    def zero_idx(self) -> np.ndarray:
        """Indices of monitored columns with (near) zero gain importance."""
        return np.where(self.importance <= 0)[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Score a monitored-feature matrix.

        The model was fitted on the full column set; monitored columns are
        written into their original positions and the rest are held at the
        training median, which is what a serving system does for a column it is
        not being handed.
        """
        full = np.tile(self._base_row, (X.shape[0], 1))
        full[:, self.predict_index] = X
        # Straight to the booster: the sklearn wrapper re-checks feature names on
        # every call, which is noise here and costs real time over thousands of
        # windows. Same numbers, and for binary LightGBM this is P(y=1).
        return self.model.booster_.predict(full)

    _base_row: np.ndarray = field(default=None, repr=False)


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode non-numeric columns; leave numerics alone. Deterministic.

    Tested against dtype rather than `== object`: pandas 3 gives string columns
    the `str` dtype, and an `== object` check silently encodes nothing.
    """
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            cats = sorted(df[col].dropna().unique())
            mapping = {c: float(i) for i, c in enumerate(cats)}
            df[col] = df[col].map(mapping).astype("float32")
    return df


def load_ieee(raw_dir: Path = RAW_DIR, cache: Path | None = None) -> pd.DataFrame:
    """Merge transaction + identity, encode, sort by time. Cached as parquet."""
    cache = cache or (ARTIFACT_DIR / "ieee_encoded.parquet")
    if cache.exists():
        return pd.read_parquet(cache)

    tx = pd.read_csv(raw_dir / "train_transaction.csv")
    idt = pd.read_csv(raw_dir / "train_identity.csv")
    df = tx.merge(idt, on="TransactionID", how="left")
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    df = _encode(df)
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype("float32")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def build_real_bundle(
    train_frac: float = 0.4,
    seed: int = 0,
    raw_dir: Path = RAW_DIR,
    cache: Path | None = None,
) -> Bundle:
    """Train on the earliest `train_frac` of the stream, hold the rest out."""
    import lightgbm as lgb

    cache = cache or (ARTIFACT_DIR / "real_bundle.pkl")
    if cache.exists():
        import joblib

        return joblib.load(cache)

    df = load_ieee(raw_dir)
    drop = ["TransactionID", "isFraud", "TransactionDT"]
    feat_cols = [c for c in df.columns if c not in drop]

    cut = int(len(df) * train_frac)
    tr, pool = df.iloc[:cut], df.iloc[cut:]

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.6,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(tr[feat_cols], tr["isFraud"])

    gain = pd.Series(model.booster_.feature_importance("gain"), index=feat_cols)
    top = list(gain.sort_values(ascending=False).head(N_TOP_FEATURES).index)
    zeros = [c for c in feat_cols if gain[c] <= 0 and c not in top]
    rng = np.random.default_rng(seed)
    noise = list(rng.choice(zeros, size=min(N_NOISE_FEATURES, len(zeros)), replace=False))
    monitored = top + noise

    base_row = tr[feat_cols].median(numeric_only=True).fillna(0.0).to_numpy("float32")
    idx = np.array([feat_cols.index(c) for c in monitored])

    from sklearn.metrics import roc_auc_score

    bundle = Bundle(
        model=model,
        pool_X=pool[monitored].to_numpy("float32"),
        pool_y=pool["isFraud"].to_numpy("int8"),
        features=monitored,
        importance=gain[monitored].to_numpy("float64"),
        predict_cols=feat_cols,
        predict_index=idx,
        meta={},
        _base_row=base_row,
    )
    # Two AUCs worth recording: the model on its full feature set, and the model
    # driven only through the monitored columns (which is the harness the whole
    # benchmark runs through). If these diverge badly the harness is not
    # measuring the model anyone deployed.
    full_auc = float(roc_auc_score(pool["isFraud"], model.predict_proba(pool[feat_cols])[:, 1]))
    harness_auc = float(roc_auc_score(bundle.pool_y, bundle.predict(bundle.pool_X)))
    bundle.meta = {
        "n_train": int(cut),
        "n_pool": int(len(pool)),
        "n_features_model": len(feat_cols),
        "n_features_monitored": len(monitored),
        "n_top": len(top),
        "n_noise": len(noise),
        "pool_fraud_rate": float(pool["isFraud"].mean()),
        "train_fraud_rate": float(tr["isFraud"].mean()),
        "pool_auc_full_features": full_auc,
        "pool_auc_monitored_only": harness_auc,
        "train_dt_max": float(tr["TransactionDT"].max()),
        "pool_dt_min": float(pool["TransactionDT"].min()),
    }

    import joblib

    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, cache)
    (ARTIFACT_DIR / "real_bundle_meta.json").write_text(json.dumps(bundle.meta, indent=2))
    return bundle


def build_synthetic_bundle(
    n: int = 200_000,
    d: int = 60,
    n_informative: int = 40,
    seed: int = 0,
) -> Bundle:
    """Correlated Gaussian inputs, logistic labels, `d - n_informative` weights
    exactly zero.

    The zero weights are the point: a feature with weight 0 cannot cause harm no
    matter how far it moves, so the irrelevant-feature archetype has ground-truth
    harm of exactly zero rather than an assumed one.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    # Block-correlated covariance so the multivariate detectors have structure to find.
    A = rng.normal(size=(d, d)) * 0.15
    cov = A @ A.T + np.eye(d)
    sd = np.sqrt(np.diag(cov))
    cov = cov / np.outer(sd, sd)
    X = rng.multivariate_normal(np.zeros(d), cov, size=n).astype("float32")

    w = np.zeros(d)
    w[:n_informative] = rng.normal(0, 1.0, size=n_informative)
    logit = X @ w * (1.6 / np.sqrt(n_informative)) - 2.5  # ~7% positive rate
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype("int8")

    cut = int(n * 0.4)
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    model.fit(X[:cut], y[:cut])
    gain = model.booster_.feature_importance("gain").astype("float64")
    # Ground truth, not estimated: features with w == 0 are irrelevant by construction.
    gain[w == 0] = 0.0

    bundle = Bundle(
        model=model,
        pool_X=X[cut:],
        pool_y=y[cut:],
        features=[f"x{i}" for i in range(d)],
        importance=gain,
        predict_cols=[f"x{i}" for i in range(d)],
        predict_index=np.arange(d),
        _base_row=np.zeros(d, dtype="float32"),
    )
    bundle.meta = {
        "n_pool": int(n - cut),
        "n_train": int(cut),
        "d": d,
        "n_informative": n_informative,
        "n_irrelevant": int((w == 0).sum()),
        "pool_positive_rate": float(y[cut:].mean()),
        "pool_auc": float(roc_auc_score(y[cut:], bundle.predict(X[cut:]))),
        "true_weights_zero": [int(i) for i in np.where(w == 0)[0]],
    }
    return bundle
