import numpy as np
import pytest

from driftharm.data import build_synthetic_bundle


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(12345)


@pytest.fixture(scope="session")
def small_bundle():
    """Tiny synthetic bundle: fast, and the irrelevant features are irrelevant
    by construction rather than by estimate."""
    return build_synthetic_bundle(n=12_000, d=12, n_informative=8, seed=7)


@pytest.fixture(scope="session")
def ctx(small_bundle):
    b = small_bundle
    ref = b.pool_X[:2000]
    return {
        "importance": b.importance,
        "zero_idx": b.zero_idx,
        "ref_median": np.nanmedian(ref, axis=0),
        "scores": b.predict(b.pool_X[2000:4000]),
    }


@pytest.fixture(scope="session")
def pair(small_bundle):
    b = small_bundle
    return b.pool_X[:2000], b.pool_y[:2000], b.pool_X[2000:4000], b.pool_y[2000:4000]
