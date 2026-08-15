"""Build the IEEE-CIS bundle: encode, split by time, train, cache.

The pre-existing model in ../ieee-fraud-ml was fitted on all 590,540 rows, so it
is in-sample everywhere and its AUC cannot degrade. Retrained here on the
earliest 40% of the stream by TransactionDT; everything after that is held out.
"""

import json
import sys
from pathlib import Path

from driftharm.data import build_real_bundle

if __name__ == "__main__":
    b = build_real_bundle(train_frac=0.4, seed=0)
    print(json.dumps(b.meta, indent=2))
    print("monitored features:", b.features)
    print("zero-importance monitored columns:", len(b.zero_idx))
    out = Path(__file__).resolve().parents[1] / "reports"
    out.mkdir(exist_ok=True)
    (out / "real_bundle_meta.json").write_text(
        json.dumps({**b.meta, "monitored_features": b.features,
                    "n_zero_importance_monitored": int(len(b.zero_idx))}, indent=2)
    )
    sys.stdout.flush()
