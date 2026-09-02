"""Recompute every metric in reports/real_ranking.csv from the confusion matrix.

The ranking table has tp/fp/fn/tn per detector, plus six derived columns:
MCC, F1, precision, recall, specificity, and balanced accuracy. The C verifier
already checks MCC from the trial level data. This checks all six derived
metrics from the published confusion matrix itself, catching any broken formula
in the table generation step (experiments/03_tables.py).

It also verifies the ranking order: rows must be sorted by MCC descending, and
the published detector order must match a fresh sort.

    python verify/metrics.py [repo root]
"""
import csv
import math
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..")
TOL = 1e-12

with open(os.path.join(root, "reports", "real_ranking.csv")) as f:
    rows = list(csv.DictReader(f))

failures = 0

def check(label, got, want):
    global failures
    gap = abs(got - want)
    ok = gap <= TOL
    if not ok:
        failures += 1
    tag = "ok" if ok else "FAIL"
    print(f"  {label:<40s} got {got:+.15f}  gap {gap:.1e}  {tag}")

print("derived metrics, recomputed from tp/fp/fn/tn")
for r in rows:
    tp = int(r["tp"])
    fp = int(r["fp"])
    fn = int(r["fn"])
    tn = int(r["tn"])
    d = r["detector"]

    # MCC
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    check(f"{d}/mcc", mcc, float(r["mcc"]))

    # precision = tp / (tp + fp)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    check(f"{d}/precision", prec, float(r["harm_precision"]))

    # recall = tp / (tp + fn)
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    check(f"{d}/recall", rec, float(r["harm_recall"]))

    # f1 = 2 * prec * rec / (prec + rec)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    check(f"{d}/f1", f1, float(r["harm_f1"]))

    # specificity = tn / (tn + fp)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    check(f"{d}/specificity", spec, float(r["specificity"]))

    # balanced accuracy = (recall + specificity) / 2
    ba = (rec + spec) / 2
    check(f"{d}/balanced_accuracy", ba, float(r["balanced_accuracy"]))

# Ranking order: rows must be sorted by MCC descending
print("\nranking order")
mccs = [float(r["mcc"]) for r in rows]
if mccs == sorted(mccs, reverse=True):
    print("  rows sorted by MCC descending: ok")
else:
    print("  rows NOT sorted by MCC descending: FAIL")
    failures += 1

if failures > 0:
    print(f"\n{failures} checks failed")
    sys.exit(1)
print(f"\nPython reproduces all {len(rows) * 6} derived metrics and the ranking order")
