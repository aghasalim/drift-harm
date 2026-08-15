"""Render the result CSVs as markdown tables.

Everything in the README is generated here rather than typed, so a number in the
prose cannot drift away from the number in reports/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DETS = ["ks", "psi", "wasserstein", "jensen_shannon", "mmd", "c2st"]
PRETTY = {
    "ks": "KS", "psi": "PSI", "wasserstein": "Wasserstein",
    "jensen_shannon": "Jensen-Shannon", "mmd": "MMD", "c2st": "C2ST",
}


def md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in d.values)
    return "\n".join([head, sep, body])


def section(which: str) -> str:
    out = [f"## {which}\n"]

    rank = pd.read_csv(REPORTS / f"{which}_ranking.csv")
    rank["detector"] = rank["detector"].map(PRETTY)
    rank = rank.rename(columns={
        "detector": "detector", "mcc": "MCC", "harm_f1": "harm-F1",
        "harm_precision": "precision", "harm_recall": "recall",
        "specificity": "specificity", "balanced_accuracy": "bal.acc",
    })
    out.append("### Harm-alignment ranking\n")
    out.append(md(rank[["detector", "MCC", "harm-F1", "precision", "recall",
                        "specificity", "tp", "fp", "fn", "tn"]]))

    nul = pd.read_csv(REPORTS / f"{which}_null_summary.csv")
    nul["detector"] = nul["detector"].map(PRETTY)
    out.append("\n### Measured null and calibrated thresholds\n")
    out.append(md(nul[["detector", "null_mean", "null_sd", "null_p95",
                       "threshold", "realised_far"]], "{:.5f}"))

    arch = pd.read_csv(REPORTS / f"{which}_by_archetype.csv")
    arch = arch.rename(columns={c: PRETTY.get(c, c) for c in arch.columns})
    out.append("\n### Alarm rate by archetype (fraction of replicates alarming)\n")
    cols = ["archetype", "expected_harm", "measured_harm_rate", "mean_auc_drop"] + \
           [PRETTY[d] for d in DETS]
    out.append(md(arch[cols]))

    grad = pd.read_csv(REPORTS / f"{which}_gradual_summary.csv")
    grad = grad.rename(columns={f"alarm_{d}": PRETTY[d] for d in DETS})
    out.append("\n### Gradual vs sudden (alarm rate per batch)\n")
    out.append(md(grad[["mode", "batch", "auc_drop", "harm"] + [PRETTY[d] for d in DETS]]))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    parts = ["# Generated result tables\n",
             "Regenerate with `python experiments/03_tables.py`.\n"]
    for which in ("real", "synthetic"):
        if (REPORTS / f"{which}_ranking.csv").exists():
            parts.append(section(which))
    text = "\n".join(parts)
    (REPORTS / "tables.md").write_text(text)
    print(text)
