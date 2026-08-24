"""Draw the two figures the README leads with, from the saved report CSVs.

Free to run: reads ``reports/``, simulates nothing.  Both figures are arguments
the tables already make, redrawn so the shape is visible at a glance -- the first
that the detector order depends entirely on how you resample, the second that the
failures are specific and mechanical rather than a matter of degree.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = ROOT / "reports" / "figures"

DETECTORS = ["ks", "psi", "wasserstein", "jensen_shannon", "mmd", "c2st"]
SCHEMES = [
    ("iid_trial", "resample trials\n(the usual choice)"),
    ("stratified_by_archetype", "stratify by archetype"),
    ("cluster_by_archetype", "resample archetypes\n(the honest choice)"),
]


def ranking_stability(out: Path) -> Path:
    """Show the same point estimates under three resampling schemes.

    The point estimates never move.  Only the uncertainty does, and under
    archetype resampling it grows until every interval covers zero and the
    ordering carries no information.
    """
    table = pd.read_csv(REPORTS / "real_rank_stability.csv")
    order = (
        table[table.scheme == "iid_trial"]
        .sort_values("mcc", ascending=True)
        .detector.tolist()
    )

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharex=True, sharey=True)
    for ax, (scheme, label) in zip(axes, SCHEMES, strict=True):
        rows = table[table.scheme == scheme].set_index("detector").loc[order]
        y = np.arange(len(order))
        # One call per detector: matplotlib's ecolor takes a single colour, and the
        # whole point here is that some intervals cover zero and others do not.
        for index, (_, row) in enumerate(rows.iterrows()):
            covers_zero = row.mcc_lo95 < 0 < row.mcc_hi95
            ax.errorbar(
                row.mcc,
                index,
                xerr=[[row.mcc - row.mcc_lo95], [row.mcc_hi95 - row.mcc]],
                fmt="o",
                ecolor="#b2182b" if covers_zero else "#1a9850",
                elinewidth=2.2,
                capsize=4,
                color="0.15",
                markersize=5,
            )
        ax.axvline(0, color="0.4", lw=0.9, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.set_title(
            f"{label}\nmean CI width {rows.ci_width.mean():.2f}", fontsize=10
        )
        ax.set_xlabel("MCC against harm")
        ax.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "Same six detectors, same 240 trials, same point estimates. "
        "Red intervals cover zero.",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def archetype_breakdown(out: Path) -> Path:
    """Contrast measured harm per failure mode against what each detector fired on.

    Reading down the two blocks is the whole argument: the archetypes where harm
    is certain are not the archetypes where the detectors alarm.
    """
    table = pd.read_csv(REPORTS / "real_by_archetype.csv").set_index("archetype")
    harm = table["measured_harm_rate"]
    alarms = table[DETECTORS]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(12.5, 5.6), gridspec_kw={"width_ratios": [1, 4.2]}, sharey=True
    )

    left.barh(
        np.arange(len(harm)),
        harm.values,
        color=["#b2182b" if h > 0.5 else "#9ecae1" for h in harm.values],
    )
    left.set_yticks(np.arange(len(harm)))
    left.set_yticklabels(harm.index, fontsize=8)
    left.invert_yaxis()
    left.set_xlim(0, 1)
    left.set_xlabel("measured harm rate")
    left.set_title("did the model actually get worse?", fontsize=10)
    left.spines[["top", "right"]].set_visible(False)

    image = right.imshow(alarms.values, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    right.set_xticks(np.arange(len(DETECTORS)))
    right.set_xticklabels(DETECTORS, rotation=30, ha="right", fontsize=9)
    right.set_title("did the detector alarm?", fontsize=10)
    for i in range(alarms.shape[0]):
        for j in range(alarms.shape[1]):
            value = alarms.values[i, j]
            missed = harm.values[i] > 0.5 and value < 0.5
            false_alarm = harm.values[i] < 0.5 and value > 0.5
            if missed or false_alarm:
                right.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False,
                        edgecolor="#b2182b" if missed else "#f46d43",
                        lw=2.2,
                    )
                )
            right.text(
                j, i, f"{value:.2f}",
                ha="center", va="center", fontsize=7.5,
                color="white" if value > 0.55 else "0.2",
            )
    figure.colorbar(image, ax=right, fraction=0.02, pad=0.02, label="alarm rate")
    figure.suptitle(
        "Red box: harm happened and the detector stayed quiet.   "
        "Orange box: it alarmed on a window that did no harm.",
        fontsize=10,
        y=0.02,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (
        ranking_stability(FIGURES / "ranking-stability.png"),
        archetype_breakdown(FIGURES / "archetype-breakdown.png"),
    ):
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
