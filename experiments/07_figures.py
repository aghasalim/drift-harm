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


def calibration_size(out: Path) -> Path:
    """How many null replicates a threshold needs before it means anything.

    Thresholds are set to hit a 5% false-alarm rate. With few calibration
    replicates the realised rate lands nowhere near it, and the p90 shows the tail
    is far worse than the mean suggests.
    """
    table = pd.read_csv(REPORTS / "real_calibration_size.csv")
    target = float(table.target_alpha.iloc[0])

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.3), sharex=True, sharey=True)
    for ax, column, title in (
        (left, "mean_far", "mean false-alarm rate"),
        (right, "p90_far", "90th percentile"),
    ):
        for detector in DETECTORS:
            rows = table[table.detector == detector].sort_values("n_calibration_reps")
            ax.plot(rows.n_calibration_reps, rows[column], "o-", lw=1.6,
                    markersize=4, label=detector)
        ax.axhline(target, color="#b2182b", ls="--", lw=1.4)
        ax.text(table.n_calibration_reps.max(), target, f"  target {target:.0%}",
                va="center", fontsize=8, color="#b2182b")
        ax.set_xscale("log")
        ax.set_xlabel("null replicates used to set the threshold")
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    left.set_ylabel("realised false-alarm rate")
    left.legend(frameon=False, fontsize=8, ncol=2)

    figure.suptitle(
        "A threshold calibrated on too few nulls does not deliver the rate it "
        "promises, and the tail misses by more than the mean.",
        fontsize=10, y=0.02,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def gradual_drift(out: Path) -> Path:
    """Harm accumulating batch by batch against when each detector first fires."""
    table = pd.read_csv(REPORTS / "real_gradual_summary.csv")
    table = table[table["mode"] == "gradual"].sort_values("batch")

    figure, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(table.batch, table.auc_drop, "o-", color="#b2182b", lw=2.4,
            label="AUC drop (harm)", zorder=3)
    ax.set_xlabel("batch along the gradual drift")
    ax.set_ylabel("AUC drop", color="#b2182b")
    ax.tick_params(axis="y", labelcolor="#b2182b")
    ax.spines[["top"]].set_visible(False)

    twin = ax.twinx()
    for detector in DETECTORS:
        column = f"alarm_{detector}"
        if column in table:
            twin.plot(table.batch, table[column], lw=1.4, alpha=0.85, label=detector)
    twin.set_ylabel("alarm rate")
    twin.set_ylim(-0.05, 1.05)
    twin.spines[["top"]].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, frameon=False, fontsize=8, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.set_title(
        "Gradual drift: MMD is the detector that sleeps through it, and MMD is "
        "the one that tops the table.",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def leave_one_out(out: Path) -> Path:
    """Rank correlation with the full suite when one archetype is removed.

    If the ordering were a property of the detectors it would survive dropping
    one failure mode. Several drops move it a long way, which is the same
    instability the headline figure shows from the resampling side.
    """
    table = pd.read_csv(REPORTS / "real_leave_one_archetype_out.csv")
    table = table.sort_values("spearman_vs_full_suite")

    figure, ax = plt.subplots(figsize=(9.5, 5.2))
    colours = ["#b2182b" if v < 0.9 else "#9ecae1" for v in table.spearman_vs_full_suite]
    ax.barh(range(len(table)), table.spearman_vs_full_suite, color=colours)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table.dropped_archetype, fontsize=8)
    ax.axvline(1.0, color="0.3", lw=1.0, ls="--")
    ax.set_xlabel("Spearman correlation with the full-suite ranking")
    ax.set_xlim(0, 1.05)
    for index, (_, row) in enumerate(table.iterrows()):
        ax.text(row.spearman_vs_full_suite + 0.01, index, f"winner: {row.winner}",
                va="center", fontsize=7.5, color="0.35")
    ax.set_title(
        "Drop one archetype, re-rank. A ranking that survived this would sit "
        "flat against the dashed line.",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def dataset_agreement(out: Path) -> Path:
    """Real-vs-synthetic disagreement against what one dataset does to itself."""
    table = pd.read_csv(REPORTS / "ranking_agreement.csv")
    real = table[table.dataset == "real"]

    figure, ax = plt.subplots(figsize=(9.5, 4.3))
    positions = range(len(real))
    for index, (_, row) in enumerate(real.iterrows()):
        ax.plot([row.self_spearman_p05, row.self_spearman_p50], [index, index],
                color="#4d4d4d", lw=6, solid_capstyle="butt", alpha=0.65)
        ax.plot(row.self_spearman_mean, index, "o", color="#2166ac", markersize=8,
                zorder=3)
    observed = real.observed_cross_dataset_spearman.iloc[0]
    ax.axvline(observed, color="#b2182b", lw=1.8, ls="--")
    ax.text(observed, len(real) - 0.35, f"  real vs synthetic: {observed:.2f}",
            fontsize=9, color="#b2182b", va="top")
    ax.set_yticks(list(positions))
    ax.set_yticklabels(real.scheme)
    ax.set_xlabel("Spearman correlation between two rankings")
    ax.set_title(
        "Grey bar: p05 to median of the real suite re-ranked against itself.\n"
        "Under archetype clustering the cross-dataset disagreement is inside it.",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (
        ranking_stability(FIGURES / "ranking-stability.png"),
        archetype_breakdown(FIGURES / "archetype-breakdown.png"),
        calibration_size(FIGURES / "calibration-size.png"),
        gradual_drift(FIGURES / "gradual-drift.png"),
        leave_one_out(FIGURES / "leave-one-archetype-out.png"),
        dataset_agreement(FIGURES / "dataset-agreement.png"),
    ):
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
