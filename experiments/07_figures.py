"""Draw the README figures from the saved report CSVs.

Free to run: reads ``reports/``, simulates nothing. Every figure here is an
argument one of the tables already makes, redrawn so the shape is visible at a
glance. Because nothing is sampled, a figure cannot disagree with a number
quoted in the prose.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

from style import PALETTE, titled

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = ROOT / "reports" / "figures"

DETECTORS = ["ks", "psi", "wasserstein", "jensen_shannon", "mmd", "c2st"]

# Red for the bad case and green for the good one, because that is how the
# README talks about them: red intervals cover zero, red boxes are harm the
# detector slept through, red bars are harm. Everything arbitrary takes a
# colour from the shared palette.
BAD, GOOD, WARN = "#b2182b", "#1a9850", "#d9822b"
QUIET = "#9ecae1"

# One colour per detector, fixed across every figure and the animation, so a
# reader who learns MMD's colour once keeps it.
DET_COLOUR = dict(zip(DETECTORS, PALETTE, strict=True))

# The CSVs carry the identifiers the code uses. A figure is read by a person, so
# it prints these instead.
DET_LABEL = {
    "ks": "KS test",
    "psi": "PSI",
    "wasserstein": "Wasserstein",
    "jensen_shannon": "Jensen-Shannon",
    "mmd": "MMD",
    "c2st": "C2ST",
}
ARCHETYPE_LABEL = {
    "true_null": "true null",
    "covariate_shift_mild": "mild covariate shift",
    "covariate_shift_moderate": "moderate covariate shift",
    "covariate_shift_strong": "strong covariate shift",
    "concept_drift_no_covariate_shift": "concept drift, no covariate shift",
    "imputation_masked_null": "masked-null imputation",
    "imputation_visible": "visible imputation",
    "dilution_shift": "dilution shift",
    "dilution_permuted": "permuted dilution",
    "irrelevant_feature_drift": "irrelevant feature drift",
    "sudden_shift": "sudden shift",
    "gradual_shift": "gradual shift",
}

SCHEMES = [
    ("iid_trial", "resample trials"),
    ("stratified_by_archetype", "stratify by archetype"),
    ("cluster_by_archetype", "resample archetypes"),
]
# Claim per panel. The CI width is read from the CSV and appended, never typed.
SCHEME_CLAIM = {
    "iid_trial": ("Resample trials: MMD looks best",
                  "240 trials treated as independent"),
    "stratified_by_archetype": ("Stratify: the bars get tighter",
                                "resampled inside each archetype"),
    "cluster_by_archetype": ("Resample archetypes: no ranking left",
                             "the twelve modes are the population"),
}

MCC_AXIS = "Matthews correlation with harm (unitless, -1 to 1)"
SPEARMAN_AXIS = "Spearman correlation between two six-detector orders (unitless, -1 to 1)"


def ranking_stability(out: Path) -> Path:
    """Show the same point estimates under three resampling schemes.

    The point estimates never move. Only the uncertainty does, and under
    archetype resampling it grows until every interval covers zero and the
    ordering carries no information.
    """
    table = pd.read_csv(REPORTS / "real_rank_stability.csv")
    order = (
        table[table.scheme == "iid_trial"]
        .sort_values("mcc", ascending=True)
        .detector.tolist()
    )

    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), sharex=True, sharey=True)
    for ax, (scheme, _) in zip(axes, SCHEMES, strict=True):
        rows = table[table.scheme == scheme].set_index("detector").loc[order]
        # One call per detector: matplotlib's ecolor takes a single colour, and the
        # whole point here is that some intervals cover zero and others do not.
        for index, (_, row) in enumerate(rows.iterrows()):
            covers_zero = row.mcc_lo95 < 0 < row.mcc_hi95
            ax.errorbar(
                row.mcc,
                index,
                xerr=[[row.mcc - row.mcc_lo95], [row.mcc_hi95 - row.mcc]],
                fmt="o",
                ecolor=BAD if covers_zero else GOOD,
                elinewidth=2.4,
                capsize=4,
                color="0.15",
                markersize=5,
            )
        ax.axvline(0, color="0.45", lw=0.9, ls="--", zorder=0)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(order)
        # Room under the bottom detector for the legend, which otherwise lands
        # on psi's interval.
        ax.set_ylim(-1.7, len(order) - 0.4)
        ax.set_xlabel(MCC_AXIS)
        claim, setup = SCHEME_CLAIM[scheme]
        titled(ax, claim, f"{setup}, CI width {rows.ci_width.mean():.2f}")

    axes[0].legend(
        handles=[
            Line2D([], [], color=GOOD, lw=2.4, label="95% interval excludes zero"),
            Line2D([], [], color=BAD, lw=2.4, label="95% interval covers zero"),
        ],
        loc="lower left",
    )
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def archetype_breakdown(out: Path) -> Path:
    """Contrast measured harm per failure mode against what each detector fired on.

    Reading across the two blocks is the whole argument: the archetypes where
    harm is certain are not the archetypes where the detectors alarm.
    """
    table = pd.read_csv(REPORTS / "real_by_archetype.csv").set_index("archetype")
    harm = table["measured_harm_rate"]
    alarms = table[DETECTORS]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13.0, 6.2), gridspec_kw={"width_ratios": [1, 3.6]}, sharey=True
    )

    left.barh(
        np.arange(len(harm)),
        harm.values,
        color=[BAD if h > 0.5 else QUIET for h in harm.values],
    )
    left.set_yticks(np.arange(len(harm)))
    left.set_yticklabels(harm.index, fontsize=8.5)
    left.invert_yaxis()
    left.set_xlim(0, 1)
    left.set_xticks([0, 0.5, 1.0])
    left.set_xlabel("measured harm rate\n(share of 20 replicates)")
    left.grid(False, axis="y")
    titled(left, "Half of them hurt",
           "measured, not assumed")

    # vmax above 1 so a saturated cell is dark grey rather than pure black: the
    # printed number and the red boxes have to stay readable on top of it.
    image = right.imshow(alarms.values, cmap="Greys", vmin=0, vmax=1.45, aspect="auto")
    right.set_xticks(np.arange(len(DETECTORS)))
    right.set_xticklabels(DETECTORS, rotation=20, ha="right")
    right.grid(False)
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
                        edgecolor=BAD if missed else WARN,
                        lw=2.4,
                    )
                )
            right.text(
                j, i, f"{value:.2f}",
                ha="center", va="center", fontsize=8,
                color="white" if value > 0.55 else "0.2",
            )
    bar = figure.colorbar(image, ax=right, fraction=0.025, pad=0.02,
                          ticks=[0, 0.5, 1.0])
    bar.set_label("alarm rate (share of 20 replicates)", fontsize=9)
    bar.outline.set_visible(False)
    # vmax runs past 1 to keep the cells readable; the bar should still stop
    # at the largest rate that exists.
    bar.ax.set_ylim(0, 1)
    right.tick_params(axis="y", length=0)

    titled(right, "The alarms do not line up with the harm",
           "share of 20 replicates in which the detector fired, thresholds calibrated to a 5% false-alarm rate")
    right.legend(
        handles=[
            Patch(facecolor="none", edgecolor=BAD, lw=2.4,
                  label="harm happened and the detector stayed quiet"),
            Patch(facecolor="none", edgecolor=WARN, lw=2.4,
                  label="it alarmed on a window that did no harm"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
    )
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def calibration_size(out: Path) -> Path:
    """How many null replicates a threshold needs before it means anything.

    Thresholds are set to hit a 5% false-alarm rate. With few calibration
    replicates the realised rate lands nowhere near it, and the p90 shows the
    tail is far worse than the mean suggests.
    """
    table = pd.read_csv(REPORTS / "real_calibration_size.csv")
    target = float(table.target_alpha.iloc[0])
    sizes = sorted(table.n_calibration_reps.unique())

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.0, 4.8),
                                         sharex=True, sharey=True)
    panels = (
        (left, "mean_far",
         "At 20 nulls the rate is double the target",
         "mean over 400 resplits of the 300 saved null replicates"),
        (right, "p90_far",
         "The tail is worse, and stays worse",
         "90th percentile of the same resplits, stepping in units of 1%"),
    )
    for ax, column, claim, setup in panels:
        for detector in DETECTORS:
            rows = table[table.detector == detector].sort_values("n_calibration_reps")
            ax.plot(rows.n_calibration_reps, rows[column], "o-", lw=1.6,
                    markersize=4.5, color=DET_COLOUR[detector], label=detector)
        ax.axhline(target, color="0.3", ls="--", lw=1.2, zorder=0)
        # Only five sizes were swept, so name them instead of letting the log
        # locator print 2x10^1.
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.minorticks_off()
        ax.set_xlim(sizes[0] * 0.88, sizes[-1] * 1.14)
        ax.set_xlabel("null replicates used to set the threshold (count)")
        titled(ax, claim, setup)

    left.set_ylabel("realised false-alarm rate\n(share of held-out null replicates)")
    left.set_ylim(0.03, 0.205)
    left.text(sizes[0] * 0.92, target + 0.002, f"target {target:.0%}", va="bottom",
              ha="left", fontsize=9, color="0.3")
    # Every curve falls left to right, so the top right corner of the left panel
    # is the one place a legend cannot land on a line.
    left.legend(loc="upper right", ncol=2)

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def gradual_drift(out: Path) -> Path:
    """Harm accumulating batch by batch against when each detector first fires.

    Five of the six sit at exactly 1.0 from batch 1 onwards. Drawing six lines
    hides five of them under the sixth, so they are drawn once as one line and
    named in the legend.
    """
    table = pd.read_csv(REPORTS / "real_gradual_summary.csv")
    table = table[table["mode"] == "gradual"].sort_values("batch")
    saturated = [d for d in DETECTORS if (table[f"alarm_{d}"] == 1.0).all()]
    moving = [d for d in DETECTORS if d not in saturated]

    figure, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.plot(table.batch, table.auc_drop, "o-", color=BAD, lw=2.6,
            label="AUC drop against the reference window (harm)", zorder=3)
    ax.set_xlabel("batch along the gradual drift (count)")
    ax.set_ylabel("AUC drop (AUC points)", color=BAD)
    ax.tick_params(axis="y", labelcolor=BAD)
    ax.set_ylim(0.06, 0.155)

    twin = ax.twinx()
    twin.grid(False)
    # Wider than the MMD line so it still reads once MMD reaches 1.0 too.
    twin.plot(table.batch, [1.0] * len(table), lw=4.0, color="0.6",
              label=f"{', '.join(saturated)}: all five pinned at 1.0", zorder=2)
    for detector in moving:
        twin.plot(table.batch, table[f"alarm_{detector}"], "o-", lw=2.2,
                  color=DET_COLOUR[detector], label=detector, zorder=4)
    twin.set_ylabel("alarm rate (share of 6 replicates)")
    twin.set_ylim(-0.06, 1.12)
    twin.spines[["top"]].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    extra_h, extra_l = twin.get_legend_handles_labels()
    # Both the harm curve and MMD climb left to right, so the bottom right
    # corner is empty for every batch after the third.
    twin.legend(handles + extra_h, labels + extra_l, loc="lower right",
                bbox_to_anchor=(0.99, 0.02))
    titled(ax, "MMD is the only detector that tracks the damage as it accrues",
           "one gradual drift over 8 batches, 6 replicates each; the other five have already fired at batch 1")
    figure.tight_layout()
    figure.savefig(out)
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
    full_winner = pd.read_csv(REPORTS / "real_ranking.csv").detector.iloc[0]

    figure, ax = plt.subplots(figsize=(10.5, 5.6))
    colours = [BAD if v < 0.9 else QUIET for v in table.spearman_vs_full_suite]
    ax.barh(range(len(table)), table.spearman_vs_full_suite, color=colours)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table.dropped_archetype)
    ax.axvline(1.0, color="0.3", lw=1.0, ls="--", zorder=0)
    ax.axvline(0.0, color="0.6", lw=0.9, zorder=0)
    ax.set_xlabel(SPEARMAN_AXIS)
    # The masked-null drop is negative, so the axis has to reach past zero or
    # its bar is invisible and its label floats outside the frame.
    ax.set_xlim(-0.62, 1.18)
    ax.set_ylim(-0.8, len(table) - 0.2)
    ax.grid(False, axis="y")
    for index, (_, row) in enumerate(table.iterrows()):
        if row.winner == full_winner:
            continue
        # A negative bar runs left from zero, so its label still goes on the
        # right of the origin or it collides with the archetype name.
        value = max(row.spearman_vs_full_suite, 0.0)
        ax.text(value + 0.02, index, f"winner becomes {row.winner}",
                va="center", ha="left", fontsize=9, color="0.25")
    ax.legend(
        handles=[
            Patch(color=QUIET, label="order essentially holds"),
            Patch(color=BAD, label="order moves (Spearman below 0.9)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
    )
    titled(ax, "Dropping one failure mode can flip the winner outright",
           f"order on the full 12-archetype suite (winner {full_winner}) against the order with that archetype removed; "
           "a stable ranking would sit flat on the dashed line")
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def dataset_agreement(out: Path) -> Path:
    """Real-vs-synthetic disagreement against what one dataset does to itself."""
    table = pd.read_csv(REPORTS / "ranking_agreement.csv")
    real = table[table.dataset == "real"].set_index("scheme")
    labels = [label for _, label in SCHEMES]
    rows = real.loc[[scheme for scheme, _ in SCHEMES]]

    observed = float(rows.observed_cross_dataset_spearman.iloc[0])
    # The p values live in their own right-hand column so they never sit on a
    # bar or run off the frame.
    column = 1.42

    figure, ax = plt.subplots(figsize=(11.0, 4.8))
    for index, (_, row) in enumerate(rows.iterrows()):
        ax.plot([row.self_spearman_p05, row.self_spearman_p50], [index, index],
                color="#8c8c8c", lw=10, solid_capstyle="butt", alpha=0.75, zorder=2,
                label="p05 to median of the self-comparison" if index == 0 else None)
        ax.plot(row.self_spearman_mean, index, "o", color=PALETTE[0], markersize=8,
                zorder=3, label="mean self-agreement" if index == 0 else None)
        ax.text(column, index, f"{row.p_self_spearman_at_or_below_observed:.3f}",
                fontsize=10, color="0.25", va="center", ha="right")
    ax.axvline(observed, color=BAD, lw=1.8, ls="--", zorder=1,
               label=f"real vs synthetic: {observed:.2f}")
    ax.text(column, len(rows) - 0.35, f"share of self-comparisons\nat or below {observed:.2f}",
            fontsize=9, color="0.45", va="center", ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.55, len(rows) - 0.05)
    ax.set_xlim(-0.75, column + 0.03)
    ax.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(SPEARMAN_AXIS)
    ax.grid(False, axis="y")
    # The reference line spans the full height, so the legend goes under the
    # axes rather than across it.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3)
    titled(ax, "Resample archetypes and the datasets agree as well as one agrees with itself",
           "2,000 bootstrap draws per scheme, real suite re-ranked against an independent resample of itself")
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def _shrink_gif(path: Path) -> Path:
    """Rewrite every frame onto one shared palette. Roughly halves the file."""
    # 160 rather than 64: the frame is mostly white, so a small palette spends
    # its boxes on the background and drags the six detector colours off the
    # ones the PNGs use. At 160 every palette colour comes back exact.
    src = Image.open(path)
    frames, durations = [], []
    try:
        while True:
            frames.append(src.convert("RGB"))
            durations.append(src.info.get("duration", 62))
            src.seek(src.tell() + 1)
    except EOFError:
        pass
    shared = frames[len(frames) // 2].quantize(160, method=Image.Quantize.MEDIANCUT)
    quantised = [f.quantize(palette=shared, dither=Image.Dither.NONE) for f in frames]
    quantised[0].save(path, save_all=True, append_images=quantised[1:], loop=0,
                      duration=durations, optimize=True)
    return path


def anim_ranking_race(out: Path, move: int = 5, hold: int = 5, fps: int = 18) -> Path:
    """Replay the leave-one-archetype-out sweep as the order re-forming.

    Reads real_ranking.csv for the full-suite estimate and
    real_leave_one_archetype_out.csv for the twelve drops. Nothing is resampled
    or simulated here, so the GIF is the same on every run.
    """
    full = pd.read_csv(REPORTS / "real_ranking.csv").set_index("detector").mcc
    drops = pd.read_csv(REPORTS / "real_leave_one_archetype_out.csv")

    captions = ["nothing dropped: all 12 archetypes"]
    values = [full[DETECTORS].to_numpy(dtype=float)]
    spearman = [1.0]
    for _, row in drops.iterrows():
        captions.append(f"dropped: {ARCHETYPE_LABEL[row.dropped_archetype]}")
        values.append(row[DETECTORS].to_numpy(dtype=float))
        spearman.append(float(row.spearman_vs_full_suite))
    values = np.asarray(values)

    # Rank 0 at the top. argsort twice turns a value into its position.
    ranks = np.argsort(np.argsort(-values, axis=1), axis=1).astype(float)

    figure, ax = plt.subplots(figsize=(11.0, 5.0))
    # Limits from the values that are actually drawn. The right pad is the room
    # the detector name needs beside its dot, not empty canvas.
    span = float(values.max() - values.min())
    ax.set_xlim(values.min() - 0.05 * span, values.max() + 0.18 * span)
    # Headroom above rank 1 for the running caption, which otherwise lands on
    # the subtitle.
    ax.set_ylim(len(DETECTORS) - 0.45, -1.7)
    ax.set_xlabel(MCC_AXIS)
    ax.set_ylabel("rank by MCC (1 = best of the six)")
    ax.set_yticks(range(len(DETECTORS)))
    ax.set_yticklabels([str(i + 1) for i in range(len(DETECTORS))])
    ax.grid(False, axis="y")
    ax.axvline(0, color="0.6", lw=0.9, ls="--", zorder=0)
    titled(ax, "Drop one failure mode and the order re-forms",
           "the same 240 trials, re-ranked with one archetype held out at a time")

    dots, tags = {}, {}
    for detector in DETECTORS:
        dots[detector] = ax.plot([], [], "o", markersize=9,
                                 color=DET_COLOUR[detector], zorder=3)[0]
        tags[detector] = ax.text(0, 0, DET_LABEL[detector], fontsize=9.5,
                                 va="center", ha="left",
                                 color=DET_COLOUR[detector])
    caption = ax.text(0.01, -1.2, "", fontsize=9.5, color="#333333",
                      va="center", ha="left", transform=ax.get_yaxis_transform())
    figure.tight_layout()
    pad = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.012

    def draw(frame):
        step = frame // (move + hold)
        within = frame % (move + hold)
        start, end = max(step - 1, 0), step
        t = min(within / move, 1.0) if move else 1.0
        ease = t * t * (3 - 2 * t)
        x = values[start] + (values[end] - values[start]) * ease
        y = ranks[start] + (ranks[end] - ranks[start]) * ease
        for index, detector in enumerate(DETECTORS):
            dots[detector].set_data([x[index]], [y[index]])
            tags[detector].set_position((x[index] + pad, y[index]))
        text = captions[end]
        if end:
            text += f"    Spearman against the full order: {spearman[end]:+.2f}"
        caption.set_text(text)
        return list(dots.values()) + list(tags.values()) + [caption]

    frames = len(values) * (move + hold)
    anim = FuncAnimation(figure, draw, frames=frames, interval=1000 // fps, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(figure)
    return _shrink_gif(out)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (
        ranking_stability(FIGURES / "ranking-stability.png"),
        archetype_breakdown(FIGURES / "archetype-breakdown.png"),
        calibration_size(FIGURES / "calibration-size.png"),
        gradual_drift(FIGURES / "gradual-drift.png"),
        leave_one_out(FIGURES / "leave-one-archetype-out.png"),
        dataset_agreement(FIGURES / "dataset-agreement.png"),
        anim_ranking_race(FIGURES / "ranking-race.gif"),
    ):
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
