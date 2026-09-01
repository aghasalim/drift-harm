# Independent statistical check of the claim the README is built on.
#
# The repository's argument is that ranking detectors by MCC looks settled under
# trial level resampling and is not settled once archetypes are resampled as
# clusters. That rests entirely on the bootstrap in experiments/06_ranking_stability.py.
# This redoes it in base R, with R's own generator, and checks two things:
#
#   deterministic  the point MCC per detector, which must match exactly
#   stochastic     the interval widths, which are Monte Carlo estimates and can
#                  only be required to agree within their own sampling error
#
# No packages, so CI needs nothing beyond the R that is already on the runner.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."
set.seed(20260901)

DRAWS <- 4000
POINT_TOL <- 1e-12
WIDTH_REL_TOL <- 0.20   # generous: 4000 draws here against 2000 in the Python

trials <- read.csv(file.path(root, "reports", "real_trials.csv"))
published <- read.csv(file.path(root, "reports", "real_ranking.csv"))
stability <- read.csv(file.path(root, "reports", "real_rank_stability.csv"))
detectors <- as.character(published$detector)

mcc <- function(harm, alarm) {
    tp <- sum(harm == 1 & alarm == 1)
    fp <- sum(harm == 0 & alarm == 1)
    fn <- sum(harm == 1 & alarm == 0)
    tn <- sum(harm == 0 & alarm == 0)
    denom <- sqrt(as.numeric(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if (denom == 0) 0 else (tp * tn - fp * fn) / denom
}

cat("point estimates, against reports/real_ranking.csv\n")
failures <- 0
for (d in detectors) {
    got <- mcc(trials$harm, trials[[paste0("alarm_", d)]])
    want <- published$mcc[published$detector == d]
    delta <- abs(got - want)
    ok <- delta <= POINT_TOL
    failures <- failures + !ok
    cat(sprintf("  %-16s mcc %+.15f  |d| %.1e  %s\n", d, got, delta,
                if (ok) "ok" else "FAIL"))
}

# Two resampling schemes. The first treats the 240 trials as independent, which
# is what makes the intervals look narrow. The second resamples whole archetypes
# with replacement, which is the honest unit when twelve failure modes are the
# population of interest.
boot_width <- function(d, cluster) {
    alarm <- trials[[paste0("alarm_", d)]]
    harm <- trials$harm
    archetypes <- unique(trials$archetype)
    stats <- numeric(DRAWS)
    for (b in seq_len(DRAWS)) {
        if (cluster) {
            picked <- sample(archetypes, length(archetypes), replace = TRUE)
            idx <- unlist(lapply(picked, function(a) which(trials$archetype == a)))
        } else {
            idx <- sample.int(nrow(trials), nrow(trials), replace = TRUE)
        }
        stats[b] <- mcc(harm[idx], alarm[idx])
    }
    q <- quantile(stats, c(0.025, 0.975), names = FALSE, na.rm = TRUE)
    q[2] - q[1]
}

cat("\ninterval widths, ", DRAWS, " draws in R against 2000 in the Python\n", sep = "")
for (scheme in c("iid_trial", "cluster_by_archetype")) {
    cluster <- scheme == "cluster_by_archetype"
    got <- sapply(detectors, function(d) boot_width(d, cluster))
    want <- sapply(detectors, function(d)
        stability$ci_width[stability$scheme == scheme & stability$detector == d])
    rel <- abs(mean(got) - mean(want)) / mean(want)
    ok <- rel <= WIDTH_REL_TOL
    failures <- failures + !ok
    cat(sprintf("  %-24s mean width R %.4f  published %.4f  rel %.1f%%  %s\n",
                scheme, mean(got), mean(want), 100 * rel,
                if (ok) "ok" else "FAIL"))
    assign(paste0("w_", scheme), mean(got))
}

# The qualitative claim, which does not depend on the tolerance above.
ratio <- w_cluster_by_archetype / w_iid_trial
cat(sprintf("\nwidth ratio, cluster over iid: %.2fx (R), published %.2fx\n",
            ratio,
            mean(stability$ci_width[stability$scheme == "cluster_by_archetype"]) /
            mean(stability$ci_width[stability$scheme == "iid_trial"])))
if (ratio < 2) {
    cat("FAIL: R does not reproduce the widening the README is built on\n")
    failures <- failures + 1
}

if (failures > 0) {
    cat(sprintf("\n%d checks failed\n", failures))
    quit(status = 1)
}
cat("\nR reproduces the point estimates exactly and the interval widening\n")
