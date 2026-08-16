# DriftHarm

Drift detectors are usually scored on whether they noticed that a distribution
moved. That is not the question a person on call has. The question is whether
the alert meant the model got worse.

So I built a suite where I know the answer. Twelve failure archetypes are
applied to windows drawn from a held-out pool, a model that has seen neither
window scores both, and the drop in its AUC — measured against a null of
window pairs where nothing happened — is the harm label. Six detectors are
calibrated against that same null at a common 5% false-alarm target, so no
detector is running a tighter threshold than any other. Then I cross-tabulate
alarms against harm and score by MCC.

The headline finding is a negative one: **the resulting order is not stable
enough to be a ranking.** What survives is the per-failure-mode table, the
measured harm labels, and a set of specific mechanisms — which is what the rest
of this is about.

Everything below comes from a file in [`reports/`](reports/). Where I did not
measure something, I say so.

## Headline: this is not a ranking

240 trials (12 archetypes × 20 replicates), 20,000-row windows, harm base rate
51.7%. MCC, precision and recall are all with respect to *harm*, not with
respect to whether the distribution moved.

| detector | MCC | 95% CI, trials resampled | 95% CI, **archetypes resampled** | P(MCC > 0) | P(best of six) | harm-precision | harm-recall | specificity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MMD | 0.189 | [0.061, 0.315] | [−0.353, 0.674] | 0.76 | 0.57 | 0.593 | 0.694 | 0.491 |
| C2ST | 0.089 | [−0.035, 0.207] | [−0.353, 0.495] | 0.66 | 0.18 | 0.539 | 0.831 | 0.241 |
| Wasserstein | 0.058 | [−0.069, 0.176] | [−0.468, 0.535] | 0.59 | 0.15 | 0.535 | 0.742 | 0.310 |
| KS | 0.051 | [−0.081, 0.179] | [−0.460, 0.542] | 0.59 | 0.11 | 0.536 | 0.661 | 0.388 |
| Jensen-Shannon | −0.003 | [−0.137, 0.121] | [−0.511, 0.476] | 0.51 | 0.00 | 0.516 | 0.669 | 0.328 |
| PSI | −0.012 | [−0.141, 0.112] | [−0.559, 0.496] | 0.50 | 0.00 | 0.512 | 0.661 | 0.328 |

Source: [`reports/real_ranking.csv`](reports/real_ranking.csv),
[`reports/real_rank_stability.csv`](reports/real_rank_stability.csv). Both
intervals are 2,000-draw percentile bootstraps; `P(MCC > 0)` and
`P(best of six)` are under the archetype-resampled one.

**Read the fourth column, not the second.** The narrow interval resamples the
240 trials as if they were 240 independent facts. They are not: in 53 of the 72
(detector × archetype) cells on real data the alarm rate is exactly 0.00 or
1.00, so a replicate tells you almost nothing the archetype has not already
fixed. Resampling the twelve archetypes as clusters instead widens every
interval by a factor of 3.5 to 4.2, and every one of them then contains zero.
The detector with the best point estimate is best in 57% of resampled suites;
three of the other five take first place between 11% and 18% of the time.

So the honest summary of this table is: **on this suite, at this size, no
detector's harm-MCC is distinguishable from zero, and the ordering should not be
read as a ranking.** Harm-precision runs 0.512 to 0.593 against a base rate of
0.517 — being told a detector fired moves my belief that the model is damaged by
between −0.5 and +7.6 percentage points. That is the result.

This corrects what this README used to say. It previously reported the narrow
interval as *the* interval and claimed MMD's excluded zero. That claim holds
only if you treat these exact twelve archetypes as the entire universe of
failures, which the limitations section already says they are not.

## Why the real and synthetic rankings disagree

Running the same code on a 60-dimensional correlated-Gaussian control gives a
different order — MMD 1st → 5th, PSI 6th → 3rd, Spearman −0.43:

| detector | MCC (synthetic) | 95% CI, archetypes resampled | P(best of six) | rank on real |
| --- | --- | --- | --- | --- |
| Jensen-Shannon | 0.169 | [−0.325, 0.566] | 0.56 | 5th |
| Wasserstein | 0.161 | [−0.329, 0.564] | 0.22 | 3rd |
| PSI | 0.127 | [−0.313, 0.468] | 0.03 | 6th |
| C2ST | 0.099 | [−0.340, 0.485] | 0.02 | 2nd |
| MMD | 0.094 | [−0.297, 0.425] | 0.17 | **1st** |
| KS | 0.026 | [−0.398, 0.446] | 0.00 | 4th |

Source: [`reports/synthetic_ranking.csv`](reports/synthetic_ranking.csv),
[`reports/synthetic_rank_stability.csv`](reports/synthetic_rank_stability.csv).

I set out to find what made the two datasets disagree. The main answer is that
**there is no cross-dataset effect left to explain once the suite's own
instability is priced in** — but two real, mechanical differences sit
underneath it, and both are worth having. In order.

### The disagreement is inside the range one dataset produces against itself

Take one dataset. Draw two independent bootstrap resamples of it. Rank the six
detectors in each. Correlate the two rankings. That is the reference
distribution the observed −0.43 has to be read against, and it is in
[`reports/ranking_agreement.csv`](reports/ranking_agreement.csv):

| resampling unit | mean self-Spearman, real | 5th pct | P(self-Spearman ≤ −0.43) |
| --- | --- | --- | --- |
| trials | +0.82 | +0.43 | 0.000 |
| replicates within archetype | +0.87 | +0.71 | 0.000 |
| **archetypes (clusters)** | **+0.31** | **−0.58** | **0.146** |

Synthetic gives 0.148 for the same cell. So under the resampling that treats the
taxonomy as one draw of twelve designed failures, a dataset resampled against
*itself* produces a ranking correlation at or below −0.43 about one time in
seven. The real-versus-synthetic disagreement is a draw from that distribution.
It is not evidence that the two datasets are asking different questions; it is
one more sample of how unstable a six-detector ranking over twelve archetypes
is.

The direct demonstration: **dropping a single archetype reorders the real
ranking more than switching datasets does.** Twelve leave-one-out rankings
([`reports/real_leave_one_archetype_out.csv`](reports/real_leave_one_archetype_out.csv)):
removing `imputation_masked_null` gives Spearman −0.46 against the full-suite
order and makes KS the winner; removing `dilution_shift` gives 0.67 and makes
Wasserstein the winner. The other ten leave MMD on top, and five of them leave
the order completely unchanged. Two archetypes out of twelve carry the result.

### What is genuinely different, and it is those same two archetypes

Same twelve archetype names, two datasets, different experiments
([`reports/archetype_disagreement.csv`](reports/archetype_disagreement.csv)).
Mean |harm-rate difference| across the twelve is 0.204:

| archetype | harm rate, real | harm rate, synthetic | mean \|alarm-rate difference\| |
| --- | --- | --- | --- |
| imputation_masked_null | 1.00 | 1.00 | **0.60** |
| gradual_shift | 1.00 | 0.55 | 0.29 |
| dilution_shift | **0.05** | **1.00** | 0.26 |
| dilution_permuted | 0.05 | 0.15 | 0.16 |
| covariate_shift_strong | 0.65 | 0.25 | 0.00 |
| irrelevant_feature_drift | 0.05 | 0.35 | 0.00 |
| the other six | — | — | ≤ 0.09 |

The two archetypes the ranking hangs on are the two the datasets disagree about
most, and MMD's first place on real data is built out of exactly those two: it
is one of only two detectors that catch `imputation_masked_null` (20/20, against
KS and PSI's 0/20) and the only one that stays quiet on `dilution_shift`, which
the aggregate harm rule scores as harmless (0.05). Neither holds on synthetic —
there everyone catches the masked null, and `dilution_shift` is harmful in
20/20 replicates, so declining to fire on it is a miss rather than a saved false
alarm.

**Why the masked-null blind spot does not reproduce.** I had written this up as
a NaN-policy result: the univariate detectors drop non-finite values, so they
compare the surviving values against an unchanged reference and see nothing.
That mechanism is incomplete. They do not see nothing — dropping 90% of a
column leaves a small-sample footprint, and it is *the same size on both
datasets*
([`reports/masked_null_footprint.csv`](reports/masked_null_footprint.csv)):

| | mean max-KS on the archetype | mean max-KS under the null | alarm threshold | alarm rate |
| --- | --- | --- | --- | --- |
| real | 0.0287 | 0.0297 | 0.0444 | 0/20 |
| synthetic | 0.0288 | 0.0163 | 0.0203 | 20/20 |

Identical footprint, opposite verdict, because the *null floors* differ. On
IEEE-CIS the max-over-columns KS null is set by columns that are already mostly
missing before any archetype touches them: `id_02` is 81.3% NaN (effective n
3,736 per window) and produces a null KS of 0.0251 on its own; `D8` is 89.8%
NaN; 19 of the 60 monitored columns are above 80% NaN
([`reports/real_null_floor_by_column.csv`](reports/real_null_floor_by_column.csv)).
Masking a dense column to 10% produces the same effective sample size those
columns already have, so the footprint lands under a floor they built. The
synthetic bundle has no missing values at all, so its floor sits at n = 20,000
and the identical footprint stands 1.8× above it.

That is a sharper finding than the one it replaces: **a drop-NaN detector's
blindness to a feed outage is not a property of the detector, it is a property
of how much missingness the reference table already had.** The synthetic bundle
is a monitored table with no missing values at all, and KS caught the same
failure there on 20 of 20 replicates.

### Labels or alarm behaviour? Both, and neither is decisive

Scoring one dataset's alarms against the other's harm labels, pairing replicates
at random within archetype and averaging over 400 pairings
([`reports/ranking_swap_decomposition.csv`](reports/ranking_swap_decomposition.csv)):

| alarms from | harm labels from | winner | Spearman vs the real ranking |
| --- | --- | --- | --- |
| real | real | MMD | 1.00 |
| real | synthetic | C2ST | 0.60 |
| synthetic | real | PSI | −0.83 |
| synthetic | synthetic | Jensen-Shannon | −0.43 |

Both factors move the order and the alarm-behaviour factor moves it more
(−0.83 versus 0.60). MMD's MCC falls from 0.189 to 0.004 under synthetic harm
labels and to 0.049 under synthetic alarm behaviour. But every one of these
shifts is inside the archetype-resampled interval, so the decomposition says
*where* the difference lives without establishing that any of it is signal.

### Two hypotheses I tested and rejected

- **Dimensionality and sample size.** Not the explanation, and not even a
  difference: the detectors consume a 60-column × 20,000-row matrix on both
  datasets, at the same α, the same 300 null replicates and the same 20
  replicates per archetype ([`reports/real_run_meta.json`](reports/real_run_meta.json),
  [`reports/synthetic_run_meta.json`](reports/synthetic_run_meta.json)). The real
  *model* has 431 features against the synthetic model's 60, but the 371 it is
  not handed are held at the training median, so that difference reaches the
  harm label, not the detectors.
- **More replicates would settle it.** They would not. Resampling replicates
  while holding the twelve archetypes fixed already gives MMD [0.142, 0.241] and
  P(best) = 1.00 — the estimate *conditional on this taxonomy* is precise
  and more trials would only tighten it further. The uncertainty is not in the
  sample size, it is in the choice of the twelve archetypes, and no number of
  replicates touches that.

I would not carry a detector choice from either of these datasets to the other,
and I do not think anyone should carry one from this benchmark to their own
stack without re-running it there.

## Where the errors actually come from

Alarm rate per archetype on real data, next to the harm rate I measured on the
same trials ([`reports/real_by_archetype.csv`](reports/real_by_archetype.csv)):

| archetype | designed harmful? | measured harm rate | mean AUC drop | KS | PSI | Wass | JS | MMD | C2ST |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| true_null | no | 0.05 | −0.005 | 0.00 | 0.00 | 0.10 | 0.05 | 0.05 | 0.05 |
| covariate_shift_mild | no | 0.10 | 0.001 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| covariate_shift_moderate | no | 0.25 | 0.011 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| covariate_shift_strong | no | **0.65** | 0.033 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| concept_drift_no_covariate_shift | yes | 1.00 | **0.337** | 0.00 | 0.00 | 0.10 | 0.05 | 0.05 | 0.05 |
| imputation_masked_null | yes | 1.00 | 0.056 | **0.00** | **0.00** | 0.30 | 0.05 | 1.00 | 1.00 |
| imputation_visible | yes | 1.00 | 0.056 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| dilution_shift | yes | 0.05 | 0.003 | 0.65 | 1.00 | 1.00 | 0.85 | 0.05 | 0.90 |
| dilution_permuted | yes | 0.05 | 0.002 | 0.00 | 0.00 | 0.10 | 0.05 | 0.05 | 0.55 |
| irrelevant_feature_drift | no | 0.05 | −0.005 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| sudden_shift | yes | 1.00 | 0.137 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| gradual_shift | yes | 1.00 | 0.093 | 1.00 | 1.00 | 1.00 | 1.00 | **0.05** | 1.00 |

Reading down the columns:

**Irrelevant-feature drift is a clean 100% false-alarm rate for all six.** I
shifted 20 monitored columns whose LightGBM gain importance is exactly zero by
3 standard deviations. The model's predictions on those trials are unchanged —
the mean AUC drop, −0.004871, is bit-identical to the true-null archetype's,
because a column that never enters a split cannot move a prediction. Every
detector fires on 20/20 replicates. This is the single largest source of false
alarms in the benchmark and it is entirely avoidable by not monitoring columns
the model does not use.

**Covariate shift is not reliably benign, and I had assumed it was.** I designed
three strengths of importance-weighted resampling of real rows, which preserves
P(y|x) exactly and moves only P(x), and labelled all three as expected-harmless.
The measured harm rates are 0.10, 0.25 and 0.65. At strength 2.0 the mean AUC
drop is 0.033 against a null threshold of 0.021, and two thirds of replicates
clear it. Preserving the conditional is not sufficient for a fixed model to
survive: a tree fitted on the training density degrades when the query density
moves far enough into its sparse regions. I have left the archetype's
`expected_harm` flag at `False` and let the disagreement stand in the artifacts,
because the disagreement is the result.

**Concept drift is the largest harm in the suite and nearly invisible.** Flipping
35% of labels with the feature matrix held bit-identical produces a mean AUC
drop of 0.337 — about 2.5× the next largest — and the detectors alarm on 0 to 2 of 20
replicates, which is their false-alarm rate. This is not a tuning failure. No
function of P(x) can see a change in P(y|x) when P(x) has not moved. It is in
the suite so the blind spot has a number attached.

**The NaN policy decides whether a feed outage is visible — on this dataset.**
`imputation_masked_null` nulls 90% of the values in the six highest-importance
columns; serving imputes the reference median, so the model loses the
information and the harm rate is 1.00. The univariate detectors drop non-finite
values before comparing — which is what `scipy.stats.ks_2samp` and every PSI
implementation I have read do — so they compare the surviving values against a
reference whose shape has not changed: KS 0/20, PSI 0/20. MMD and C2ST catch
20/20, but only because they must impute before they can compute anything, so
they are looking at the model's view by accident rather than by design.

The qualifier matters and I only found it by chasing the synthetic disagreement.
Dropping the nulls does not erase the signal, it converts it into a
sample-size effect, and whether that clears the alarm threshold depends on how
much missingness the *reference* columns already carried. The same archetype on
the synthetic bundle gives KS 20/20 off an almost identical raw statistic. See
the diagnosis section above for the two null floors.

`imputation_visible` is the same failure with the monitor wired to the
post-imputation feature vector. Identical harm (mean AUC drop 0.0560 vs 0.0559,
both 1.00 harm rate); KS and PSI go from 0/20 to 20/20. **Changing which table
the monitor reads bought more recall on this failure than changing the
detector did.**

**The best-scoring detector has the worst detection delay.** In the gradual-vs-sudden
run ([`reports/real_gradual_summary.csv`](reports/real_gradual_summary.csv)), the
gradual arm's harm rate is already 1.00 at batch 1 with a mean AUC drop of
0.076. Five detectors alarm on 6/6 replicates at batch 1. MMD alarms on 0/6, and
does not reach 6/6 until batch 5 — by which point the mean AUC drop is 0.129.
On the synthetic bundle, where the gradual arm's harm rate rises from 0.33 at
batch 1 to 1.00 by batch 3, MMD again lags to batch 5 while PSI and JS are at
1.00 by batch 2. MMD's top score on real data is bought partly with
insensitivity: it declines the `dilution_shift` false alarms that cost the other
five (0/19 versus 12–19/19) and pays for it by missing 19/20 gradual trials.

## Instrument findings

Things I found wrong with the measuring apparatus, kept here rather than fixed
in silence.

### 1. The harm label is blind to segment damage, and the ranking depends on it

The headline harm label is a threshold on the *aggregate* AUC drop over the
whole 20,000-row window. Both dilution archetypes confine their damage to the
top 3% of rows by predicted risk — a slice carrying 40.9% of the positives — and
under that rule they score as harmless: harm rate 0.05, mean aggregate AUC drop
0.003 and 0.002.

They are not harmless. The 3%-segment AUC drop on those same trials averages
0.182 (`dilution_shift`) and 0.381 (`dilution_permuted`) against a separately
null-calibrated segment threshold of 0.111, and clears it in 20/20 replicates
for both. So the detectors that fire on dilution are being charged for false
alarms on trials where the model is badly damaged inside the segment an operator
would actually care about.

Re-scoring with `harm = aggregate OR segment` inverts the ranking
([`reports/real_ranking_segment_aware.csv`](reports/real_ranking_segment_aware.csv)):

| detector | MCC (headline) | MCC (segment-aware) |
| --- | --- | --- |
| C2ST | 0.089 | **0.046** |
| Wasserstein | 0.058 | −0.081 |
| PSI | −0.012 | −0.132 |
| Jensen-Shannon | −0.003 | −0.145 |
| KS | 0.051 | −0.172 |
| MMD | **0.189** | **−0.216** |

MMD goes from first to last; C2ST goes from second to first and is the only
detector still above zero. Every MCC falls. I report the aggregate rule as the
headline because it is one consistent rule applied to all twelve archetypes,
whereas only two archetypes define a segment — but the honest summary is that
**the order is an artifact of a harm definition that a reasonable person could
set differently, and setting it differently reverses it.** Both tables are in
`reports/`; neither is the answer. This was the first sign of the instability
the headline section now quantifies: one defensible change to the labelling
moves the order about as far as changing dataset does, and both moves are inside
the archetype-resampled interval.

### 2. Threshold calibration needs more null replicates than I first used

A threshold is the (1 − α) empirical quantile of a null sample, which means at
α = 0.05 it is fitted to a handful of order statistics. I originally calibrated
on 60 null replicates. Sweeping the replicate count over the 300 saved null
scores — 400 random calibration/validation resplits per size, 100 held-out
replicates each — shows what that costs
([`reports/real_calibration_size.csv`](reports/real_calibration_size.csv)):

| calibration reps | mean realised FAR, real (across 6 detectors) | p90 realised FAR |
| --- | --- | --- |
| 20 | 8.9% – 9.3% | 18% – 19% |
| 40 | 7.0% – 7.7% | 13% – 14% |
| 60 | 6.2% – 6.8% | 11% – 12% |
| 100 | 5.7% – 6.0% | 10% – 11% |
| 150 | 5.5% – 5.8% | 10% |

Target is 5%. The benchmark now uses 300 null replicates split 150 for
calibration and 150 held out, and the calibration half is frozen before any
archetype is scored.

Two corrections to how I first reported this, both against the artifacts:

- I previously wrote that out-of-sample false-alarm rates were "4.0–6.7%". The
  actual realised range in
  [`reports/real_null_summary.csv`](reports/real_null_summary.csv) is **0.0% to
  6.7%** — MMD realised 0.0% and C2ST 2.0%, both well *under* target, which is
  its own calibration problem. 4.0–6.7% was the four univariate detectors only.
  On synthetic the range is **2.7% to 10.0%**
  ([`reports/synthetic_null_summary.csv`](reports/synthetic_null_summary.csv)),
  with Wasserstein at exactly double the target. Calibration is not solved; it
  is better.
- I previously quoted a 16.7% realised false-alarm rate for KS at 60 calibration
  replicates. **I could not reproduce that figure from the saved artifacts.** The
  sweep gives KS a mean of 6.75% at 60 replicates with a p90 of 12%; 16.7%
  (= 10/60) implies a validation split I no longer have. I am leaving the finding
  in because the direction it points at is confirmed by the table above, but the
  specific number was a single draw and should not be quoted.

### 3. The synthetic bundle's "irrelevant" features are not causally irrelevant

`build_synthetic_bundle` sets 20 of 60 generative weights to exactly zero and
then forces their reported gain importance to zero, so that
`irrelevant_feature_drift` has ground-truth-zero harm. It does not. The design
matrix is block-correlated (`cov = A Aᵀ + I`), so the zero-weight columns carry
information about the informative ones, and the fitted LightGBM puts **3.6% of
its total split gain** on them. Shifting them therefore does move predictions:
the measured harm rate for `irrelevant_feature_drift` on synthetic is **0.35**,
not the ~0.05 the design intended.

The real bundle does not have this problem — its zero-gain columns produce a
mean AUC drop of −0.004871, bit-identical to true_null. So the synthetic
irrelevant-feature result should be read as "drift in weakly-correlated
low-importance columns", and the clean version of that archetype is the real one.
Fixing this needs an independent-covariance synthetic bundle, which I did not run.

### 4. MCC over F1, for a reason that shows up in the numbers

C2ST has the best harm-F1 on real data (0.654) and comes second on MCC (0.089),
because F1 ignores true negatives and C2ST buys its 0.831 recall with 88 false
positives and a specificity of 0.241. MCC responds to the whole table, which is
why it is the scoring column. Both are in `reports/`.

## Limitations

- **One real dataset, one synthetic generator.** IEEE-CIS is tabular fraud with a
  3.7% positive rate. Nothing here has been checked on text, images, time series,
  or a different tabular domain.
- **The suite is the sample size, not the trial count.** 240 trials sounds like
  240 facts and is closer to 12: the alarm rate is exactly 0.00 or 1.00 in 53 of
  the 72 (detector × archetype) cells. Resampling archetypes instead of trials
  puts zero inside every detector's interval on both datasets
  ([`reports/real_rank_stability.csv`](reports/real_rank_stability.csv)). More
  replicates cannot fix this; more *archetypes* might.
- **The 12 archetypes are my taxonomy, not an exhaustive one**, and that is where
  essentially all of the uncertainty lives. Dropping one of the twelve reorders
  the real result by Spearman −0.46. I have not run a significance test on any
  pairwise MCC gap, and given the above I do not think one would be meaningful.
- **Detector hyperparameters are fixed and not swept.** 10 PSI bins, 20 JS bins,
  1,500-row MMD subsample, 8,000-row C2ST subsample with a 120-tree LightGBM
  discriminator. A detector may look bad here because of a setting rather than
  because of the method.
- **Aggregation over columns is `max`, always.** The four univariate detectors
  alert if any monitored feature drifts. Mean, or a top-k rule, or a
  multiple-testing correction would each give different numbers. I did not
  measure them.
- **Harm is binary AUC drop.** Not calibration, not precision at an operating
  threshold, not money. A detector optimal for AUC-drop harm need not be optimal
  for the thing anyone is paid to protect.
- **Window size is fixed at 20,000 rows.** Since prior work finds false-alarm
  behaviour to be batch-size dependent, results at 500 or 200,000 rows are
  unmeasured here.
- **The gradual archetype is synthetic corruption accrued over batches**, not real
  drift read off the time axis. I did not build a time-ordered gradual arm.
- **Detection delay is measured in batches, not wall-clock or transaction count**,
  and only for one corruption profile.

## Relation to my own earlier work, and to prior art

I need to be precise about what is new here, because the headline observation is
not.

**[aghasalim/mlops-fraud-pipeline](https://github.com/aghasalim/mlops-fraud-pipeline)**
is mine and already showed that drift alerts do not track performance loss. It
monitored KS, PSI and missing-rate over eight windows of IEEE-CIS traffic and
found prediction PSI correlating −0.709 with AUC loss — prediction stability
looking best exactly where the model was worst — and noted, with n = 8, that this
was suggestive rather than conclusive. It also identified the dropped-NaN blind
spot and the invisibility of label shift to input monitors.

So **"drift ≠ harm" is the premise of this repo, not its finding.** What
DriftHarm adds:

1. **Six detectors instead of three**, spanning univariate statistics
   (KS, PSI, Wasserstein, Jensen-Shannon) and multivariate ones (MMD,
   classifier-two-sample-test), all reduced to one comparable scalar.
2. **Constructed ground-truth harm instead of an observational correlation.**
   The earlier repo watched real traffic and correlated two series; here each
   trial has a designed failure mode and a harm label measured against a null,
   so the alert can be scored as a true or false positive rather than correlated.
3. **A taxonomy of twelve archetypes** with the mechanism of each failure written
   down, including three that the earlier repo could not have distinguished:
   segment-confined dilution, marginal-preserving permutation, and the
   monitored-table-versus-model-table pair that isolates observability from harm.
4. **A common calibration procedure** — every detector at the same measured 5%
   null false-alarm rate — so a ranking cannot be explained by one detector
   having a tighter threshold.
5. **A scoring metric** (MCC over the harm/alarm table) with a bootstrap
   interval — and then the finding that under the resampling that matches the
   design, the score cannot separate any two of the six. The negative result is
   the contribution here, not the order it happens to produce.

**Prior art I checked and confirmed:**

- NannyML's public writing makes the same core argument — that drift methods
  produce false alarms because not all drift affects performance, and that
  performance estimation should replace drift as the primary signal. Their
  ["Don't let yourself be fooled by data drift"](https://www.nannyml.com/blog/when-data-drift-does-not-affect-performance-machine-learning-models)
  post demonstrates it on a single dataset (Tetouan City power consumption)
  comparing univariate drift against their DLE performance estimator. It is a
  demonstration rather than a benchmark: it does not rank detectors and does not
  report false-alarm or precision/recall statistics for drift alerts. The
  argument is theirs; the measurement here is not the same measurement.
- **Singh, "When Drift Detectors cry Wolf: False Alarm Rates in continuous ML
  Monitoring"**, [arXiv:2607.17336](https://arxiv.org/abs/2607.17336) (19 Jul
  2026, ICLR 2026 CAO workshop). Measures false positives across PSI, KS, MMD,
  LSDD and adversarial validation under continuous monitoring, and finds PSI
  strongly batch-size sensitive above/below roughly 200 samples. Closest
  published work to finding 2 above, and it measures false alarms *without* harm
  labels — which is precisely the gap this repo tries to fill from the other side.
- **Giobergia, Pastor, de Alfaro & Baralis, "A Synthetic Benchmark to Explore
  Limitations of Localized Drift Detections"**,
  [arXiv:2408.14687](https://arxiv.org/abs/2408.14687) (26 Aug 2024). Induces
  drift in a randomly chosen subgroup and shows commonly adopted detectors fail
  when drift is confined to a small subpopulation. This is direct prior art for
  the two dilution archetypes; my contribution there is only that I also measure
  the segment-level *harm*, which is what exposed instrument finding 1.
- **Cerqueira, Gomes, Heyden, Pfahringer & Bifet, "A Framework for Evaluating and
  Benchmarking Concept Drift Detection Methods"**,
  [arXiv:2606.07789](https://arxiv.org/abs/2606.07789) (5 Jun 2026). Benchmarks
  14 concept-drift detection methods over 7 real datasets with timing-aware
  metrics and Monte-Carlo drift injection. Larger and more rigorous than this
  repo on the detection-quality axis; it scores detection, not downstream harm.

If you want the reliable version of the argument, read those. This repo's claim
is narrower: given a fixed model and a taxonomy of failures with measured harm,
here is what six standard detectors actually score, here is the mechanism behind
each disagreement, and here is the demonstration that the score is too unstable
to be read as a ranking.

## Reproducing

```bash
make setup                # venv + editable install
make test                 # 45 tests, ~7s, no dataset needed
make bench-synthetic      # synthetic run end to end (~24 min on an M-series laptop)
make analysis             # regenerate every table above from reports/*.csv (free)
```

The real run needs the IEEE-CIS `train_transaction.csv` and `train_identity.csv`
from the [Kaggle competition](https://www.kaggle.com/c/ieee-fraud-detection/data)
in `~/ieee-fraud-ml/data/raw/` (path set by `RAW_DIR` in `src/driftharm/data.py`);
then `make bench-real`. It trains a fresh LightGBM on the earliest 40% of the
stream by `TransactionDT` (236,216 rows) and holds out the remaining 354,324 —
the model in my earlier repo was fitted on all 590,540 rows, so it is in-sample
everywhere and its AUC cannot degrade, which makes it useless as a harm
instrument. Held-out AUC is 0.891 on all 431 features and 0.857 through the
60 monitored columns the benchmark drives. The real run took 1,197 s, synthetic
1,437 s.

CI runs the tests on Python 3.12 against the synthetic bundle only, so it never
needs the 700 MB download.

## Layout

```
src/driftharm/
  detectors.py     six detectors, each (ref, cur) -> one scalar; NaN policy stated per detector
  scenarios.py     the twelve archetypes; each returns model-view and monitor-view matrices
  harm.py          AUC drop, aggregate and segment-restricted
  calibration.py   null run, thresholds at a target FAR, out-of-sample null summary
  suite.py         window drawing, archetype application, alarm labelling, gradual curve
  metrics.py       harm precision/recall/F1/MCC, per-archetype table, three bootstrap schemes
  data.py          IEEE-CIS bundle (train early, hold out late) and the synthetic control
experiments/       01 prepare, 02 benchmark, 03 tables, 04 calibration sweep,
                   05 harm-label sensitivity, 06 rank stability and the real-vs-synthetic diagnosis
reports/           every CSV/JSON quoted above — tracked on purpose
tests/             45 tests on the generators and metrics
```

The tests check invariants of the instrument, not that it runs: that
`concept_drift` leaves the input matrix bit-identical, that `dilution_permuted`
preserves every marginal exactly, that `irrelevant_feature_drift` touches only
zero-importance columns, that covariate shift draws only real rows, that MCC
matches scikit-learn, that an always-alarm detector scores zero MCC despite
perfect recall, that a threshold fitted on one null half holds its false-alarm
rate on the other, and that the cluster bootstrap is more than 3× wider than the
flat one when the alarm is fixed by the archetype — which is the claim the
headline section rests on.

MIT licensed.
