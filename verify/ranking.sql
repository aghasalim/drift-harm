-- Recompute reports/real_ranking.csv from the trial level data.
--
-- The published ranking is produced by experiments/03_tables.py in pandas.
-- This derives the same six rows from reports/real_trials.csv with nothing but
-- SQL, so an error in the pandas aggregation would have to be reproduced here
-- to survive. verify/verify.sh diffs the two.
--
-- Run: sqlite3 -init verify/ranking.sql :memory: ""

.mode csv
.headers off
.import --csv reports/real_trials.csv trials

CREATE TEMP VIEW alarms AS
    SELECT 'mmd'            AS detector, harm, alarm_mmd            AS alarm FROM trials
    UNION ALL SELECT 'c2st',            harm, alarm_c2st            FROM trials
    UNION ALL SELECT 'wasserstein',     harm, alarm_wasserstein     FROM trials
    UNION ALL SELECT 'ks',              harm, alarm_ks              FROM trials
    UNION ALL SELECT 'jensen_shannon',  harm, alarm_jensen_shannon  FROM trials
    UNION ALL SELECT 'psi',             harm, alarm_psi             FROM trials;

-- Harm is the truth column and alarm is the prediction, so a true positive is
-- a detector that fired on a window where the model actually got worse.
CREATE TEMP VIEW confusion AS
    SELECT detector,
           SUM(CAST(harm AS INT) = 1 AND CAST(alarm AS INT) = 1) AS tp,
           SUM(CAST(harm AS INT) = 0 AND CAST(alarm AS INT) = 1) AS fp,
           SUM(CAST(harm AS INT) = 1 AND CAST(alarm AS INT) = 0) AS fn,
           SUM(CAST(harm AS INT) = 0 AND CAST(alarm AS INT) = 0) AS tn
    FROM alarms
    GROUP BY detector;

.headers on
SELECT detector,
       (tp * tn - fp * fn) /
           sqrt(CAST((tp + fp) AS REAL) * (tp + fn) * (tn + fp) * (tn + fn)) AS mcc,
       2.0 * tp / (2.0 * tp + fp + fn)          AS harm_f1,
       CAST(tp AS REAL) / (tp + fp)             AS harm_precision,
       CAST(tp AS REAL) / (tp + fn)             AS harm_recall,
       CAST(tn AS REAL) / (tn + fp)             AS specificity,
       (CAST(tp AS REAL) / (tp + fn) + CAST(tn AS REAL) / (tn + fp)) / 2.0
                                                AS balanced_accuracy,
       tp, fp, fn, tn
FROM confusion
ORDER BY mcc DESC;
