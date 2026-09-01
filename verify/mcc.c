/* Recompute the detector ranking from the trial data, in C.
 *
 * Third independent implementation of the same six rows, after pandas in
 * experiments/03_tables.py and SQL in verify/ranking.sql. The point is not
 * speed, it is that a mistake in the pandas aggregation would have to be made
 * identically here and in the SQL to go unnoticed.
 *
 * Reads reports/real_trials.csv, compares against reports/real_ranking.csv,
 * exits non-zero on the first disagreement past the tolerance.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define N_DET 6
#define TOL 1e-12
#define LINE 8192

static const char *DETECTORS[N_DET] = {
    "mmd", "c2st", "wasserstein", "ks", "jensen_shannon", "psi"
};

typedef struct { long tp, fp, fn, tn; } Confusion;

/* Index of a named column in a CSV header, or -1. Resolving by name rather
 * than by position means a column added upstream cannot silently shift what
 * this reads. */
static int column_of(const char *header, const char *name)
{
    char buf[LINE];
    strncpy(buf, header, sizeof buf - 1);
    buf[sizeof buf - 1] = '\0';

    int i = 0;
    for (char *tok = strtok(buf, ",\r\n"); tok; tok = strtok(NULL, ",\r\n"), i++)
        if (strcmp(tok, name) == 0)
            return i;
    return -1;
}

static const char *field(const char *line, int index)
{
    static char out[512];
    int col = 0;
    const char *p = line;
    while (col < index) {
        p = strchr(p, ',');
        if (!p)
            return NULL;
        p++;
        col++;
    }
    const char *end = strchr(p, ',');
    size_t n = end ? (size_t)(end - p) : strlen(p);
    if (n >= sizeof out)
        n = sizeof out - 1;
    memcpy(out, p, n);
    out[n] = '\0';
    char *nl = strpbrk(out, "\r\n");
    if (nl)
        *nl = '\0';
    return out;
}

static double mcc_of(Confusion c)
{
    const double denom = sqrt((double)(c.tp + c.fp) * (c.tp + c.fn)
                            * (c.tn + c.fp) * (c.tn + c.fn));
    return denom == 0.0 ? 0.0 : (double)(c.tp * c.tn - c.fp * c.fn) / denom;
}

int main(int argc, char **argv)
{
    const char *root = argc > 1 ? argv[1] : ".";
    char path[1024], line[LINE], header[LINE];

    snprintf(path, sizeof path, "%s/reports/real_trials.csv", root);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }

    if (!fgets(header, sizeof header, f)) { fclose(f); return 2; }
    const int harm_col = column_of(header, "harm");
    int alarm_col[N_DET];
    for (int d = 0; d < N_DET; d++) {
        char name[128];
        snprintf(name, sizeof name, "alarm_%s", DETECTORS[d]);
        alarm_col[d] = column_of(header, name);
        if (alarm_col[d] < 0) {
            fprintf(stderr, "no column %s in real_trials.csv\n", name);
            fclose(f);
            return 2;
        }
    }
    if (harm_col < 0) { fprintf(stderr, "no harm column\n"); fclose(f); return 2; }

    Confusion conf[N_DET] = {{0, 0, 0, 0}};
    long rows = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '\n' || line[0] == '\0')
            continue;
        const int harm = atoi(field(line, harm_col));
        for (int d = 0; d < N_DET; d++) {
            const int alarm = atoi(field(line, alarm_col[d]));
            Confusion *c = &conf[d];
            if (harm && alarm)        c->tp++;
            else if (!harm && alarm)  c->fp++;
            else if (harm && !alarm)  c->fn++;
            else                      c->tn++;
        }
        rows++;
    }
    fclose(f);
    printf("read %ld trials from real_trials.csv\n", rows);

    snprintf(path, sizeof path, "%s/reports/real_ranking.csv", root);
    f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!fgets(header, sizeof header, f)) { fclose(f); return 2; }
    const int det_col = column_of(header, "detector");
    const int mcc_col = column_of(header, "mcc");
    const int tp_col = column_of(header, "tp");
    const int fp_col = column_of(header, "fp");
    const int fn_col = column_of(header, "fn");
    const int tn_col = column_of(header, "tn");

    int failures = 0, checked = 0;
    while (fgets(line, sizeof line, f)) {
        char det[128];
        strncpy(det, field(line, det_col), sizeof det - 1);
        det[sizeof det - 1] = '\0';

        int d = -1;
        for (int k = 0; k < N_DET; k++)
            if (strcmp(det, DETECTORS[k]) == 0)
                d = k;
        if (d < 0) {
            fprintf(stderr, "unknown detector %s in real_ranking.csv\n", det);
            failures++;
            continue;
        }

        const double want_mcc = atof(field(line, mcc_col));
        const long want[4] = { atol(field(line, tp_col)), atol(field(line, fp_col)),
                               atol(field(line, fn_col)), atol(field(line, tn_col)) };
        const long got[4] = { conf[d].tp, conf[d].fp, conf[d].fn, conf[d].tn };
        const double got_mcc = mcc_of(conf[d]);

        int bad = fabs(got_mcc - want_mcc) > TOL;
        for (int k = 0; k < 4; k++)
            bad |= got[k] != want[k];

        printf("  %-16s tp/fp/fn/tn %3ld %3ld %3ld %3ld   mcc %+.15f   "
               "|d| %.1e  %s\n", det, got[0], got[1], got[2], got[3], got_mcc,
               fabs(got_mcc - want_mcc), bad ? "FAIL" : "ok");
        failures += bad;
        checked++;
    }
    fclose(f);

    if (checked != N_DET) {
        fprintf(stderr, "expected %d detectors, checked %d\n", N_DET, checked);
        return 1;
    }
    if (failures) {
        printf("\n%d of %d rows disagree with reports/real_ranking.csv\n",
               failures, checked);
        return 1;
    }
    printf("\nC agrees with the published ranking on all %d detectors "
           "(tolerance %.0e)\n", checked, TOL);
    return 0;
}
