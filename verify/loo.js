// Leave one archetype out, recomputed in JavaScript from the trial level data.
//
// reports/real_leave_one_archetype_out.csv is the table behind the README's
// claim that dropping a single archetype can change the winner outright, and
// behind the leave-one-archetype-out figure and the ranking-race gif. It comes
// out of pandas and scipy like everything else, so nothing has ever checked it.
//
// This drops each of the twelve archetypes in turn, recomputes the six MCCs
// from reports/real_trials.csv, ranks them, correlates that ranking against the
// full suite ranking with a Spearman coefficient written here rather than
// imported, and requires the published table to match. It then checks the one
// number the README quotes in prose out of that table.
//
//   node verify/loo.js [repo root]

const fs = require("fs");
const path = require("path");

const root = process.argv[2] || path.join(__dirname, "..");
const DETECTORS = ["ks", "psi", "wasserstein", "jensen_shannon", "mmd", "c2st"];
const TOL = 1e-12;

function readCsv(file) {
    const lines = fs.readFileSync(file, "utf8").trim().split("\n");
    const head = lines[0].trim().split(",");
    return lines.slice(1).map((line) => {
        const cell = line.trim().split(",");
        if (cell.length !== head.length) {
            throw new Error(`ragged row in ${path.basename(file)}: ${cell.length} of ${head.length}`);
        }
        return Object.fromEntries(head.map((h, i) => [h, cell[i]]));
    });
}

// The same definition as src/driftharm/metrics.py: zero when the denominator
// vanishes, which is what a detector that never fires gives you.
function mcc(harm, alarm) {
    let tp = 0, fp = 0, fn = 0, tn = 0;
    for (let i = 0; i < harm.length; i++) {
        if (harm[i]) { alarm[i] ? tp++ : fn++; } else { alarm[i] ? fp++ : tn++; }
    }
    const denom = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
    return denom > 0 ? (tp * tn - fp * fn) / denom : 0.0;
}

// Midranks, so ties get the average rank, as scipy does.
function rank(v) {
    const order = v.map((x, i) => i).sort((a, b) => v[a] - v[b]);
    const r = new Array(v.length);
    for (let i = 0; i < order.length;) {
        let j = i;
        while (j + 1 < order.length && v[order[j + 1]] === v[order[i]]) j++;
        const mid = (i + j) / 2 + 1;
        for (let k = i; k <= j; k++) r[order[k]] = mid;
        i = j + 1;
    }
    return r;
}

function spearman(a, b) {
    const x = rank(a), y = rank(b);
    const mx = x.reduce((s, v) => s + v, 0) / x.length;
    const my = y.reduce((s, v) => s + v, 0) / y.length;
    let num = 0, dx = 0, dy = 0;
    for (let i = 0; i < x.length; i++) {
        num += (x[i] - mx) * (y[i] - my);
        dx += (x[i] - mx) ** 2;
        dy += (y[i] - my) ** 2;
    }
    return num / Math.sqrt(dx * dy);
}

function mccVector(rows) {
    const harm = rows.map((r) => Number(r.harm));
    return DETECTORS.map((d) => mcc(harm, rows.map((r) => Number(r[`alarm_${d}`]))));
}

const trials = readCsv(path.join(root, "reports", "real_trials.csv"));
const published = readCsv(path.join(root, "reports", "real_leave_one_archetype_out.csv"));

const archetypes = [...new Set(trials.map((r) => r.archetype))];
if (archetypes.length !== published.length) {
    console.error(`JS: ${archetypes.length} archetypes in the trials, ${published.length} rows published`);
    process.exit(1);
}

const full = mccVector(trials);
let worst = 0;
const problems = [];

for (const row of published) {
    const a = row.dropped_archetype;
    if (!archetypes.includes(a)) { problems.push(`${a}: not an archetype in real_trials.csv`); continue; }
    const kept = trials.filter((r) => r.archetype !== a);
    const m = mccVector(kept);
    const rho = spearman(full, m);
    const winner = DETECTORS[m.indexOf(Math.max(...m))];

    DETECTORS.forEach((d, k) => {
        const gap = Math.abs(m[k] - Number(row[d]));
        worst = Math.max(worst, gap);
        if (gap > TOL) problems.push(`${a}/${d}: ${m[k]} against published ${row[d]}`);
    });
    const gap = Math.abs(rho - Number(row.spearman_vs_full_suite));
    worst = Math.max(worst, gap);
    if (gap > TOL) problems.push(`${a}/spearman: ${rho} against published ${row.spearman_vs_full_suite}`);
    if (winner !== row.winner) problems.push(`${a}/winner: ${winner} against published ${row.winner}`);
}

// The one row of that table the README quotes in prose.
const masked = published.find((r) => r.dropped_archetype === "imputation_masked_null");
if (!masked) {
    problems.push("imputation_masked_null is missing from the published table");
} else {
    const rho = Number(masked.spearman_vs_full_suite);
    if (masked.winner !== "ks") problems.push(`README says KS wins without imputation_masked_null, table says ${masked.winner}`);
    if (rho.toFixed(2) !== "-0.46") problems.push(`README says Spearman -0.46 without imputation_masked_null, table says ${rho.toFixed(2)}`);
}

if (problems.length) {
    console.error("JS disagrees with the published leave-one-archetype-out table:");
    for (const p of problems.slice(0, 20)) console.error(`  ${p}`);
    process.exit(1);
}

console.log(`JS reproduces all ${published.length} rows of reports/real_leave_one_archetype_out.csv`);
console.log(`  6 MCCs, the winner and the Spearman per row, worst gap ${worst.toExponential(1)}`);
console.log("  and the README's -0.46 / KS claim matches that table");
