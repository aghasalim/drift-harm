//! How much of the published bootstrap interval is Monte Carlo noise?
//!
//! `experiments/06_ranking_stability.py` draws 2000 bootstrap replicates. That
//! is a Monte Carlo estimate of an interval width, so it carries its own error,
//! and nothing in the repository measured it. This does two things the Python
//! cannot afford:
//!
//!   1. a 100,000 draw reference width, accurate enough to treat as the truth
//!   2. 30 independent 2000 draw runs, whose spread is the error bar on the
//!      published number
//!
//! Then it checks the published width sits inside that error bar. A pass means
//! 2000 draws was enough for the claim being made; a failure would mean the
//! published interval is reporting noise.

use std::env;
use std::fs;
use std::process::exit;

const REFERENCE_DRAWS: usize = 100_000;
const PUBLISHED_DRAWS: usize = 2_000;
const REPLICATES: usize = 30;
const SIGMA: f64 = 4.0;

/// xorshift64*. Not cryptographic and not meant to be: it needs to be uniform,
/// fast, and seeded reproducibly so a failure here can be re-run.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

#[derive(Clone)]
struct Trials {
    harm: Vec<u8>,
    alarms: Vec<Vec<u8>>,
    archetype_groups: Vec<Vec<usize>>,
    detectors: Vec<String>,
}

fn mcc(harm: &[u8], alarm: &[u8], idx: &[usize]) -> f64 {
    let (mut tp, mut fp, mut fn_, mut tn) = (0i64, 0i64, 0i64, 0i64);
    for &i in idx {
        match (harm[i], alarm[i]) {
            (1, 1) => tp += 1,
            (0, 1) => fp += 1,
            (1, 0) => fn_ += 1,
            _ => tn += 1,
        }
    }
    let denom = (((tp + fp) as f64) * ((tp + fn_) as f64)
        * ((tn + fp) as f64) * ((tn + fn_) as f64))
        .sqrt();
    if denom == 0.0 {
        0.0
    } else {
        (tp * tn - fp * fn_) as f64 / denom
    }
}

fn width(t: &Trials, det: usize, cluster: bool, draws: usize, rng: &mut Rng) -> f64 {
    let n = t.harm.len();
    let mut stats = Vec::with_capacity(draws);
    let mut idx = Vec::with_capacity(n);

    for _ in 0..draws {
        idx.clear();
        if cluster {
            for _ in 0..t.archetype_groups.len() {
                let g = rng.below(t.archetype_groups.len());
                idx.extend_from_slice(&t.archetype_groups[g]);
            }
        } else {
            for _ in 0..n {
                idx.push(rng.below(n));
            }
        }
        stats.push(mcc(&t.harm, &t.alarms[det], &idx));
    }

    stats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    quantile(&stats, 0.975) - quantile(&stats, 0.025)
}

/// Linear interpolation between order statistics, which is numpy's default and
/// R's type 7. Matching the convention matters: the alternatives shift a
/// quantile by up to one order statistic, which is visible at this width.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (pos - lo as f64) * (sorted[hi] - sorted[lo])
    }
}

fn load(root: &str) -> Trials {
    let path = format!("{}/reports/real_trials.csv", root);
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| { eprintln!("cannot read {}: {}", path, e); exit(2) });

    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("empty file").split(',').collect();
    let col = |name: &str| header.iter().position(|h| *h == name);

    let harm_col = col("harm").unwrap_or_else(|| { eprintln!("no harm column"); exit(2) });
    let arch_col = col("archetype").unwrap_or_else(|| { eprintln!("no archetype"); exit(2) });

    let detectors: Vec<String> = header
        .iter()
        .filter(|h| h.starts_with("alarm_"))
        .map(|h| h.trim_start_matches("alarm_").to_string())
        .collect();
    let alarm_cols: Vec<usize> =
        detectors.iter().map(|d| col(&format!("alarm_{}", d)).unwrap()).collect();

    let mut harm = Vec::new();
    let mut alarms: Vec<Vec<u8>> = vec![Vec::new(); detectors.len()];
    let mut names: Vec<String> = Vec::new();

    for line in lines.filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').collect();
        harm.push(f[harm_col].trim().parse::<u8>().unwrap_or(0));
        for (k, &c) in alarm_cols.iter().enumerate() {
            alarms[k].push(f[c].trim().parse::<u8>().unwrap_or(0));
        }
        names.push(f[arch_col].to_string());
    }

    let mut uniq: Vec<&String> = names.iter().collect();
    uniq.sort();
    uniq.dedup();
    let archetype_groups = uniq
        .iter()
        .map(|a| names.iter().enumerate()
            .filter(|(_, n)| n == a).map(|(i, _)| i).collect())
        .collect();

    Trials { harm, alarms, archetype_groups, detectors }
}

fn published_width(root: &str, scheme: &str, detector: &str) -> Option<f64> {
    let text = fs::read_to_string(format!("{}/reports/real_rank_stability.csv", root)).ok()?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next()?.split(',').collect();
    let s = header.iter().position(|h| *h == "scheme")?;
    let d = header.iter().position(|h| *h == "detector")?;
    let w = header.iter().position(|h| *h == "ci_width")?;
    for line in lines {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() > w && f[s] == scheme && f[d] == detector {
            return f[w].trim().parse().ok();
        }
    }
    None
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let root = args.get(1).map(String::as_str).unwrap_or(".");
    let t = load(root);

    println!("{} trials, {} archetypes, {} detectors",
             t.harm.len(), t.archetype_groups.len(), t.detectors.len());
    println!("reference {} draws, error bar from {} runs of {} draws\n",
             REFERENCE_DRAWS, REPLICATES, PUBLISHED_DRAWS);

    let mut failures = 0;
    for (scheme, cluster) in [("iid_trial", false), ("cluster_by_archetype", true)] {
        println!("{}", scheme);
        for (d, name) in t.detectors.iter().enumerate() {
            let mut rng = Rng::new(0x5EED_0000 + d as u64 * 7919 + cluster as u64);
            let reference = width(&t, d, cluster, REFERENCE_DRAWS, &mut rng);

            let mut reps = Vec::with_capacity(REPLICATES);
            for r in 0..REPLICATES {
                let mut rr = Rng::new(0xC0FFEE + (r as u64) * 104_729 + d as u64);
                reps.push(width(&t, d, cluster, PUBLISHED_DRAWS, &mut rr));
            }
            let mean: f64 = reps.iter().sum::<f64>() / reps.len() as f64;
            let sd = (reps.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                      / (reps.len() - 1) as f64).sqrt();

            match published_width(root, scheme, name) {
                Some(pubw) => {
                    let z = (pubw - reference).abs() / sd.max(1e-12);
                    let ok = z <= SIGMA;
                    failures += !ok as i32;
                    println!("  {:<16} ref {:.4}  published {:.4}  \
                              noise sd {:.4}  {:.1} sd  {}",
                             name, reference, pubw, sd, z,
                             if ok { "ok" } else { "FAIL" });
                }
                None => {
                    println!("  {:<16} no published width to compare", name);
                    failures += 1;
                }
            }
        }
        println!();
    }

    if failures > 0 {
        println!("{} widths sit further than {} sd from the reference", failures, SIGMA);
        exit(1);
    }
    println!("every published interval width is within {} sd of a {} draw reference,\n\
              so 2000 draws was enough for the claim made from it", SIGMA, REFERENCE_DRAWS);
}
