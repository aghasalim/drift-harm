#!/usr/bin/env bash
# Recompute the published ranking in every language here and require agreement.
#
# The numbers in the README come from pandas. So does every figure. If the
# aggregation in experiments/03_tables.py were wrong, nothing downstream would
# notice, because everything downstream reads the same output. These are
# independent implementations from the trial level data, and a mistake would
# have to be made identically in all of them to survive.
#
# Each is skipped with a clear message if its toolchain is absent, so this runs
# on a laptop with only some of them. CI has all of them.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

pass=0 fail=0 skip=0

run () {
    local name="$1" tool="$2"; shift 2
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    if "$@"; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
}

# SQL has no assertion of its own, so compare its output here. Both sides are
# reformatted to a fixed precision first: sqlite prints 15 significant digits
# and pandas prints 17, and that difference is not a disagreement.
check_sql () {
    local a b
    a=$(sqlite3 -init verify/ranking.sql :memory: "" 2>/dev/null \
        | awk -F, 'NR>1 {printf "%s %.10f %.10f %.10f %.10f %d %d %d %d\n", \
                          $1,$2,$3,$4,$5,$8,$9,$10,$11}' | sort)
    b=$(awk -F, 'NR>1 {printf "%s %.10f %.10f %.10f %.10f %d %d %d %d\n", \
                       $1,$2,$3,$4,$5,$8,$9,$10,$11}' reports/real_ranking.csv | sort)
    if [ "$a" = "$b" ]; then
        echo "SQL reproduces all 6 rows of reports/real_ranking.csv to 1e-10"
        return 0
    fi
    echo "SQL disagrees with the published ranking:"
    diff <(echo "$a") <(echo "$b") | head -20
    return 1
}

check_c () {
    cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror \
       -o "${TMPDIR:-/tmp}/mcc" verify/mcc.c -lm || return 1
    "${TMPDIR:-/tmp}/mcc" "$root"
}

check_go () { ( cd verify/gocheck && go run . -root "$root" ); }

check_rust () {
    ( cd verify/bootstrap && cargo run --release --quiet -- "$root" )
}

run "SQL, aggregation"        sqlite3 check_sql
run "C, metric kernel"        cc      check_c
run "Go, file validation"     go      check_go
run "R, statistical inference" Rscript Rscript verify/verify.R "$root"
run "Rust, bootstrap error"   cargo   check_rust
run "JS, leave one archetype out" node    node verify/loo.js "$root"

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
