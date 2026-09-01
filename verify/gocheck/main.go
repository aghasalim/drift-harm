// Structural validation of everything under reports/, plus a fourth
// independent recomputation of the detector ranking.
//
// The CSVs in reports/ are the evidence for every number in the README. Nothing
// checked that they are well formed: a truncated write, a column that drifted,
// or a NaN that leaked out of a division would all be invisible until someone
// read the table. This walks every file, and recomputes the ranking from the
// trial data as pandas, SQL and C also do.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const tol = 1e-12

var detectors = []string{"mmd", "c2st", "wasserstein", "ks", "jensen_shannon", "psi"}

type confusion struct{ tp, fp, fn, tn int }

func (c confusion) mcc() float64 {
	d := math.Sqrt(float64(c.tp+c.fp) * float64(c.tp+c.fn) *
		float64(c.tn+c.fp) * float64(c.tn+c.fn))
	if d == 0 {
		return 0
	}
	return float64(c.tp*c.tn-c.fp*c.fn) / d
}

func readCSV(path string) ([]string, [][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = 0 // a ragged file is an error, which is the point
	rows, err := r.ReadAll()
	if err != nil {
		return nil, nil, err
	}
	if len(rows) < 2 {
		return nil, nil, fmt.Errorf("only %d rows", len(rows))
	}
	return rows[0], rows[1:], nil
}

func col(header []string, name string) int {
	for i, h := range header {
		if h == name {
			return i
		}
	}
	return -1
}

// validate reports every structural problem in one file rather than the first,
// so a broken run is diagnosed in one pass.
func validate(path string) []string {
	var problems []string
	header, rows, err := readCSV(path)
	if err != nil {
		return []string{fmt.Sprintf("unreadable: %v", err)}
	}

	seen := map[string]bool{}
	for _, h := range header {
		if h == "" {
			problems = append(problems, "a column has an empty name")
		}
		if seen[h] {
			problems = append(problems, fmt.Sprintf("duplicate column %q", h))
		}
		seen[h] = true
	}

	for i, row := range rows {
		for j, cell := range row {
			low := strings.ToLower(strings.TrimSpace(cell))
			if low == "nan" || low == "inf" || low == "-inf" {
				problems = append(problems,
					fmt.Sprintf("row %d column %s is %s", i+2, header[j], cell))
			}
		}
	}
	return problems
}

func main() {
	root := flag.String("root", ".", "repository root")
	flag.Parse()

	reports := filepath.Join(*root, "reports")
	entries, err := filepath.Glob(filepath.Join(reports, "*.csv"))
	if err != nil || len(entries) == 0 {
		fmt.Fprintf(os.Stderr, "no CSVs under %s\n", reports)
		os.Exit(2)
	}
	sort.Strings(entries)

	bad := 0
	fmt.Printf("validating %d files under reports/\n", len(entries))
	for _, path := range entries {
		if problems := validate(path); len(problems) > 0 {
			bad += len(problems)
			for _, p := range problems {
				fmt.Printf("  %s: %s\n", filepath.Base(path), p)
			}
		}
	}
	if bad == 0 {
		fmt.Printf("  no ragged rows, duplicate columns, NaN or Inf anywhere\n")
	}

	// Fourth recomputation of the ranking.
	header, rows, err := readCSV(filepath.Join(reports, "real_trials.csv"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "real_trials.csv: %v\n", err)
		os.Exit(2)
	}
	harmCol := col(header, "harm")
	if harmCol < 0 {
		fmt.Fprintln(os.Stderr, "real_trials.csv has no harm column")
		os.Exit(2)
	}

	conf := map[string]*confusion{}
	for _, d := range detectors {
		c := col(header, "alarm_"+d)
		if c < 0 {
			fmt.Fprintf(os.Stderr, "no alarm_%s column\n", d)
			os.Exit(2)
		}
		cf := &confusion{}
		for _, row := range rows {
			harm, _ := strconv.Atoi(row[harmCol])
			alarm, _ := strconv.Atoi(row[c])
			switch {
			case harm == 1 && alarm == 1:
				cf.tp++
			case harm == 0 && alarm == 1:
				cf.fp++
			case harm == 1 && alarm == 0:
				cf.fn++
			default:
				cf.tn++
			}
		}
		conf[d] = cf
	}

	pubHeader, pubRows, err := readCSV(filepath.Join(reports, "real_ranking.csv"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "real_ranking.csv: %v\n", err)
		os.Exit(2)
	}
	dCol, mCol := col(pubHeader, "detector"), col(pubHeader, "mcc")

	fmt.Printf("\nrecomputing the ranking from %d trials\n", len(rows))
	for _, row := range pubRows {
		cf, ok := conf[row[dCol]]
		if !ok {
			fmt.Printf("  %s: not in the trial data\n", row[dCol])
			bad++
			continue
		}
		want, _ := strconv.ParseFloat(row[mCol], 64)
		got := cf.mcc()
		delta := math.Abs(got - want)
		status := "ok"
		if delta > tol {
			status = "FAIL"
			bad++
		}
		fmt.Printf("  %-16s mcc %+.15f  |d| %.1e  %s\n", row[dCol], got, delta, status)
	}

	if bad > 0 {
		fmt.Printf("\n%d problems\n", bad)
		os.Exit(1)
	}
	fmt.Printf("\nGo agrees with the published ranking and reports/ is well formed\n")
}
