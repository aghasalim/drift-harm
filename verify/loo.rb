# Recompute the leave-one-archetype-out table in Ruby from trial level data.
#
# This is the same check as verify/loo.js but in a second language: drop each
# archetype, recompute the six MCCs, find the winner, compute the Spearman
# correlation against the full-suite ranking, and require the published table
# to match exactly.
#
#   ruby verify/loo.rb [repo root]

root = ARGV[0] || File.join(File.dirname(__FILE__), "..")
DETECTORS = %w[ks psi wasserstein jensen_shannon mmd c2st].freeze
TOL = 1e-12

def read_csv(path)
  lines = File.readlines(path, chomp: true)
  head = lines[0].split(",")
  lines[1..].map { |l| head.zip(l.split(",")).to_h }
end

def mcc(harm, alarm)
  tp = fp = fn = tn = 0
  harm.each_index do |i|
    h = harm[i].to_i
    a = alarm[i].to_i
    if h == 1
      a == 1 ? tp += 1 : fn += 1
    else
      a == 1 ? fp += 1 : tn += 1
    end
  end
  denom = Math.sqrt((tp + fp).to_f * (tp + fn) * (tn + fp) * (tn + fn))
  denom > 0 ? (tp * tn - fp * fn).to_f / denom : 0.0
end

def midrank(v)
  order = v.each_index.sort_by { |i| v[i] }
  r = Array.new(v.size)
  i = 0
  while i < order.size
    j = i
    j += 1 while j + 1 < order.size && v[order[j + 1]] == v[order[i]]
    mid = (i + j) / 2.0 + 1
    (i..j).each { |k| r[order[k]] = mid }
    i = j + 1
  end
  r
end

def spearman(a, b)
  x = midrank(a)
  y = midrank(b)
  mx = x.sum / x.size.to_f
  my = y.sum / y.size.to_f
  num = dx = dy = 0.0
  x.each_index do |i|
    num += (x[i] - mx) * (y[i] - my)
    dx += (x[i] - mx)**2
    dy += (y[i] - my)**2
  end
  num / Math.sqrt(dx * dy)
end

def mcc_vector(rows)
  harm = rows.map { |r| r["harm"] }
  DETECTORS.map { |d| mcc(harm, rows.map { |r| r["alarm_#{d}"] }) }
end

trials = read_csv(File.join(root, "reports", "real_trials.csv"))
published = read_csv(File.join(root, "reports", "real_leave_one_archetype_out.csv"))

archetypes = trials.map { |r| r["archetype"] }.uniq
if archetypes.size != published.size
  $stderr.puts "Ruby: #{archetypes.size} archetypes in the trials, #{published.size} rows published"
  exit 1
end

full = mcc_vector(trials)
problems = []
worst = 0.0

published.each do |row|
  a = row["dropped_archetype"]
  unless archetypes.include?(a)
    problems << "#{a}: not an archetype in real_trials.csv"
    next
  end
  kept = trials.reject { |r| r["archetype"] == a }
  m = mcc_vector(kept)
  rho = spearman(full, m)
  winner = DETECTORS[m.each_with_index.max_by { |v, _| v }[1]]

  DETECTORS.each_with_index do |d, k|
    gap = (m[k] - row[d].to_f).abs
    worst = [worst, gap].max
    problems << "#{a}/#{d}: #{m[k]} against published #{row[d]}" if gap > TOL
  end
  rho_gap = (rho - row["spearman_vs_full_suite"].to_f).abs
  worst = [worst, rho_gap].max
  problems << "#{a}/spearman: #{rho} against published #{row["spearman_vs_full_suite"]}" if rho_gap > TOL
  problems << "#{a}/winner: #{winner} against published #{row["winner"]}" if winner != row["winner"]
end

# The one row the README quotes.
masked = published.find { |r| r["dropped_archetype"] == "imputation_masked_null" }
if masked.nil?
  problems << "imputation_masked_null is missing from the published table"
else
  rho = masked["spearman_vs_full_suite"].to_f
  problems << "README says KS wins without imputation_masked_null, table says #{masked["winner"]}" unless masked["winner"] == "ks"
  problems << "README says Spearman -0.46 without imputation_masked_null, table says #{format("%.2f", rho)}" unless format("%.2f", rho) == "-0.46"
end

unless problems.empty?
  $stderr.puts "Ruby disagrees with the published leave-one-archetype-out table:"
  problems.first(20).each { |p| $stderr.puts "  #{p}" }
  exit 1
end

puts "Ruby reproduces all #{published.size} rows of reports/real_leave_one_archetype_out.csv"
puts "  6 MCCs, the winner and the Spearman per row, worst gap #{format("%.1e", worst)}"
puts "  and the README's -0.46 / KS claim matches that table"
