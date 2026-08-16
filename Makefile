PY := .venv/bin/python
PIP := .venv/bin/pip
PYTHON312 ?= /Users/salim/.local/bin/python3.12

.PHONY: setup test bench bench-real bench-synthetic prepare analysis clean

setup:
	$(PYTHON312) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

test:
	$(PY) -m pytest -q

# Requires the IEEE-CIS CSVs; see README. Trains the model and caches the
# held-out pool. Takes a few minutes and ~170 MB of artifacts/.
prepare:
	$(PY) experiments/01_prepare_real.py

bench-real: prepare
	$(PY) experiments/02_run_benchmark.py real

bench-synthetic:
	$(PY) experiments/02_run_benchmark.py synthetic

bench: bench-synthetic bench-real

# Re-derives every table in the README from the saved CSVs. Free: no simulation.
analysis:
	$(PY) experiments/03_tables.py
	$(PY) experiments/04_calibration_size.py
	$(PY) experiments/05_harm_label_sensitivity.py
	$(PY) experiments/06_ranking_stability.py

clean:
	rm -rf artifacts/*.pkl artifacts/*.parquet .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
