.PHONY: setup setup-train lint format typecheck test test-quick check splits eval-gold synth docker-up clean

setup:
	uv pip install --system -e ".[dev]"

setup-train:
	uv pip install --system -e ".[dev,train]"

format:
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

typecheck:
	mypy src/math2code tests

test:
	pytest tests/ -q

test-quick:
	pytest tests/ -q -m "not slow"

check: lint typecheck test

eval-gold:
	python -m math2code.evaluation.eval gold --split data/split/test.json --n 100

# Code-first synthetic data: generate + oracle-verify (accept/reject)
# default: 400 derivative/integration objects -> data/synthetic/calculus_indefinite_v1.jsonl
synth:
	python scripts/generate_verified_data.py --families derivative,integration --kind indefinite --count 400 --seed 42 --out data/synthetic/calculus_indefinite_v1.jsonl

synth-definite:
	python scripts/generate_verified_data.py --families integration --kind definite --count 300 --seed 42 --out data/synthetic/calculus_definite_v1.jsonl

synth-variable:
	python scripts/generate_verified_data.py --families integration --kind variable --count 300 --seed 42 --out data/synthetic/calculus_variable_v1.jsonl

synth-derivative:
	python scripts/generate_verified_data.py --families derivative --count 400 --seed 42 --out data/synthetic/derivative_v1.jsonl

synth-functions:
	python scripts/generate_verified_data.py --families functions --count 800 --seed 42 --out data/synthetic/functions_v1.jsonl

synth-ode:
	python scripts/generate_verified_data.py --families ode --count 350 --seed 42 --out data/synthetic/ode_v1.jsonl

synth-multivariate:
	python scripts/generate_verified_data.py --families multivariate --count 600 --seed 42 --out data/synthetic/multivariate_v1.jsonl

synth-sequences:
	python scripts/generate_verified_data.py --families sequences --count 200 --seed 42 --out data/synthetic/sequences_v1.jsonl

synth-geometry:
	python scripts/generate_verified_data.py --families geometry --count 200 --seed 42 --out data/synthetic/geometry_v1.jsonl

synth-edge:
	python scripts/generate_verified_data.py --families edge --count 200 --seed 42 --out data/synthetic/edge_v1.jsonl

synth-numtheory:
	python scripts/generate_verified_data.py --families numtheory --count 400 --seed 42 --out data/synthetic/numtheory_v1.jsonl

synth-diff-c1:
	python scripts/generate_verified_data.py --families differential_c1 --count 350 --seed 42 --out data/synthetic/ode_c1_v1.jsonl

synth-summation:
	python scripts/generate_verified_data.py --families summation --count 300 --seed 42 --out data/synthetic/summation_v1.jsonl

synth-limits:
	python scripts/generate_verified_data.py --families limits --count 200 --seed 42 --out data/synthetic/limits_v1.jsonl

synth-series:
	python scripts/generate_verified_data.py --families series_coeff --count 200 --seed 42 --out data/synthetic/series_v1.jsonl

synth-elementary:
	python scripts/generate_verified_data.py --families elementary_ext --count 600 --seed 42 --out data/synthetic/elementary_v1.jsonl

synth-complex:
	python scripts/generate_verified_data.py --families complex_eval --count 300 --seed 42 --out data/synthetic/complex_v1.jsonl

synth-polynomials:
	python scripts/generate_verified_data.py --families polynomial_invariants --count 300 --seed 42 --out data/synthetic/polynomials_v1.jsonl

synth-matrix:
	python scripts/generate_verified_data.py --families matrix_scalars --count 250 --seed 42 --out data/synthetic/matrix_v1.jsonl

synth-ntheory-ext:
	python scripts/generate_verified_data.py --families ntheory_ext --count 300 --seed 42 --out data/synthetic/ntheory_ext_v1.jsonl

synth-combinatorics:
	python scripts/generate_verified_data.py --families combinatorics --count 250 --seed 42 --out data/synthetic/combinatorics_v1.jsonl

synth-geometry-ext:
	python scripts/generate_verified_data.py --families geometry_ext --count 200 --seed 42 --out data/synthetic/geometry_ext_v1.jsonl

# Gated portfolio slices: verified but EXCLUDED from the default RL mixture
synth-special:
	python scripts/generate_verified_data.py --families special_functions --count 200 --seed 42 --out data/synthetic/special_v1.jsonl

synth-stats:
	python scripts/generate_verified_data.py --families stats_moments --count 200 --seed 42 --out data/synthetic/stats_v1.jsonl

synth-sets:
	python scripts/generate_verified_data.py --families sets_cardinality --count 200 --seed 42 --out data/synthetic/sets_v1.jsonl

synth-solving:
	python scripts/generate_verified_data.py --families solving_scalarized --count 200 --seed 42 --out data/synthetic/solving_v1.jsonl

synth-all:
	python scripts/generate_verified_data.py --families derivative,integration --kind indefinite --count 400 --seed 42 --out data/synthetic/calculus_indefinite_v1.jsonl
	python scripts/generate_verified_data.py --families integration --kind definite --count 300 --seed 42 --out data/synthetic/calculus_definite_v1.jsonl
	python scripts/generate_verified_data.py --families integration --kind variable --count 300 --seed 42 --out data/synthetic/calculus_variable_v1.jsonl
	python scripts/generate_verified_data.py --families derivative --count 400 --seed 42 --out data/synthetic/derivative_v1.jsonl
	python scripts/generate_verified_data.py --families functions --count 800 --seed 42 --out data/synthetic/functions_v1.jsonl
	python scripts/generate_verified_data.py --families ode --count 350 --seed 42 --out data/synthetic/ode_v1.jsonl
	python scripts/generate_verified_data.py --families differential_c1 --count 350 --seed 42 --out data/synthetic/ode_c1_v1.jsonl
	python scripts/generate_verified_data.py --families summation --count 300 --seed 42 --out data/synthetic/summation_v1.jsonl
	python scripts/generate_verified_data.py --families limits --count 200 --seed 42 --out data/synthetic/limits_v1.jsonl
	python scripts/generate_verified_data.py --families series_coeff --count 200 --seed 42 --out data/synthetic/series_v1.jsonl
	python scripts/generate_verified_data.py --families elementary_ext --count 600 --seed 42 --out data/synthetic/elementary_v1.jsonl
	python scripts/generate_verified_data.py --families complex_eval --count 300 --seed 42 --out data/synthetic/complex_v1.jsonl
	python scripts/generate_verified_data.py --families polynomial_invariants --count 300 --seed 42 --out data/synthetic/polynomials_v1.jsonl
	python scripts/generate_verified_data.py --families matrix_scalars --count 250 --seed 42 --out data/synthetic/matrix_v1.jsonl
	python scripts/generate_verified_data.py --families ntheory_ext --count 300 --seed 42 --out data/synthetic/ntheory_ext_v1.jsonl
	python scripts/generate_verified_data.py --families combinatorics --count 250 --seed 42 --out data/synthetic/combinatorics_v1.jsonl
	python scripts/generate_verified_data.py --families geometry_ext --count 200 --seed 42 --out data/synthetic/geometry_ext_v1.jsonl
	python scripts/generate_verified_data.py --families special_functions --count 200 --seed 42 --out data/synthetic/special_v1.jsonl
	python scripts/generate_verified_data.py --families stats_moments --count 200 --seed 42 --out data/synthetic/stats_v1.jsonl
	python scripts/generate_verified_data.py --families sets_cardinality --count 200 --seed 42 --out data/synthetic/sets_v1.jsonl
	python scripts/generate_verified_data.py --families solving_scalarized --count 200 --seed 42 --out data/synthetic/solving_v1.jsonl

# Training mixture: 65% frozen competition + 35% verified synthetic (caps),
# latex-deduped, contamination-checked vs frozen test/val. Deterministic (seed 42).
mixture:
	python scripts/build_mixture.py --size 22002 --out data/synthetic/train_mixture_v1.jsonl

smoke-pool:
	python scripts/smoke_pool.py

publish-data:
	python scripts/publish_hf.py

bench-api:
	python -m math2code.evaluation.runner --split data/split/test.json --model api:deepseek

bench-hf:
	python -m math2code.evaluation.runner --split data/split/test.json --model hf:AI-MO/NuminaMath-7B-TIR

baselines:
	python -m math2code.evaluation.baselines --split data/split/test.json

plots:
	python scripts/plot_results.py --predictions $$(ls -t results/baseline_latex_parse_*.csv | head -1)

analyze:
	python scripts/analyze_results.py --split data/split/test.json --predictions $$(ls -t results/baseline_latex_parse_*.csv | head -1)

docker-up:
	docker compose up --build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
