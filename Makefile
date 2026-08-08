.PHONY: setup setup-train lint format typecheck test test-quick check splits eval-gold docker-up clean

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
