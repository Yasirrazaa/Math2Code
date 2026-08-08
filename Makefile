.PHONY: setup setup-train lint format typecheck test test-quick splits eval-gold docker-up clean

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

splits:
	python scripts/make_splits.py

eval-gold:
	python -m math2code.evaluation.eval gold --split data/split/test.json --n 100

docker-up:
	docker compose up --build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
