.PHONY: setup lint format typecheck test run-api run-ui docker-up clean

setup:
	uv pip install --system -e .[dev]

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

lint:
	ruff check src/ tests/

typecheck:
	mypy src/ tests/

test:
	pytest tests/

run-api:
	uvicorn src.serve.api:app --reload --host 0.0.0.0 --port 8000

run-ui:
	python src/serve/app.py

docker-up:
	docker compose up --build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
