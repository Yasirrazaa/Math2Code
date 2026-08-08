# --- Stage 1: build the wheel (validates packaging; not shipped) ---
FROM python:3.11-slim AS builder
RUN pip install --no-cache-dir uv build
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN python -m build --wheel -o /dist

# --- Stage 2: runtime image ---
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# non-root user
RUN useradd -m -u 1000 app && pip install --no-cache-dir uv

WORKDIR /app
COPY --from=builder /dist /dist
RUN uv pip install --system --no-cache /dist/*.whl

COPY src/ ./src/

USER app
EXPOSE 8000 8501
# Overridden in docker-compose for the UI service
CMD ["uvicorn", "math2code.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
