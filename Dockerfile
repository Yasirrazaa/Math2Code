# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install uv

# Copy project configuration
COPY pyproject.toml .

# Install dependencies
RUN uv pip install --system -e .

# Copy source code
COPY src/ src/

# Expose ports for FastAPI (8000) and Gradio (8501)
EXPOSE 8000
EXPOSE 8501

# Command is overridden in docker-compose
CMD ["python", "src/serve/api.py"]
