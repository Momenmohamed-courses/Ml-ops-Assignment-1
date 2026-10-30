FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project

COPY production_app ./production_app

RUN useradd --create-home appuser
USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_PATH=models/pipeline.joblib

EXPOSE 8000

CMD ["uvicorn", "production_app.api:app", "--host", "0.0.0.0", "--port", "8000"]