FROM python:3.10.20-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /bin/
LABEL org.opencontainers.image.source=https://github.com/SENERGY-Platform/consumption_forecast_ml
WORKDIR /usr/src/app
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
# operator-lib is installed from its git repository.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
RUN uv sync --locked
COPY . .
# Ray propagates the `uv run` environment to the cluster; training relies on it.
CMD ["uv", "run", "--locked", "python", "-u", "main.py"]
