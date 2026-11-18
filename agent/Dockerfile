# Build context: repository root (see docker-compose.yml).

# --- React chat UI (static build, served by the agent at /ui/) ---
FROM node:22-alpine AS ui
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY ui/ ./
RUN npm run build

# --- Agent service ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/srv/agent \
    PATH=/srv/agent/.venv/bin:$PATH \
    UI_DIST=/srv/ui/dist \
    MODELS_DIR=/srv/models
WORKDIR /srv/agent
# Depth Anything V2 Small weights (pinned, SHA-256 checked) in their own layer: code changes
# never download them again, and the running stack needs no network for them.
COPY agent/app/imaging/depth_weights.py /tmp/depth_weights.py
RUN python /tmp/depth_weights.py /srv/models && rm /tmp/depth_weights.py
COPY agent/pyproject.toml agent/uv.lock ./
RUN uv sync --frozen --no-dev
COPY agent/app ./app
COPY agent/agents ./agents
COPY --from=ui /ui/dist /srv/ui/dist

EXPOSE 8000
CMD ["python", "-m", "app.serve"]
