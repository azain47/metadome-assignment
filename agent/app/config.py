"""Settings read from the environment (see .env.example for every variable)."""

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


@dataclass(frozen=True)
class Settings:
    app_name: str = "trueshot"
    user_id: str = field(default_factory=lambda: _env("USER_ID", "demo"))

    # LLM endpoint (OpenAI by default; any OpenAI-compatible server via LLM_API_BASE)
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-5.6-terra"))
    llm_api_base: str = field(default_factory=lambda: _env("LLM_API_BASE"))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    # USD per 1M tokens, for the spend meter (defaults: gpt-5.6-terra list prices)
    llm_price_in: float = field(default_factory=lambda: _float("LLM_PRICE_IN_PER_M", 2.0))
    llm_price_cached: float = field(default_factory=lambda: _float("LLM_PRICE_CACHED_PER_M", 0.2))
    llm_price_out: float = field(default_factory=lambda: _float("LLM_PRICE_OUT_PER_M", 12.0))

    # Image generation (plates)
    image_backend: str = field(default_factory=lambda: _env("IMAGE_BACKEND", "openai"))
    image_model: str = field(default_factory=lambda: _env("IMAGE_MODEL"))
    plate_candidates: int = field(default_factory=lambda: _int("PLATE_CANDIDATES", 2))

    # Budgets, enforced in code (graph dispatch/decide and plugins)
    image_budget_per_shot: int = field(default_factory=lambda: _int("IMAGE_BUDGET_PER_SHOT", 8))
    loops_per_shot: int = field(default_factory=lambda: _int("LOOPS_PER_SHOT", 6))
    max_llm_calls: int = field(default_factory=lambda: _int("MAX_LLM_CALLS", 120))
    spend_cap_usd: float = field(default_factory=lambda: _float("SPEND_CAP_USD", 45.0))

    # Render settings for the final car pass
    render_samples: int = field(default_factory=lambda: _int("RENDER_SAMPLES", 128))

    database_url: str = field(default_factory=lambda: _env(
        "DATABASE_URL", "postgresql://trueshot:trueshot-dev-pw@localhost:5433/trueshot"))

    s3_endpoint: str = field(default_factory=lambda: _env("S3_ENDPOINT", "http://127.0.0.1:8333"))
    s3_public_endpoint: str = field(default_factory=lambda: _env("S3_PUBLIC_ENDPOINT", "http://127.0.0.1:8333"))
    s3_access_key: str = field(default_factory=lambda: _env("S3_ACCESS_KEY", "trueshot"))
    s3_secret_key: str = field(default_factory=lambda: _env("S3_SECRET_KEY"))
    s3_bucket: str = field(default_factory=lambda: _env("S3_BUCKET", "trueshot"))

    blender_mcp_url: str = field(default_factory=lambda: _env("BLENDER_MCP_URL", "http://127.0.0.1:8765/"))
    assets_dir: str = field(default_factory=lambda: _env("ASSETS_DIR", "../assets"))
    outputs_dir: str = field(default_factory=lambda: _env("OUTPUTS_DIR", "../outputs"))
    ui_dist: str = field(default_factory=lambda: _env("UI_DIST", "../ui/dist"))
    # Depth Anything weights (downloaded and checked on first use; baked into the agent image)
    models_dir: str = field(default_factory=lambda: _env("MODELS_DIR", "~/.cache/trueshot/models"))

    @property
    def session_service_uri(self) -> str:
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)


settings = Settings()
