"""HTTP entry point: the ADK API server (sessions on Postgres, /run_sse) plus our routes.

Routes added on top of ADK's:
- GET /img/{key}     302 to a presigned URL (the browser loads images straight from S3)
- GET /api/health    Blender, storage, db, image backend, LLM
- GET /ui/           the React chat app (static build); `/` redirects here
ADK's own dev UI stays at /dev-ui/ for debugging.

Run: `python -m app.serve` (compose) or `uv run python -m app.serve` (dev).
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.sessions.database_session_service import DatabaseSessionService

from app.config import settings
from app.run_guard import PendingLookup, RunGuard, pending_requests
from app.services import services
from app.storage import seed_assets

log = logging.getLogger("trueshot")

AGENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agents")
HEALTH_TIMEOUT_S = 8.0


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    svc = services()
    wait_ready = getattr(svc.store, "wait_ready", None)
    if wait_ready is not None:
        await wait_ready()
    await svc.db.start()
    await seed_assets(svc.store, settings.assets_dir)
    yield
    await svc.db.stop()


async def _probe(check: Awaitable[dict[str, Any]]) -> dict[str, Any]:
    try:
        return await asyncio.wait_for(check, HEALTH_TIMEOUT_S)
    except TimeoutError:
        return {"ok": False, "detail": f"no answer within {HEALTH_TIMEOUT_S:.0f} s"}


async def health_report() -> dict[str, Any]:
    svc = services()
    names = ("blender", "storage", "db", "image_backend")
    results = await asyncio.gather(
        _probe(svc.blender.health()), _probe(svc.store.health()), _probe(svc.db.health()),
        _probe(svc.images.health()))
    report = dict(zip(names, results, strict=True))
    report["llm"] = {
        "ok": bool(settings.openai_api_key or settings.llm_api_base),
        "model": settings.llm_model,
        "endpoint": settings.llm_api_base or "openai",
    }
    return report


def _install_routes(app: FastAPI) -> None:
    @app.get("/img/{key:path}", include_in_schema=False)
    async def image(key: str) -> RedirectResponse:
        return RedirectResponse(services().store.public_url(key), status_code=302)

    @app.get("/api/health")
    async def api_health() -> dict[str, Any]:
        return await health_report()

    # Replace ADK's "/" -> /dev-ui/ redirect with one to our UI.
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) != "/"]

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    if os.path.isdir(settings.ui_dist):
        app.mount("/ui", StaticFiles(directory=settings.ui_dist, html=True), name="ui")

        @app.middleware("http")
        async def revalidate_ui_html(request: Request, call_next: Any) -> Any:
            # index.html names the hashed bundles, so it must never be served stale after a rebuild.
            response = await call_next(request)
            if request.url.path.startswith("/ui") and response.headers.get("content-type", "").startswith("text/html"):
                response.headers["Cache-Control"] = "no-cache"
            return response
    else:
        log.warning("UI build not found at %s; /ui/ is unavailable", settings.ui_dist)


def _pending_lookup() -> PendingLookup:
    """Unanswered approval ids of a session, read from ADK's own session tables."""
    sessions: DatabaseSessionService | None = None

    async def lookup(app_name: str, user_id: str, session_id: str) -> set[str] | None:
        nonlocal sessions
        if sessions is None:
            sessions = DatabaseSessionService(db_url=settings.session_service_uri)
        session = await sessions.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
        return None if session is None else pending_requests(session.events)

    return lookup


def create_app() -> FastAPI:
    app = get_fast_api_app(
        agents_dir=AGENTS_DIR,
        session_service_uri=settings.session_service_uri,
        use_local_storage=False,
        web=True,
        host="0.0.0.0",
        port=8000,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        lifespan=lifespan,
        max_llm_calls=settings.max_llm_calls,
    )
    _install_routes(app)
    app.add_middleware(RunGuard, pending=_pending_lookup())
    return app


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), log_level="info")


if __name__ == "__main__":
    main()
