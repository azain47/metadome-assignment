"""The one place that builds adapters from settings.

Graph nodes, plugins and routes get their dependencies from `services()`. Tests call
`use(Services(...))` with fakes (MemoryStore, a fake Blender, the `fake` image backend).
"""

from dataclasses import dataclass

from app.blender import Blender, BlenderClient
from app.config import Settings, settings
from app.db import Database
from app.imagegen import ImageBackend, get_backend
from app.storage import ObjectStore, S3Store


@dataclass
class Services:
    settings: Settings
    store: ObjectStore
    blender: Blender
    db: Database
    images: ImageBackend


_current: Services | None = None


def build(cfg: Settings = settings) -> Services:
    return Services(
        settings=cfg,
        store=S3Store(
            endpoint=cfg.s3_endpoint, public_endpoint=cfg.s3_public_endpoint,
            access_key=cfg.s3_access_key, secret_key=cfg.s3_secret_key, bucket=cfg.s3_bucket),
        blender=BlenderClient(cfg.blender_mcp_url),
        db=Database(cfg.database_url),
        images=get_backend(cfg.image_backend, cfg.image_model or None),
    )


def use(svc: Services) -> None:
    global _current
    _current = svc


def services() -> Services:
    global _current
    if _current is None:
        _current = build()
    return _current
