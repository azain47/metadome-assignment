"""Object storage. Images and other blobs live here; Postgres and ADK events hold keys only.

Interface (`ObjectStore`): put / get / size / list / public_url, plus put_json / get_json.
Adapters: `S3Store` (SeaweedFS in compose) and `MemoryStore` (tests).
"""

import asyncio
import json
import logging
import mimetypes
import os
from typing import Any, Protocol

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)


def content_type_for(key: str) -> str:
    if key.endswith(".glb"):
        return "model/gltf-binary"
    if key.endswith(".exr"):
        return "image/x-exr"
    return mimetypes.guess_type(key)[0] or "application/octet-stream"


class ObjectStore(Protocol):
    async def put(self, key: str, data: bytes, content_type: str | None = None) -> str: ...
    async def get(self, key: str) -> bytes: ...
    async def size(self, key: str) -> int | None:
        """Object size in bytes, or None when the key does not exist."""
    async def list(self, prefix: str) -> list[str]: ...
    def public_url(self, key: str, expires_s: int = 3600) -> str:
        """URL the browser and the host-side Blender can GET directly."""
    async def health(self) -> dict[str, Any]: ...


async def put_json(store: ObjectStore, key: str, value: Any) -> str:
    return await store.put(key, json.dumps(value, indent=1, default=str).encode(), "application/json")


async def get_json(store: ObjectStore, key: str) -> Any:
    return json.loads(await store.get(key))


class S3Store:
    """S3 adapter. Two clients: in-network endpoint for I/O, public endpoint for presigned URLs.

    Presigned URLs are signed for the exact public host string (SigV4 signs the Host header),
    so S3_PUBLIC_ENDPOINT must be what the browser/Blender use (e.g. http://localhost:8333).
    """

    _CONFIG = Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
        retries={"max_attempts": 5, "mode": "standard"},
    )

    def __init__(self, *, endpoint: str, public_endpoint: str, access_key: str, secret_key: str, bucket: str) -> None:
        self.bucket = bucket
        self._endpoint = endpoint
        self._io = self._client(endpoint, access_key, secret_key)
        self._signer = self._client(public_endpoint, access_key, secret_key)

    @classmethod
    def _client(cls, endpoint: str, access_key: str, secret_key: str) -> Any:
        return boto3.client(
            "s3", endpoint_url=endpoint, aws_access_key_id=access_key, aws_secret_access_key=secret_key,
            region_name="us-east-1", config=cls._CONFIG)

    async def put(self, key: str, data: bytes, content_type: str | None = None) -> str:
        await asyncio.to_thread(
            self._io.put_object, Bucket=self.bucket, Key=key, Body=data,
            ContentType=content_type or content_type_for(key))
        return key

    async def get(self, key: str) -> bytes:
        def _get() -> bytes:
            return self._io.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        return await asyncio.to_thread(_get)

    async def size(self, key: str) -> int | None:
        def _head() -> int | None:
            try:
                return int(self._io.head_object(Bucket=self.bucket, Key=key)["ContentLength"])
            except ClientError:
                return None
        return await asyncio.to_thread(_head)

    async def list(self, prefix: str) -> list[str]:
        def _list() -> list[str]:
            keys: list[str] = []
            for page in self._io.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
                keys.extend(obj["Key"] for obj in page.get("Contents", []))
            return keys
        return await asyncio.to_thread(_list)

    def public_url(self, key: str, expires_s: int = 3600) -> str:
        return self._signer.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires_s)

    async def health(self) -> dict[str, Any]:
        try:
            await asyncio.to_thread(self._io.head_bucket, Bucket=self.bucket)
            return {"ok": True, "detail": f"{self._endpoint}/{self.bucket}"}
        except Exception as ex:  # noqa: BLE001
            return {"ok": False, "detail": str(ex)[:200]}

    async def wait_ready(self, timeout_s: float = 60.0) -> None:
        """`weed mini` answers /healthz before it creates the bucket; wait for the bucket."""
        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            try:
                await asyncio.to_thread(self._io.head_bucket, Bucket=self.bucket)
                return
            except Exception:  # noqa: BLE001
                try:
                    await asyncio.to_thread(self._io.create_bucket, Bucket=self.bucket)
                    return
                except Exception:  # noqa: BLE001
                    if asyncio.get_running_loop().time() > deadline:
                        raise
            await asyncio.sleep(1.0)


class MemoryStore:
    """In-memory adapter for tests."""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}

    async def put(self, key: str, data: bytes, content_type: str | None = None) -> str:
        self.objects[key] = (data, content_type or content_type_for(key))
        return key

    async def get(self, key: str) -> bytes:
        return self.objects[key][0]

    async def size(self, key: str) -> int | None:
        return len(self.objects[key][0]) if key in self.objects else None

    async def list(self, prefix: str) -> list[str]:
        return sorted(k for k in self.objects if k.startswith(prefix))

    def public_url(self, key: str, expires_s: int = 3600) -> str:
        return f"memory://{key}"

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "detail": "memory"}


# Folders under the assets directory that are mirrored to the store at startup.
SEEDED_FOLDERS = ("cars", "hdri", "refs")


async def seed_assets(store: ObjectStore, assets_dir: str) -> dict[str, int]:
    """Mirror assets/<cars|hdri|refs>/** to keys of the same path. Same-size objects are skipped."""
    uploaded: dict[str, int] = {}
    for folder in SEEDED_FOLDERS:
        root = os.path.join(assets_dir, folder)
        for dirpath, _dirs, files in os.walk(root):
            for name in sorted(files):
                if name.startswith(".") or name.endswith(".md"):
                    continue
                path = os.path.join(dirpath, name)
                key = f"{folder}/{os.path.relpath(path, root).replace(os.sep, '/')}"
                if await store.size(key) == os.path.getsize(path):
                    continue
                with open(path, "rb") as fh:
                    await store.put(key, fh.read())
                uploaded[folder] = uploaded.get(folder, 0) + 1
    log.info("asset seeding uploaded %s", uploaded or "nothing new")
    return uploaded
