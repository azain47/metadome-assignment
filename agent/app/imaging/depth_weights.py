"""Depth Anything V2 Small weights (Apache-2.0): the ONNX export by onnx-community, pinned to a
revision and checked by SHA-256. Standard library only, so the agent image can fetch the weights
in their own cached layer before the rest of the code is copied:

    python app/imaging/depth_weights.py /srv/models
"""

import hashlib
import os
import sys
import tempfile
import urllib.request

REPO = "onnx-community/depth-anything-v2-small"
REVISION = "4472b7362082ad9968fee890ca0f1e5aca36b93d"
FILE = "onnx/model.onnx"
SHA256 = "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c"
URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILE}"
NAME = "depth-anything-v2-small.onnx"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_weights(models_dir: str) -> str:
    """Path of the weights in `models_dir`, downloaded and verified on first use."""
    folder = os.path.expanduser(models_dir)
    path = os.path.join(folder, NAME)
    if os.path.exists(path):
        return path
    os.makedirs(folder, exist_ok=True)
    fd, part = tempfile.mkstemp(dir=folder, suffix=".part")
    os.close(fd)
    try:
        urllib.request.urlretrieve(URL, part)
        if (got := _sha256(part)) != SHA256:
            raise RuntimeError(f"{URL}: sha256 {got}, expected {SHA256}")
        os.chmod(part, 0o644)  # mkstemp makes it owner-only
        os.replace(part, path)
    finally:
        if os.path.exists(part):
            os.remove(part)
    return path


if __name__ == "__main__":
    print(ensure_weights(sys.argv[1] if len(sys.argv) > 1 else "~/.cache/trueshot/models"))
