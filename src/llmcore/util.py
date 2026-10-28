from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def code_version() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=Path(__file__).parent
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git repository)"


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
