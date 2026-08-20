"""Cached HTTP fetch for source parquet shards.

pyarrow cannot read an https:// URI directly, and re-downloading a shard on every run
would make resuming pointless, so shards land in a local cache keyed by URL path.
"""

from __future__ import annotations

from pathlib import Path

import requests

from ..progress import atomic_path

CACHE_DIR = Path("hf_cache")


def fetch(url: str, name: str, cache_dir: Path | None = None) -> Path:
    """Download `url` to the cache unless already present. Returns the local path."""
    root = Path(cache_dir or CACHE_DIR)
    dest = root / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with atomic_path(dest) as tmp:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for block in resp.iter_content(chunk_size=1 << 20):
                    fh.write(block)
    return dest
