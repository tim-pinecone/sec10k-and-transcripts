"""Deterministic record IDs.

Deterministic IDs make upserts idempotent. They do *not* make bulk imports
idempotent — an import can only target a namespace that does not yet exist — so the
unit of re-runnability for the import path is a whole year namespace.

Uniqueness is asserted rather than assumed: the transcripts source is not verified
unique on (symbol, year, quarter), and a collision would silently overwrite a record
instead of failing.
"""

from __future__ import annotations

import re

from .config import MAX_ID_CHARS

_SAFE = re.compile(r"[^A-Za-z0-9.\-_]")


def _slug(value: object) -> str:
    """Normalize a component so IDs stay ASCII-safe and stable across runs."""
    return _SAFE.sub("-", str(value).strip().upper())


def sec_chunk_id(ticker: str, fiscal_year: int, chunk_id: int) -> str:
    return f"{_slug(ticker)}_{int(fiscal_year)}_10K_CHUNK_{int(chunk_id)}"


def transcript_chunk_id(
    symbol: str, year: int, quarter: int, chunk_index: int
) -> str:
    return (
        f"{_slug(symbol)}_{int(year)}_Q{int(quarter)}_TRANSCRIPT_{int(chunk_index)}"
    )


def validate_id(record_id: str) -> str:
    if not record_id:
        raise ValueError("record ID is empty")
    if len(record_id) > MAX_ID_CHARS:
        raise ValueError(
            f"record ID is {len(record_id)} chars, over the {MAX_ID_CHARS} limit: "
            f"{record_id[:80]}…"
        )
    return record_id


class UniquenessGuard:
    """Detects ID collisions within a namespace during staging.

    Holding one set per namespace, not one for the whole corpus: a year namespace is
    the scope in which an ID must be unique, and per-namespace sets stay small enough
    to keep in memory.
    """

    def __init__(self, scope: str) -> None:
        self.scope = scope
        self._seen: set[str] = set()
        self.collisions: list[str] = []

    def add(self, record_id: str) -> bool:
        """Return True if the ID is new. Records the collision otherwise."""
        if record_id in self._seen:
            self.collisions.append(record_id)
            return False
        self._seen.add(record_id)
        return True

    def __len__(self) -> int:
        return len(self._seen)

    def raise_if_collisions(self) -> None:
        if self.collisions:
            sample = ", ".join(self.collisions[:5])
            raise ValueError(
                f"{len(self.collisions)} duplicate record IDs in namespace "
                f"{self.scope!r} — these would silently overwrite each other. "
                f"Sample: {sample}"
            )
