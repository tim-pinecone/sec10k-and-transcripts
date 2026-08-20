"""Validators for Pinecone's hard payload limits.

Batch size is computed from measured serialized bytes rather than a guessed constant.
The plan's "350-500 vectors per payload" is only safe on gRPC at <=768 dims; on REST
at 1536 dims the real ceiling is closer to 90 records, because JSON floats cost ~13
bytes each against protobuf's 4.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Iterator

from .config import (
    EMBED_DIMS,
    MAX_ID_CHARS,
    MAX_METADATA_BYTES,
    MAX_UPSERT_BYTES,
    MAX_UPSERT_RECORDS,
)

# Bytes per float on the wire.
GRPC_BYTES_PER_FLOAT = 4
REST_BYTES_PER_FLOAT = 13  # conservative: JSON float repr, e.g. "-0.023456789,"


def metadata_bytes(metadata: dict[str, Any]) -> int:
    return len(json.dumps(metadata, separators=(",", ":")).encode("utf-8"))


def validate_metadata(metadata: dict[str, Any], record_id: str = "") -> dict[str, Any]:
    size = metadata_bytes(metadata)
    if size > MAX_METADATA_BYTES:
        raise ValueError(
            f"metadata for {record_id or '<record>'} is {size:,} bytes, over the "
            f"{MAX_METADATA_BYTES:,}-byte limit"
        )
    for key, value in metadata.items():
        if isinstance(value, dict):
            raise ValueError(
                f"metadata field {key!r} is an object; Pinecone metadata values must "
                f"be string, number, boolean, or list of strings"
            )
        if isinstance(value, list) and not all(isinstance(v, str) for v in value):
            raise ValueError(
                f"metadata field {key!r} is a list of non-strings; only lists of "
                f"strings are supported"
            )
    return metadata


def record_wire_bytes(
    record_id: str, metadata: dict[str, Any], dims: int = EMBED_DIMS, grpc: bool = True
) -> int:
    per_float = GRPC_BYTES_PER_FLOAT if grpc else REST_BYTES_PER_FLOAT
    return len(record_id.encode()) + metadata_bytes(metadata) + dims * per_float


def max_batch_records(
    avg_metadata_bytes: int,
    avg_id_bytes: int = 26,
    dims: int = EMBED_DIMS,
    grpc: bool = True,
    safety: float = 0.9,
) -> int:
    """Largest batch that stays under the 2 MB payload cap, with headroom."""
    per_record = record_wire_bytes(
        "x" * avg_id_bytes, {}, dims=dims, grpc=grpc
    ) + avg_metadata_bytes
    fits = int(MAX_UPSERT_BYTES * safety // max(per_record, 1))
    return max(1, min(fits, MAX_UPSERT_RECORDS))


def batched_by_bytes(
    records: Iterable[tuple[str, list[float], dict[str, Any]]],
    dims: int = EMBED_DIMS,
    grpc: bool = True,
    safety: float = 0.9,
) -> Iterator[list[tuple[str, list[float], dict[str, Any]]]]:
    """Yield batches bounded by both the byte cap and the record cap."""
    budget = int(MAX_UPSERT_BYTES * safety)
    batch: list[tuple[str, list[float], dict[str, Any]]] = []
    used = 0
    for rec in records:
        rid, _, meta = rec
        size = record_wire_bytes(rid, meta, dims=dims, grpc=grpc)
        if batch and (used + size > budget or len(batch) >= MAX_UPSERT_RECORDS):
            yield batch
            batch, used = [], 0
        batch.append(rec)
        used += size
    if batch:
        yield batch


def index_size_bytes(
    n_records: int, avg_metadata_bytes: int, avg_id_bytes: int = 26,
    dims: int = EMBED_DIMS,
) -> int:
    """Pinecone's index-size formula: records x (id + metadata + dims x 4 bytes)."""
    return n_records * (avg_id_bytes + avg_metadata_bytes + dims * 4)


__all__ = [
    "MAX_ID_CHARS",
    "batched_by_bytes",
    "index_size_bytes",
    "max_batch_records",
    "metadata_bytes",
    "record_wire_bytes",
    "validate_metadata",
]
