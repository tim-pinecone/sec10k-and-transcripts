"""Convert staged Parquet parts into the gzipped JSONL that FTS import consumes.

Parquet stays the durable local artifact — typed, columnar, cheap to re-read — and
JSONL is generated as a derived transport format at upload time. That split exists for
a mundane reason: a 1536-dim vector costs ~6 KB as binary float32 and ~23 KB as JSON
floats, so holding both formats for the full corpus would need ~67 GB of disk against
the ~52 GB available.

Floats are rounded to 6 decimal places, which cuts about a third of the bytes. With
components on the order of 1e-2, a 1e-6 absolute rounding step is far below anything
that affects cosine ranking.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from .fts import TEXT_FIELD, VECTOR_FIELD

FLOAT_DP = 6


def document_from_row(
    record_id: str, values: list[float], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Build one FTS document.

    The FTS text lives in a schema-declared top-level field, not inside a metadata
    blob, so `text` is lifted out of the Parquet metadata JSON. Everything left over
    stays top-level too and is auto-indexed as filterable metadata.
    """
    meta = dict(metadata)
    text = meta.pop("text", "")
    doc: dict[str, Any] = {
        "_id": record_id,
        TEXT_FIELD: text,
        VECTOR_FIELD: [round(float(v), FLOAT_DP) for v in values],
    }
    for key, value in meta.items():
        # Reserved prefixes are rejected by the server; catching it here is cheaper
        # than a failed 400 MB import.
        if key.startswith(("_", "$")):
            raise ValueError(
                f"metadata field {key!r} on {record_id} starts with a reserved "
                f"character; FTS rejects field names beginning with '_' or '$'"
            )
        if isinstance(value, list) and not all(isinstance(v, str) for v in value):
            # "An array of numbers in an undeclared field is rejected rather than
            # stored as metadata."
            raise ValueError(
                f"metadata field {key!r} on {record_id} is an array of non-strings; "
                f"FTS rejects those rather than storing them"
            )
        doc[key] = value
    return doc


def iter_documents(parquet_path: Path) -> Iterator[dict[str, Any]]:
    """Stream documents from one staged Parquet part."""
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(batch_size=512):
        rows = batch.to_pylist()
        for row in rows:
            yield document_from_row(
                row["id"], row["values"], json.loads(row["metadata"])
            )


def write_jsonl_gz(parquet_path: Path, out_path: Path) -> tuple[int, int]:
    """Write one Parquet part out as `.jsonl.gz`. Returns (documents, bytes).

    `mtime=0` keeps the gzip output byte-identical across runs for identical input,
    so a re-converted part matches what was already uploaded.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.GzipFile(out_path, "wb", compresslevel=6, mtime=0) as raw:
        for doc in iter_documents(parquet_path):
            raw.write(json.dumps(doc, separators=(",", ":")).encode("utf-8"))
            raw.write(b"\n")
            count += 1
    return count, out_path.stat().st_size


def jsonl_key_for(parquet_relative: str) -> str:
    """`…/part-00000.parquet` -> `…/part-00000.jsonl.gz`."""
    if not parquet_relative.endswith(".parquet"):
        raise ValueError(f"expected a .parquet path, got {parquet_relative!r}")
    return parquet_relative[: -len(".parquet")] + ".jsonl.gz"
