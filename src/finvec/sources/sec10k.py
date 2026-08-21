"""Reader for `astr010/sec-10k-lsh-chunks`.

13,578,263 pre-chunked 10-K passages across 1,380 companies, 2004-2025. The dataset
is published as 1,380 parquet shards (one per company), which makes a shard the
natural checkpoint unit: one shard is ~10k records, cheap to redo and small enough
that losing one to a crash costs seconds.

Field names in this dataset already match our metadata schema exactly - unlike the
transcripts source - so no renaming is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from ..config import SEC_DATASET, SEC_SHARD_COUNT
from ..ids import sec_chunk_id, validate_id
from .base import Record

# Columns read from the source, keyed by the canonical name this code uses, with the
# aliases actually observed in the published shards.
#
# The dataset is NOT schema-uniform: shard 1184 of 1,380 names its chunk ordinal
# `chunk_index` where every other shard calls it `chunk_id`. Requesting a fixed column
# list therefore fails on that one shard with
#   ArrowInvalid: No match for FieldRef.Name(chunk_id)
# which killed a staging run at 86%. Columns are resolved against each shard's actual
# schema instead, so a single renamed field in one shard cannot take down the job.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "ticker": ("ticker",),
    "cik": ("cik",),
    "sic": ("sic",),
    "sic_description": ("sic_description",),
    "fiscal_year": ("fiscal_year", "year"),
    "chunk_id": ("chunk_id", "chunk_index"),
    "is_table": ("is_table",),
    "is_boilerplate": ("is_boilerplate",),
    "token_count": ("token_count", "char_count"),
    "text": ("text",),
}

COLUMNS = list(COLUMN_ALIASES)


def resolve_columns(available: list[str]) -> dict[str, str]:
    """Map canonical field -> the column name this shard actually uses.

    Raises rather than silently dropping a field: a missing `text` or `fiscal_year`
    would produce records that are wrong rather than absent, which is worse than a
    hard failure.
    """
    present = set(available)
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for canonical, aliases in COLUMN_ALIASES.items():
        match = next((a for a in aliases if a in present), None)
        if match is None:
            missing.append(canonical)
        else:
            resolved[canonical] = match
    if missing:
        raise ValueError(
            f"shard schema is missing required field(s) {missing} under any known "
            f"alias. Columns present: {sorted(present)}. Add the new name to "
            f"COLUMN_ALIASES in sources/sec10k.py."
        )
    return resolved

PARQUET_URL = (
    "https://huggingface.co/api/datasets/{ds}/parquet/default/train/{shard}.parquet"
)


def shard_urls(shards: list[int] | None = None) -> list[tuple[int, str]]:
    """(shard index, URL) pairs. Shards are numbered 0..1379."""
    idxs = shards if shards is not None else range(SEC_SHARD_COUNT)
    return [(i, PARQUET_URL.format(ds=SEC_DATASET, shard=i)) for i in idxs]


def to_record(row: dict[str, Any]) -> Record:
    """Map one source row to a staged record.

    `is_boilerplate` is carried through rather than used to drop the row: keeping it
    filterable means a query can exclude boilerplate, search it deliberately, or
    ignore the distinction, instead of that choice being frozen at ingest time.
    """
    year = int(row["fiscal_year"])
    rid = validate_id(sec_chunk_id(row["ticker"], year, row["chunk_id"]))
    text = row["text"] or ""
    return Record(
        id=rid,
        namespace=str(year),
        text=text,
        metadata={
            "ticker": str(row["ticker"]),
            "cik": str(row["cik"]),
            "fiscal_year": year,
            "sic": str(row["sic"]),
            "sic_description": str(row["sic_description"] or ""),
            "chunk_id": int(row["chunk_id"]),
            "is_table": bool(row["is_table"]),
            "is_boilerplate": bool(row["is_boilerplate"]),
            "token_count": int(row["token_count"]),
            "text": text,
        },
        token_count=int(row["token_count"]),
    )


def shard_path(shard: int) -> "Path":
    """Local path of a shard, downloading it to the cache if needed."""
    from ._cache import fetch

    _, url = shard_urls([shard])[0]
    return fetch(url, f"sec10k/{shard}.parquet")


def shard_schema(shard: int) -> list[str]:
    """Column names as this shard actually publishes them."""
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(shard_path(shard)).schema_arrow.names)


def shard_table(shard: int):
    """Read one shard, normalising its columns to the canonical names."""
    import pyarrow.parquet as pq  # imported lazily to keep `--help` fast

    path = shard_path(shard)
    parquet = pq.ParquetFile(path)
    resolved = resolve_columns(list(parquet.schema_arrow.names))
    table = pq.read_table(path, columns=list(resolved.values()))
    # Rename back to canonical so downstream code never sees the shard's variant.
    return table.rename_columns([
        canonical for canonical, _ in sorted(
            resolved.items(), key=lambda kv: table.column_names.index(kv[1])
        )
    ])


def iter_shard(shard: int) -> Iterator[Record]:
    """Stream one shard's records. Downloads on first read, cached after."""
    for batch in shard_table(shard).to_batches(max_chunksize=2048):
        for row in batch.to_pylist():
            yield to_record(row)
