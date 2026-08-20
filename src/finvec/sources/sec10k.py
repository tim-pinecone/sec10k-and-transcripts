"""Reader for `astr010/sec-10k-lsh-chunks`.

13,578,263 pre-chunked 10-K passages across 1,380 companies, 2004-2025. The dataset
is published as 1,380 parquet shards (one per company), which makes a shard the
natural checkpoint unit: one shard is ~10k records, cheap to redo and small enough
that losing one to a crash costs seconds.

Field names in this dataset already match our metadata schema exactly - unlike the
transcripts source - so no renaming is needed.
"""

from __future__ import annotations

from typing import Any, Iterator

from ..config import SEC_DATASET, SEC_SHARD_COUNT
from ..ids import sec_chunk_id, validate_id
from .base import Record

# Columns read from the source. `text` is both the embedding input and, retained in
# metadata, what makes a search result readable without a second lookup.
COLUMNS = [
    "ticker",
    "cik",
    "sic",
    "sic_description",
    "fiscal_year",
    "chunk_id",
    "is_table",
    "is_boilerplate",
    "token_count",
    "text",
]

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


def shard_table(shard: int, columns: list[str] | None = None):
    """Read one shard into an Arrow table, downloading it to the cache if needed."""
    import pyarrow.parquet as pq  # imported lazily to keep `--help` fast

    from ._cache import fetch

    _, url = shard_urls([shard])[0]
    path = fetch(url, f"sec10k/{shard}.parquet")
    return pq.read_table(path, columns=columns or COLUMNS)


def iter_shard(shard: int) -> Iterator[Record]:
    """Stream one shard's records. Downloads on first read, cached after."""
    for batch in shard_table(shard).to_batches(max_chunksize=2048):
        for row in batch.to_pylist():
            yield to_record(row)
