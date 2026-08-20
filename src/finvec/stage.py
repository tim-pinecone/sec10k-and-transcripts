"""Embed the corpus and write parquet into the per-year import layout.

The checkpoint unit is one source shard — one company, ~10k chunks. That is the unit
because it is the unit the source is published in, it is cheap to redo (seconds of
embedding), and a shard's output is a complete, self-contained set of files that can be
written atomically. Killing this mid-run and re-running the same command re-embeds at
most one shard.

A shard spans ~10-20 fiscal years, so one shard writes one parquet part into each of
those years' namespace directories, named by shard index. Files stay modest (~24 MB per
shard across all its years), and `finvec compact` can coalesce them into larger parts
before upload if fewer, bigger objects are wanted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .config import EMBED_DIMS, SEC_SHARD_COUNT
from .embed import Embedder
from .ids import UniquenessGuard
from .layout import staging_namespace_dir
from .limits import validate_metadata
from .merge import merge_records
from .progress import Checkpoint, Progress, atomic_path
from .sources import sec10k
from .sources.base import Record

# The three columns bulk import reads. Anything else in the file is ignored, so there
# is no reason to write more.
IMPORT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("values", pa.list_(pa.float32())),
        pa.field("metadata", pa.string()),
    ]
)

SHARD_PART = "shard-{shard:05d}.parquet"


def parse_shard_range(spec: str | None) -> list[int]:
    """`'0-4'`, `'7'`, or `'0-4,10,20-22'` -> explicit shard indices."""
    if not spec:
        return list(range(SEC_SHARD_COUNT))
    out: list[int] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lo, hi = piece.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(piece))
    bad = [s for s in out if not 0 <= s < SEC_SHARD_COUNT]
    if bad:
        raise ValueError(
            f"shards out of range 0..{SEC_SHARD_COUNT - 1}: {bad[:5]}"
        )
    return sorted(set(out))


def _write_part(
    path: Path, records: list[Record], vectors: list[list[float]]
) -> int:
    """Write one namespace's parquet part atomically.

    Atomic rename matters here specifically: `upload` decides a file is already
    uploaded by comparing sizes, so a truncated local part would be uploaded and then
    treated as complete.
    """
    table = pa.table(
        {
            "id": pa.array([r.id for r in records], pa.string()),
            "values": pa.array(vectors, pa.list_(pa.float32())),
            "metadata": pa.array(
                [json.dumps(r.metadata, separators=(",", ":")) for r in records],
                pa.string(),
            ),
        },
        schema=IMPORT_SCHEMA,
    )
    with atomic_path(path) as tmp:
        pq.write_table(table, tmp, compression="zstd")
    return path.stat().st_size


def stage_shard(
    shard: int,
    staging_dir: Path,
    embedder: Embedder,
    dataset: str = "sec",
    target_chars: int | None = None,
) -> dict[str, int]:
    """Merge, embed, and write one shard. Returns per-namespace record counts."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    kwargs = {"target_chars": target_chars} if target_chars else {}
    merged = list(merge_records(sec10k.iter_shard(shard), **kwargs))

    by_namespace: dict[str, list[Record]] = {}
    for record in merged:
        # Fill in the real token count the source column never provided, and validate
        # metadata before it can reach a parquet file.
        record.token_count = len(enc.encode(record.text))
        record.metadata["token_count"] = record.token_count
        validate_metadata(record.metadata, record.id)
        by_namespace.setdefault(record.namespace, []).append(record)

    counts: dict[str, int] = {}
    for namespace, records in sorted(by_namespace.items()):
        # IDs must be unique within a namespace or Pinecone silently overwrites.
        guard = UniquenessGuard(namespace)
        for record in records:
            guard.add(record.id)
        guard.raise_if_collisions()

        vectors = embedder.embed([(r.text, r.token_count) for r in records])
        if len(vectors) != len(records):
            raise RuntimeError(
                f"shard {shard} namespace {namespace}: {len(vectors)} vectors for "
                f"{len(records)} records"
            )
        if any(len(v) != EMBED_DIMS for v in vectors):
            raise RuntimeError(
                f"shard {shard}: embedder returned a vector of the wrong width"
            )

        directory = staging_namespace_dir(staging_dir, dataset, namespace)
        _write_part(directory / SHARD_PART.format(shard=shard), records, vectors)
        counts[namespace] = len(records)

    return counts


def stage(
    staging_dir: Path,
    state_dir: Path,
    shards: list[int],
    concurrency: int = 8,
    dataset: str = "sec",
    status_path: Path | None = None,
) -> Checkpoint:
    """Stage every shard not already done, reporting progress as it goes."""
    checkpoint = Checkpoint(f"stage-{dataset}", state_dir, flush_every=5)
    pending = checkpoint.pending([str(s) for s in shards])
    already = len(shards) - len(pending)

    if already:
        print(
            f"resuming: {already:,} of {len(shards):,} shards already staged "
            f"({checkpoint.totals('records'):,} records)",
            flush=True,
        )
    if not pending:
        print("nothing to stage — every requested shard is already done", flush=True)
        return checkpoint

    embedder = Embedder(concurrency=concurrency)
    prog = Progress(
        f"stage {dataset}",
        total=len(shards),
        done=already,
        status_path=status_path,
    )

    for key in pending:
        shard = int(key)
        counts = stage_shard(shard, staging_dir, embedder, dataset=dataset)
        checkpoint.mark(
            key,
            records=sum(counts.values()),
            namespaces=len(counts),
            tokens=embedder.tokens_embedded,
        )
        prog.advance(
            current=f"shard {shard}",
            records=f"{checkpoint.totals('records'):,}",
            tokens=f"{embedder.tokens_embedded:,}",
        )

    checkpoint.flush()
    prog.finish(
        f"{checkpoint.totals('records'):,} records, "
        f"{embedder.tokens_embedded:,} tokens embedded"
    )
    return checkpoint
