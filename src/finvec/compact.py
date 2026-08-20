"""Coalesce per-shard parquet parts into fewer, larger ones.

Staging writes one part per (shard, year), which keeps the checkpoint unit small but
leaves each year holding hundreds of ~1 MB files. Import tolerates that — the caps are
100,000 files and 10 GB per file — but fewer, larger objects mean fewer S3 PUTs and
less per-file parquet overhead.

Purely derived and repeatable: it reads staged parquet and writes staged parquet, with
no API calls and nothing to pay for. Safe to skip, and safe to re-run.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from .layout import PART_TEMPLATE
from .progress import Progress, atomic_path

TARGET_PART_BYTES = 400 * 1024 * 1024


def compact_namespace(
    directory: Path, target_bytes: int = TARGET_PART_BYTES, keep_inputs: bool = False
) -> tuple[int, int]:
    """Rewrite one namespace directory into `part-*.parquet` files.

    Returns (parts written, inputs consumed). Inputs are deleted only after every
    output part has been renamed into place, so an interrupted compaction leaves the
    originals intact and the whole thing can simply be re-run.
    """
    inputs = sorted(directory.glob("shard-*.parquet"))
    if not inputs:
        return (0, 0)

    # Numbering continues past any parts already here. Compaction can legitimately run
    # more than once on the same namespace — stage gets interrupted, compact runs,
    # stage resumes and adds more shard files — and restarting at part-00000 would
    # overwrite the earlier part with only the new records, silently losing the rest.
    existing = sorted(directory.glob("part-*.parquet"))
    next_index = 0
    if existing:
        next_index = max(
            int(p.stem.rsplit("-", 1)[1]) for p in existing
        ) + 1

    groups: list[list[Path]] = [[]]
    running = 0
    for path in inputs:
        size = path.stat().st_size
        if groups[-1] and running + size > target_bytes:
            groups.append([])
            running = 0
        groups[-1].append(path)
        running += size

    written: list[Path] = []
    for offset, group in enumerate(groups):
        out = directory / PART_TEMPLATE.format(index=next_index + offset)
        # Streamed one input file at a time rather than concatenated in memory: a
        # 400 MB target would otherwise mean holding ~1 GB of Arrow buffers.
        with atomic_path(out) as tmp:
            writer = None
            try:
                for path in group:
                    table = pq.read_table(path)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            tmp, table.schema, compression="zstd"
                        )
                    writer.write_table(table)
            finally:
                if writer is not None:
                    writer.close()
        written.append(out)

    if not keep_inputs:
        for path in inputs:
            path.unlink()

    return (len(written), len(inputs))


def compact(
    staging_dir: Path,
    dataset: str,
    target_bytes: int = TARGET_PART_BYTES,
    keep_inputs: bool = False,
    status_path: Path | None = None,
) -> dict[str, tuple[int, int]]:
    directories = sorted(
        p for p in (Path(staging_dir) / dataset).glob("import-*/*") if p.is_dir()
    )
    prog = Progress(f"compact {dataset}", total=len(directories),
                    status_path=status_path)
    results: dict[str, tuple[int, int]] = {}
    for directory in directories:
        namespace = directory.name
        results[namespace] = compact_namespace(directory, target_bytes, keep_inputs)
        prog.advance(current=f"namespace {namespace}")
    prog.finish(
        f"{sum(w for w, _ in results.values())} parts from "
        f"{sum(c for _, c in results.values())} shard files"
    )
    return results
