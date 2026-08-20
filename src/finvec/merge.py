"""Merge the source's over-fine chunks into retrieval-sized ones.

The source is chunked to a 512-*character* cap, so chunks average ~95 real tokens —
a sentence or two. That is too thin to retrieve against, and wasteful: the fixed
6,144-byte vector gets paid once per fragment.

Consecutive chunks are merged, but only within a run that shares the same
`(fiscal_year, is_boilerplate, is_table)` key. That keeps `is_boilerplate` and
`is_table` exactly boolean on the merged record, so a query filter means precisely what
the schema says — no `boilerplate_frac` thresholds.

Within a run, chunks accumulate greedily until the merged text reaches the character
target, then a new merged chunk starts. Runs can be long (17 chunks observed), and
merging a whole long run would produce a 1,600-token blob, so the target bounds the
result from above as well as below.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator

from .ids import sec_chunk_id, validate_id
from .sources.base import Record

# ~1,600 characters lands near 400 real tokens, given the corpus averages ~4 chars per
# token. Measured, not assumed: see `finvec profile`.
MERGE_TARGET_CHARS = 1600

# Paragraph break between merged fragments; tables read better with a single newline.
JOIN_PROSE = "\n\n"
JOIN_TABLE = "\n"


def _run_key(record: Record) -> tuple[Any, ...]:
    meta = record.metadata
    return (meta["ticker"], meta["fiscal_year"], meta["is_boilerplate"],
            meta["is_table"])


def _sort_key(record: Record) -> tuple[int, int]:
    return (record.metadata["fiscal_year"], record.metadata["chunk_id"])


def merge_records(
    records: Iterable[Record], target_chars: int = MERGE_TARGET_CHARS
) -> Iterator[Record]:
    """Yield merged records from one company's chunks.

    Input is sorted by `(fiscal_year, chunk_id)` rather than trusted to arrive in
    order: merging out-of-order fragments would splice unrelated passages together,
    and the cost of sorting one company's ~10k rows is nothing.
    """
    ordered = sorted(records, key=_sort_key)
    group: list[Record] = []
    group_chars = 0
    key: tuple[Any, ...] | None = None

    for record in ordered:
        record_key = _run_key(record)
        if key is not None and record_key != key and group:
            yield _emit(group)
            group, group_chars = [], 0
        key = record_key
        group.append(record)
        group_chars += len(record.text)
        if group_chars >= target_chars:
            yield _emit(group)
            group, group_chars = [], 0

    if group:
        yield _emit(group)


def _emit(group: list[Record]) -> Record:
    """Fold a run of fragments into one record.

    The merged ID uses the *first* fragment's `chunk_id`, which keeps IDs deterministic
    and preserves document order across merged records.
    """
    first = group[0]
    meta = dict(first.metadata)
    joiner = JOIN_TABLE if meta["is_table"] else JOIN_PROSE
    text = joiner.join(r.text for r in group)

    meta["text"] = text
    meta["source_chunk_ids"] = [str(r.metadata["chunk_id"]) for r in group]
    meta["merged_from"] = len(group)
    # The source's `token_count` column is a character count. Storing it under that
    # name would mislead anyone who later filters on it, so it becomes `char_count`
    # and `token_count` is left to the staging step to fill with a real count.
    meta["char_count"] = sum(r.metadata["token_count"] for r in group)
    meta.pop("token_count", None)

    return Record(
        id=validate_id(
            sec_chunk_id(meta["ticker"], meta["fiscal_year"], meta["chunk_id"])
        ),
        namespace=str(meta["fiscal_year"]),
        text=text,
        metadata=meta,
    )
