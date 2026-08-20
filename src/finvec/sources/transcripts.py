"""Reader and chunker for `Bose345/sp500_earnings_transcripts`.

33,362 earnings calls, 2005-2025. Three things about this source differ from what the
original plan assumed:

1. Field names are `symbol`, `year`, `date` - not `ticker`, `fiscal_year`, `call_date`.
2. `structured_content` is already a list of `{speaker, text}` turns, so speaker
   attribution is exact. Chunks are cut *within* a turn and never across one, which
   means every chunk has exactly one correct speaker.
3. It is a single 1.82 GB parquet whose row groups exceed Hugging Face's 300 MB scan
   limit, so `streaming=True` does not stream cheaply. Download once and read
   row-group-wise; budget ~2 GB of RAM.

`company_id` is float64 in the source and may be NaN, so it is cast defensively.
"""

from __future__ import annotations

import math
from typing import Any, Iterator, Sequence

from ..config import CHUNK_OVERLAP_TOKENS, CHUNK_TOKENS
from ..ids import transcript_chunk_id, validate_id
from .base import Record

COLUMNS = [
    "symbol",
    "quarter",
    "year",
    "date",
    "structured_content",
    "company_name",
    "company_id",
]


def _clean_float(value: Any) -> float | None:
    """`company_id` is float64 and may be NaN; NaN is not valid JSON metadata."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def chunk_turn(
    tokens: Sequence[int],
    size: int = CHUNK_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> Iterator[tuple[int, int]]:
    """Yield (start, end) token windows over a single speaker turn.

    A turn shorter than `size` yields exactly one window, so short exchanges - most of
    the Q&A section - stay intact rather than being padded or merged with a neighbour.
    """
    if not tokens:
        return
    if overlap >= size:
        raise ValueError(f"overlap {overlap} must be smaller than size {size}")
    step = size - overlap
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + size, n)
        yield start, end
        if end == n:
            return
        start += step


def iter_call_records(
    row: dict[str, Any], encoder: Any
) -> Iterator[Record]:
    """Chunk one earnings call into per-speaker-turn records.

    `chunk_index` runs across the whole call, not per turn, so it is stable and unique
    within the call - which is what the record ID depends on.
    """
    symbol = str(row["symbol"])
    year = int(row["year"])
    quarter = int(row["quarter"])
    namespace = str(year)
    company_name = str(row.get("company_name") or "")
    call_date = str(row.get("date") or "")
    company_id = _clean_float(row.get("company_id"))

    turns = row.get("structured_content") or []
    chunk_index = 0
    for turn_index, turn in enumerate(turns):
        speaker = str((turn or {}).get("speaker") or "Unknown")
        text = ((turn or {}).get("text") or "").strip()
        if not text:
            continue
        tokens = encoder.encode(text)
        for start, end in chunk_turn(tokens):
            chunk_text = encoder.decode(tokens[start:end]).strip()
            if not chunk_text:
                continue
            rid = validate_id(
                transcript_chunk_id(symbol, year, quarter, chunk_index)
            )
            metadata: dict[str, Any] = {
                "ticker": symbol,
                "company_name": company_name,
                "fiscal_year": year,
                "quarter": quarter,
                "call_date": call_date,
                "speaker": speaker,
                "turn_index": turn_index,
                "chunk_index": chunk_index,
                "token_count": end - start,
                "text": chunk_text,
            }
            if company_id is not None:
                metadata["company_id"] = company_id
            yield Record(
                id=rid,
                namespace=namespace,
                text=chunk_text,
                metadata=metadata,
                token_count=end - start,
            )
            chunk_index += 1
