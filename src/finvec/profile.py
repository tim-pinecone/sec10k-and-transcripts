"""Measure the corpus and project cost before spending anything.

Everything downstream of this - embeddings, import, storage - costs real money in
proportion to numbers this module measures rather than guesses: how many tokens the
corpus actually holds, how big a record's metadata really is, and what fraction is
LSH-flagged boilerplate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import (
    EMBED_DIMS,
    SEC_SHARD_COUNT,
    USD_PER_GB_IMPORT,
    USD_PER_GB_MONTH_STORAGE,
    USD_PER_MILLION_EMBED_TOKENS,
    USD_PER_MILLION_WRITE_UNITS,
)
from .limits import index_size_bytes, max_batch_records, metadata_bytes
from .progress import Progress, atomic_write_bytes
from .sources import sec10k

SEC_TOTAL_ROWS = 13_578_263
# The corpus spans fiscal years 2004-2025, so the SEC index has 22 year namespaces.
# A shard sample only touches the years one company filed in, so the namespace count
# must come from the known span rather than from the sample.
SEC_YEAR_NAMESPACES = 22
# Pinecone bills in decimal GB, not GiB: its own worked example is
# 500,000 x (8 + 768x4 + 500) = 1,790,000,000 bytes = "1.79 GB".
GB = 1_000_000_000


@dataclass
class CorpusProfile:
    """What a sample of shards implies about the whole corpus."""

    shards_sampled: int
    rows_sampled: int
    total_rows: int
    avg_claimed_tokens: float
    avg_real_tokens: float
    avg_text_bytes: float
    avg_metadata_bytes: float
    avg_id_bytes: float
    boilerplate_frac: float
    table_frac: float
    mean_flag_run_length: float
    years: dict[str, int]

    @property
    def est_total_tokens(self) -> float:
        return self.avg_real_tokens * self.total_rows

    @property
    def est_index_bytes(self) -> int:
        return index_size_bytes(
            self.total_rows,
            int(self.avg_metadata_bytes),
            int(self.avg_id_bytes),
            dims=EMBED_DIMS,
        )


def sample_shards(n: int) -> list[int]:
    """Evenly spaced shard indices - deterministic, and spread across the alphabet
    of tickers rather than clustered at one end."""
    n = max(1, min(n, SEC_SHARD_COUNT))
    step = SEC_SHARD_COUNT / n
    return sorted({int(i * step) for i in range(n)})


def profile_sec(n_shards: int = 12, status_path: Path | None = None) -> CorpusProfile:
    """Sample shards and measure what the corpus actually contains.

    Real token counts are measured with cl100k_base rather than trusted from the
    source's `token_count` column, because that column is effectively a character
    count capped at 512 and overstates true tokens by roughly 4x. Every downstream
    cost estimate depends on getting this right.
    """
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    shards = sample_shards(n_shards)
    prog = Progress("profile sec", total=len(shards), status_path=status_path)

    rows = claimed = real = text_bytes = meta_bytes = id_bytes = 0
    boiler = tables = 0
    years: dict[str, int] = {}
    runs: list[int] = []

    for shard in shards:
        run_key, run_len = None, 0
        for rec in sec10k.iter_shard(shard):
            rows += 1
            meta = rec.metadata
            claimed += rec.token_count
            real += len(enc.encode(meta["text"]))
            text_bytes += len(meta["text"].encode())
            meta_bytes += metadata_bytes(meta)
            id_bytes += len(rec.id.encode())
            boiler += bool(meta["is_boilerplate"])
            tables += bool(meta["is_table"])
            years[rec.namespace] = years.get(rec.namespace, 0) + 1

            # How many consecutive chunks share the same filter flags, which bounds
            # how far chunks can be merged without blurring is_boilerplate/is_table.
            key = (meta["fiscal_year"], meta["is_boilerplate"], meta["is_table"])
            if key == run_key:
                run_len += 1
            else:
                if run_key is not None:
                    runs.append(run_len)
                run_key, run_len = key, 1
        if run_key is not None:
            runs.append(run_len)
        prog.advance(current=f"shard {shard}", rows=f"{rows:,}")
    prog.finish(f"{rows:,} rows sampled")

    if rows == 0:
        raise SystemExit("sampled zero rows - check network access to Hugging Face")

    return CorpusProfile(
        shards_sampled=len(shards),
        rows_sampled=rows,
        total_rows=SEC_TOTAL_ROWS,
        avg_claimed_tokens=claimed / rows,
        avg_real_tokens=real / rows,
        avg_text_bytes=text_bytes / rows,
        avg_metadata_bytes=meta_bytes / rows,
        avg_id_bytes=id_bytes / rows,
        boilerplate_frac=boiler / rows,
        table_frac=tables / rows,
        mean_flag_run_length=(sum(runs) / len(runs)) if runs else 1.0,
        years=dict(sorted(years.items())),
    )


def render(p: CorpusProfile) -> str:
    """A cost projection, plus the comparisons that justify (or refute) the design."""
    size_gb = p.est_index_bytes / GB
    tokens_m = p.est_total_tokens / 1e6

    embed_usd = tokens_m * USD_PER_MILLION_EMBED_TOKENS
    import_usd = size_gb * USD_PER_GB_IMPORT
    storage_usd = size_gb * USD_PER_GB_MONTH_STORAGE
    # Write units bill at roughly 1 per KB of record.
    upsert_usd = (p.est_index_bytes / 1024 / 1e6) * USD_PER_MILLION_WRITE_UNITS

    n_ns = SEC_YEAR_NAMESPACES
    ru_per_year_ns = max(size_gb / n_ns, 0.25)
    ru_flat = max(size_gb, 0.25)

    grpc_batch = max_batch_records(int(p.avg_metadata_bytes), grpc=True)
    rest_batch = max_batch_records(int(p.avg_metadata_bytes), grpc=False)

    boiler_saving = tokens_m * p.boilerplate_frac * USD_PER_MILLION_EMBED_TOKENS

    # Merging consecutive chunks that share the same filter flags raises chunks to a
    # useful retrieval size and cuts the record count. Total text is unchanged, so
    # embedding cost is unchanged, but the per-record 6 KB vector is paid once per
    # merged chunk instead of once per fragment.
    merge_n = max(p.mean_flag_run_length, 1.0)
    merged_rows = p.total_rows / merge_n
    merged_meta = p.avg_metadata_bytes * merge_n
    merged_bytes = index_size_bytes(
        int(merged_rows), int(merged_meta), int(p.avg_id_bytes), dims=EMBED_DIMS
    )
    merged_gb = merged_bytes / GB

    lines = [
        "",
        f"Measured from {p.shards_sampled} shards / {p.rows_sampled:,} rows"
        f" (of {p.total_rows:,} total)",
        "",
        f"  avg text bytes/chunk       {p.avg_text_bytes:>10.0f}",
        f"  avg REAL tokens/chunk      {p.avg_real_tokens:>10.1f}   (cl100k_base)",
        f"  source `token_count` says  {p.avg_claimed_tokens:>10.1f}"
        f"   <- overstates by {p.avg_claimed_tokens / max(p.avg_real_tokens, 1e-9):.1f}x;"
        " it is a char count",
        f"  avg metadata bytes         {p.avg_metadata_bytes:>10.0f}",
        f"  boilerplate                {p.boilerplate_frac:>10.1%}",
        f"  tables                     {p.table_frac:>10.1%}",
        f"  mean same-flag chunk run   {p.mean_flag_run_length:>10.1f}   chunks",
        "",
        f"Projected corpus @ {EMBED_DIMS} dims",
        f"  records                    {p.total_rows:>10,}",
        f"  total tokens               {tokens_m:>10,.0f} M",
        f"  index size                 {size_gb:>10,.1f} GB",
        "",
        "Cost projection (USD)",
        f"  embeddings (one-time)      {embed_usd:>10,.2f}",
        f"  bulk import (one-time)     {import_usd:>10,.2f}",
        f"  storage (per month)        {storage_usd:>10,.2f}",
        f"  -- if upserted instead     {upsert_usd:>10,.2f}"
        f"   ({upsert_usd / max(import_usd, 1e-9):.0f}x the import cost)",
        "",
        f"If chunks were merged within same-flag runs (~{merge_n:.1f} chunks each)",
        f"  records                    {merged_rows:>10,.0f}"
        f"   ({p.total_rows / max(merged_rows, 1):.1f}x fewer)",
        f"  avg tokens/chunk           {p.avg_real_tokens * merge_n:>10.0f}",
        f"  index size                 {merged_gb:>10,.1f} GB",
        f"  import (one-time)          {merged_gb * USD_PER_GB_IMPORT:>10,.2f}",
        f"  storage (per month)        "
        f"{merged_gb * USD_PER_GB_MONTH_STORAGE:>10,.2f}",
        "  embeddings                    unchanged   (same total text)",
        "",
        f"Read units per query ({n_ns} year namespaces)",
        f"  single year namespace      {ru_per_year_ns:>10,.1f} RU",
        f"  one flat namespace         {ru_flat:>10,.1f} RU"
        f"   ({ru_flat / max(ru_per_year_ns, 1e-9):.0f}x worse)",
        f"  fan-out over all {n_ns} years  {ru_per_year_ns * n_ns:>10,.1f} RU",
        "",
        "Safe upsert batch size (2 MB cap, measured not guessed)",
        f"  gRPC                       {grpc_batch:>10} records",
        f"  REST                       {rest_batch:>10} records",
        "",
        f"Excluding boilerplate would save ~${boiler_saving:,.2f} of embedding"
        f" and ~{size_gb * p.boilerplate_frac:,.1f} GB of storage.",
        "",
    ]
    if size_gb > 10:
        lines += [
            f"NOTE: {size_gb:,.0f} GB exceeds the Builder plan's 10 GB storage cap"
            " (and Starter's 2 GB). This requires a Standard plan.",
            "",
        ]
    return "\n".join(lines)


def save(p: CorpusProfile, path: Path) -> None:
    atomic_write_bytes(path, json.dumps(asdict(p), indent=1).encode())
