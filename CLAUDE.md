# sec10k-and-transcripts

Dual-index Pinecone vector store over 13.6M SEC 10-K chunks and 33k S&P 500 earnings
call transcripts, partitioned into year namespaces and loaded via bulk import from S3.

## Stack

- Python 3.12, [uv](https://docs.astral.sh/uv/)
- Pinecone Python SDK 9.x with gRPC extras (`pinecone[grpc]`)
- OpenAI `text-embedding-3-small` @ **1536 dims**
- `datasets` + `pyarrow` for source reading, `boto3` for S3 staging
- `typer` CLI, `tqdm` progress, `tenacity` retries, `pydantic-settings` config

## Architecture (decided — see PLAN_REVIEW.md for the reasoning)

Two serverless indexes, **AWS-hosted** (bulk import from S3 only reaches AWS indexes):

| Index | Records | Namespaces | Size @1536d |
|---|---|---|---|
| `sec-10k-index` | ~13.58M | `2004`…`2025` (fiscal_year) | ~108 GB |
| `sp500-transcripts-index` | ~1.5M chunks from 33,362 calls | `2005`…`2025` (year) | ~12 GB |

Metric is **`dotproduct`** — OpenAI embeddings are L2-normalized, so dotproduct is
numerically identical to cosine, but it leaves the door open to hybrid dense+sparse in
a single index later. Cosine would foreclose that.

Pipeline is strictly staged, and every stage is resumable:

```
profile → stage (embed → parquet) → upload (S3) → import (per namespace) → verify → search
```

### Non-negotiable constraints

- **Bulk import can only target namespaces that do not yet exist.** A year must be
  *completely* staged before its import starts. Re-ingesting a year means
  `delete_namespace(year)` then re-import. There is no partial top-up.
- Import staging layout must be `s3://{bucket}/{prefix}/{namespace}/*.parquet`.
  Parquet columns: `id` (STRING), `values` (LIST<FLOAT>), `metadata` (STRING, JSON).
  Any other column is silently ignored.
- Import limits: 10,000 namespaces, 100,000 files, 10 GB/file, 1 TB total. Target
  ~250–500 MB per staged file.
- Storage caps are 2 GB (Starter) / 10 GB (Builder) — **this project requires the
  Standard plan.**
- Filterable metadata ≤ 40 KB/record; IDs ≤ 512 chars. Our records are ~2 KB and
  ~26 chars, so both are comfortable.

### Deterministic IDs

- SEC: `{TICKER}_{FISCAL_YEAR}_10K_CHUNK_{CHUNK_ID}`
- Transcripts: `{SYMBOL}_{YEAR}_Q{QUARTER}_TRANSCRIPT_{CHUNK_INDEX}`

Uniqueness is **asserted**, not assumed — the transcripts source is not verified
unique on `(symbol, year, quarter)`, and a collision would silently overwrite.

### Source dataset gotchas

- `astr010/sec-10k-lsh-chunks` — 1,380 parquet shards (one per company). Field names
  match our metadata schema exactly. Shard = checkpoint unit.
- `Bose345/sp500_earnings_transcripts` — field names are `symbol`, `year`, `date`,
  **not** `ticker`/`fiscal_year`/`call_date`. `company_id` is `float64` (cast it,
  expect NaN). Use `structured_content` (`list<{speaker, text}>`) and chunk *within*
  speaker turns — never across them. It is a **single 1.82 GB parquet whose row groups
  exceed HF's 300 MB scan limit**, so `streaming=True` does not stream cheaply:
  download once, read row-group-wise, budget ~2 GB RAM.

## Engineering rules (apply to every long-running loop here)

1. **Resumable.** Checkpoint per shard/unit. Write to a temp path and atomically
   rename. On startup, detect completed work and skip it. Killing and re-running the
   same command must pick up where it left off.
2. **Observable.** Flushed progress with a denominator, rate, and ETA
   (`shard 412/1380 (30%) · 18.2/s · ETA 14m`), plus a tailable `status.json` for
   background runs. Never `print` without `flush=True`.

Never spend embedding money without running `profile` first — it measures real token
counts and the `is_boilerplate` fraction and prints a projected cost table.

## Environment Setup

```bash
cp .env.example .env   # then fill in values
uv sync
```

`PINECONE_API_KEY`, `OPENAI_API_KEY`, `PINECONE_STORAGE_INTEGRATION_ID`, and the AWS
S3 variables are all required for a full run. `profile` needs none of them.

## Project Structure

```
sec10k-and-transcripts/
  src/finvec/
    config.py         # pydantic-settings; .env + config.yaml
    progress.py       # checkpoint manifest + flushed progress + status.json
    ids.py            # deterministic ID construction and validation
    limits.py         # payload/metadata/ID limit validators
    sources/
      sec10k.py       # shard iterator → records
      transcripts.py  # speaker-turn chunker → records
    embed.py          # OpenAI batched embeddings with tenacity retries
    stage.py          # embed → staging/{namespace}/*.parquet (resumable)
    upload.py         # staging → S3
    pinecone_ops.py   # index creation, start_import, poll, stats
    search.py         # single-year query + query_namespaces + rerank
    cli.py            # typer entry point
  tests/              # offline tests: IDs, chunking, limits
  PLAN_REVIEW.md      # review of the original plan, with cost math
  implementation_plan.md
```

## Development Commands

```bash
uv sync
uv run finvec --help
uv run finvec profile                 # measure + cost projection, spends nothing
uv run pytest -q
uv add <package>
```

## Conventions

- All Pinecone reads/writes target an index by **host**, not name (`pc.Index(host=...)`).
- `uv run` for everything. Never call `python3` directly.
- Cost-incurring commands require an explicit `--yes` or an interactive confirmation.
