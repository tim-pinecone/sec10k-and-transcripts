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

### Chunk merging (decided)

The source's chunks are ~95 real tokens / ~380 bytes each — too thin for financial
retrieval, and wasteful, because the fixed 6,144-byte vector gets paid once per
sentence-sized fragment. Before embedding, **merge consecutive chunks that share the
same `(ticker, fiscal_year, is_boilerplate, is_table)` key**, in `chunk_id` order.

Merging within same-flag runs — which average 3.3 chunks — keeps `is_boilerplate` and
`is_table` exactly boolean, so query filters mean what the schema says they mean. No
`boilerplate_frac` thresholds.

Result: 4.1M records at ~314 tokens, 33.4 GB, ~$8 import, ~$11/mo storage. Embedding
cost is unchanged because it is the same text.

Merged-record metadata:

- `chunk_id` — the **first** source `chunk_id` in the run (keeps IDs deterministic and
  ordering meaningful).
- `source_chunk_ids` — list of strings, the fragments merged in.
- `char_count` — sum of the source `token_count` column. **Never name this field
  `token_count`:** the source column is a character count, and propagating that name
  would mislead anyone who filters on it later.
- `token_count` — real `cl100k_base` count of the merged text, computed by us.
- `is_boilerplate`, `is_table` — unchanged booleans, identical across the run by
  construction.

### S3 staging bucket (public, read-only)

The bucket is **public** for one concrete reason: Pinecone's docs state that *"an
Integration ID isn't needed to import from a public bucket."* Public read therefore
deletes the entire AWS prerequisite chain — IAM policy, a role trusting Pinecone's
account `713131977538`, and a per-project storage integration — for us and for every
other person who wants this corpus. They call `start_import(uri=...)` and nothing else.

The policy grants `s3:GetObject`, `s3:ListBucket`, and `s3:GetBucketLocation`. **No
write actions of any kind.** `BlockPublicAcls` and `IgnorePublicAcls` stay ON, so the
one bucket policy is the only route to public access — an object ACL cannot expose
anything even by mistake. `tests/test_s3_setup.py` asserts the policy grants no
`s3:Put*` or `s3:Delete*`.

Nothing in the bucket is confidential — it is embeddings derived from public SEC
filings and public HF datasets — so the exposure is billing, not secrecy. Egress is
paid by the bucket owner. Keep the bucket in the **same region as the index**: S3 to an
AWS service in the same region transfers free, so the import path costs no egress.
Anonymous internet downloads bill at the normal rate, ~$3 per full copy of the merged
corpus. `PINECONE_REGION` defaults to `AWS_REGION` to keep these aligned.

Created with boto3 via `finvec s3-setup` (dry run by default; `--apply` to execute).

### Per-year import layout

Two Pinecone rules decide the whole directory scheme:

- `start_import` reads namespace names from the **immediate subdirectories** of `uri`.
- A namespace that already exists **cannot** be imported into.

The obvious layout (`{dataset}/{year}/`, one import at `{dataset}/`) creates all 22
namespaces in one call — and works exactly once. A partway failure leaves some
namespaces created, and retrying the same prefix then fails on those, so recovery means
deleting every created namespace and re-importing everything. Instead each year gets an
isolated import root:

```
{prefix}/{dataset}/import-2004/2004/part-00000.parquet
{prefix}/{dataset}/import-2005/2005/part-00000.parquet

start_import(uri="s3://{bucket}/{prefix}/{dataset}/import-2004/")  ->  namespace "2004"
```

One import per year, started concurrently and polled together. A failed year is
dropped and redone alone; the other 21 are untouched. Data is stored once — the extra
path level costs nothing.

The hazard this creates is pointing `uri` one level too high, at `{dataset}/`, which
would create 22 namespaces named `import-2004` … `import-2025`. Every call path goes
through `layout.assert_import_uri`, and `tests/test_layout.py` covers both the
too-high and too-low mistakes.

Retry path, since it is not optional: `finvec drop-namespace {dataset} {year}` then
`finvec import {dataset} --namespaces {year}`.

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

## Running it

`./run.sh` is the orchestrator: AWS SSO login -> preflight -> stage -> compact ->
s3-setup -> upload -> [prune] -> import -> verify. `./watch.sh` is the live monitor,
reading `data/state/status.json` and `logs/latest.log`.

Because every stage is resumable, `run.sh` has no phase markers — re-running it is the
recovery mechanism. Preflight checks keys, the AWS identity, and free disk *before*
anything expensive starts.

### Staging checkpoint unit

One source shard = one company = ~10k chunks. Chosen because it is the unit the source
publishes, it costs seconds to redo, and a shard's output is written atomically. A shard
spans ~10-20 fiscal years, so it writes one `shard-{NNNNN}.parquet` into each of those
years' namespace directories; `compact` coalesces those into `part-{NNNNN}.parquet` of
~400 MB.

`compact` numbers new parts *after* any existing ones. Restarting at `part-00000` would
overwrite the previous part with only the newly staged records — silent data loss when
staging is interrupted, compacted, resumed, and compacted again. Covered by a test.

### The silent-incompleteness trap

An import cannot add to a namespace that already exists; `start_import` skips it. So a
namespace created by an earlier *partial* import keeps its partial contents forever and
no error is raised. This is the one failure mode here that is invisible by default,
which is why `verify` compares live counts against staged parquet row counts rather than
just printing what the index holds. Recovery is `drop-namespace` then re-import that
year.

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
