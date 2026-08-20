# sec10k-and-transcripts

Semantic search over two decades of what public companies **wrote down** and what they
**said out loud** — 13.6M SEC Form 10-K chunks and 33,362 S&P 500 earnings call
transcripts — in two Pinecone serverless indexes partitioned by fiscal year.

Ask "how did NVDA's stated supply-chain risk in its 10-K compare to what management
said on the Q3 call?" and get both sides.

## What's in it

| Index | Source | Records | Namespaces |
|---|---|---|---|
| `sec-10k-index` | [`astr010/sec-10k-lsh-chunks`](https://huggingface.co/datasets/astr010/sec-10k-lsh-chunks) | 13,578,263 chunks · 1,380 companies · 2004–2025 | fiscal year |
| `sp500-transcripts-index` | [`Bose345/sp500_earnings_transcripts`](https://huggingface.co/datasets/Bose345/sp500_earnings_transcripts) | ~1.5M speaker-turn chunks from 33,362 calls · 2005–2025 | calendar year |

Design notes worth knowing before you run anything:

- **Year namespaces, not one flat namespace.** Pinecone read units scale with
  *namespace* size, so a single-year query costs ~3 RU instead of ~108 RU — 22× cheaper
  — while a full-corpus fan-out via `query_namespaces` costs the same either way.
- **Bulk import from S3, not upsert.** Import is $0.25/GB; upserting the same 13.6M
  records burns ~$424 in write units. Same data, ~16× the price.
- **Boilerplate is indexed, not discarded.** The source dataset flags LSH-detected
  boilerplate. It's kept in metadata as `is_boilerplate`, so you can filter it out at
  query time (`filter={"is_boilerplate": False}`) or search it deliberately — instead
  of that decision being baked in at ingest.
- **Speaker attribution is exact.** The transcripts source is already segmented into
  speaker turns, so chunks never cross a speaker boundary.

## Cost and prerequisites

Running the **full** corpus is not free. Measured, not guessed:

| Item | One-time | Monthly |
|---|---|---|
| OpenAI embeddings (~6B tokens @ $0.02/M) | ~$120 | — |
| Pinecone bulk import (~120 GB @ $0.25/GB) | ~$30 | — |
| Pinecone storage (~120 GB @ $0.33/GB) | — | ~$40 |
| Pinecone Standard plan minimum | — | $50 |

Disk: staging the full corpus locally is ~33 GB before upload. `finvec prune --apply`
(or `run.sh --prune`) deletes local parts once each one is individually confirmed
present in S3 at the same size.

You need:

- A Pinecone **Standard** plan — Starter caps storage at 2 GB and Builder at 10 GB,
  and this is ~120 GB.
- An **AWS S3** bucket plus a Pinecone storage integration. Import from S3 only reaches
  AWS-hosted indexes.
- An OpenAI API key.

`uv run finvec profile` measures the corpus and prints a cost projection **without
spending anything**. Run it first.

## Load this corpus into your own Pinecone index

The staged embeddings live in a **public, read-only S3 bucket**, and Pinecone doesn't
require a storage integration to import from a public bucket. So you need no AWS
account, no IAM role, and no console setup — just an index of the right shape and one
call per year:

```python
from pinecone import Pinecone, ServerlessSpec, ImportErrorMode

pc = Pinecone(api_key="...")
pc.create_index(
    name="sec-10k-index",
    dimension=1536,
    metric="dotproduct",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # must be AWS
)
index = pc.Index(host=pc.describe_index("sec-10k-index").host)

for year in range(2004, 2026):
    index.start_import(
        uri=f"s3://{BUCKET}/{PREFIX}/sec/import-{year}/",  # note: import-{year}/
        error_mode=ImportErrorMode.CONTINUE,
        # no integration_id — the bucket is public
    )
```

Each year is a separate import into its own namespace, so a failure costs you one year
rather than the whole corpus. Point `uri` at `.../sec/` instead of `.../sec/import-2004/`
and you'll get 22 namespaces named `import-2004`…`import-2025` — the extra path level
is what makes per-year retries possible.

Requirements on your side: an **AWS-hosted** index (S3 imports can't reach GCP or Azure
indexes), and a **Standard** plan, since imports are Standard/Enterprise only and the
corpus is far past Builder's 10 GB storage cap.

## Setup

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
git clone git@github.com:tim-pinecone/sec10k-and-transcripts.git
cd sec10k-and-transcripts

uv sync

cp .env.example .env
# Fill in PINECONE_API_KEY, OPENAI_API_KEY, PINECONE_STORAGE_INTEGRATION_ID,
# and the AWS S3 variables
```

## Usage

### One command, start to finish

```bash
./run.sh                              # full corpus, foreground, logged
./run.sh --shards 0-9                 # smoke run over 10 companies (cents)
./run.sh --detach --yes --prune       # background; monitor with ./watch.sh
./run.sh --stages stage,upload        # run only some stages
```

`run.sh` logs into AWS via SSO, preflights the keys and free disk, embeds the corpus,
uploads to S3, imports into Pinecone, and verifies the counts. Every stage is resumable,
so the script itself is safe to re-run — it skips work already paid for. Output is teed
to `logs/latest.log`.

Monitor a run from another shell:

```bash
./watch.sh          # live progress, rate, ETA, import status, log tail
tail -f logs/latest.log
cat data/state/status.json
```

**Sequencing matters if you smoke-test first.** A partial import creates real
namespaces, and imports cannot add to an existing namespace — later runs *skip* it, so
the year silently keeps only its smoke-test subset. Either smoke-test without the
import stage:

```bash
./run.sh --shards 0-9 --stages preflight,stage,compact,s3,upload
```

or drop the namespaces afterwards (`finvec drop-namespace sec <year>`) before the full
run. `finvec verify` compares live counts against staged row counts specifically to
catch this.

### Individual stages

Every stage is resumable — kill any of them and re-run the same command to pick up
where it left off.

```bash
uv run finvec profile                    # measure corpus + project cost (free)
uv run finvec stage sec                  # merge + embed → staging/sec/import-{year}/{year}/
uv run finvec compact sec                # coalesce per-shard parts (free, optional)
uv run finvec s3-setup                   # dry run; --apply to create the public bucket
uv run finvec upload sec                 # staging → s3://$S3_BUCKET/$S3_PREFIX
uv run finvec prune sec                  # dry run; --apply to reclaim ~33 GB of disk
uv run finvec import sec                 # one import per year namespace, polled together
uv run finvec verify sec                 # staged vs. live counts, flags incompleteness
uv run finvec drop-namespace sec 2024    # required before re-importing a year
uv run finvec search "supply chain risk" --year 2024 --ticker NVDA
```

Progress is flushed live and mirrored to a `status.json` you can `tail` from another
shell:

```
staging sec · shard 412/1380 (30%) · 18.2 shard/s · ETA 14m · ticker=NVDA year=2019
```

### Query examples

```python
# Single year, boilerplate excluded
sec.query(
    vector=q, namespace="2024", top_k=10, include_metadata=True,
    filter={"ticker": "NVDA", "is_boilerplate": False},
)

# Multi-year trend. pool_threads MUST be set or the fan-out serializes.
index = pc.Index(host=HOST, pool_threads=32, connection_pool_maxsize=32)
index.query_namespaces(
    vector=q, namespaces=["2021", "2022", "2023", "2024"],
    metric="dotproduct", top_k=10, include_metadata=True,
    filter={"ticker": "NVDA"},
)
```

## Development

```bash
uv run pytest -q        # offline tests: IDs, chunking, payload limits
uv add <package>
```

See [`CLAUDE.md`](CLAUDE.md) for architecture and constraints, and
[`PLAN_REVIEW.md`](PLAN_REVIEW.md) for the cost math and the review that produced these
decisions.

## Attribution

Source datasets are third-party works on the Hugging Face Hub — see their dataset cards
for licensing and terms before redistributing derived embeddings. SEC filings
themselves are US government works in the public domain.

## License

Apache-2.0. See [LICENSE](LICENSE).
