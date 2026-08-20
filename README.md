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

You need:

- A Pinecone **Standard** plan — Starter caps storage at 2 GB and Builder at 10 GB,
  and this is ~120 GB.
- An **AWS S3** bucket plus a Pinecone storage integration. Import from S3 only reaches
  AWS-hosted indexes.
- An OpenAI API key.

`uv run finvec profile` measures the corpus and prints a cost projection **without
spending anything**. Run it first.

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

The pipeline is staged, and every stage is resumable — kill any of them and re-run the
same command to pick up where it left off.

```bash
uv run finvec profile                    # measure corpus + project cost (free)
uv run finvec stage sec                  # embed → staging/{year}/*.parquet
uv run finvec stage transcripts
uv run finvec upload                     # staging → s3://$S3_BUCKET/$S3_PREFIX
uv run finvec import sec                 # start_import per year namespace, then poll
uv run finvec verify                     # per-namespace counts vs. expected
uv run finvec search "supply chain risk" --year 2024 --ticker NVDA
uv run finvec search "AI capex" --years 2021-2024 --index transcripts
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
