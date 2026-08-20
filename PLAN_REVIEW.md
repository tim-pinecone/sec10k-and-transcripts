# Review of `implementation_plan.md`

Verified against live Pinecone docs (limits, import, pricing) and the actual HF dataset
metadata on 2026-08-20. Numbers below are derived, not recalled.

> **Sections 1–7 were written from the dataset card and Pinecone's published formulas.
> Section 8 supersedes their cost figures with numbers measured from the actual corpus
> — the dataset's `token_count` column turned out to be wrong, which moved the
> embedding estimate by 4×. Read §8 before acting on §2 or §7.**

## Verdict

The core architecture is sound — two indexes, year namespaces, deterministic IDs,
boilerplate kept-and-filterable are all good calls, and the namespace design is
provably right (see §3). But the plan picks the **expensive ingestion path**, gets
**one dataset's shape wrong**, and has **no resumability or observability** for a
13.5M-record job. Those three things are the whole difference between a weekend
build and a $600 lesson.

---

## 1. Dataset reality check

### `astr010/sec-10k-lsh-chunks` — matches the plan exactly ✅

- **13,578,263 rows** (plan said 13.56M — correct), 1,380 auto-converted parquet
  shards, 2.09 GB compressed.
- Every field the plan's schema assumes exists with the right type: `ticker`, `cik`,
  `sic`, `sic_description`, `fiscal_year` (int64), `chunk_id` (int64), `is_table`
  (bool), `is_boilerplate` (bool), `token_count` (int64), `text`.
- Sampled chunks ran 273–511 tokens → the "~300–500 tokens" claim holds. Estimated
  corpus ≈ **5.5B tokens**.
- 1,380 shards is a gift: it's a natural checkpoint unit (one shard = one company).

### `Bose345/sp500_earnings_transcripts` — the plan is wrong here ⚠️

| Plan assumes | Reality |
|---|---|
| "~20,000 transcripts" | **33,362 rows** (+67%) |
| `ticker` | `symbol` |
| `fiscal_year` | `year` |
| `call_date` | `date` |
| speaker recovered by a chunker | **`structured_content`: `list<{speaker, text}>` — already segmented by speaker turn** |
| "Stream the dataset" | **Single 1.82 GB parquet with row groups so large the HF datasets-server refuses to scan it** (tried to read 1.79 GB against a 300 MB limit) |

Three consequences:

1. **Don't run a blind recursive chunker.** `structured_content` already gives clean
   speaker turns. Chunk *within* each turn (~400 tokens, 50 overlap, never crossing a
   speaker boundary). Speaker attribution becomes exact instead of best-effort — a
   real quality upgrade over the plan.
2. **`streaming=True` will not stream cheaply.** The row groups are effectively the
   whole file. Download once (1.82 GB) and read row-group-wise with pyarrow; budget
   ~2 GB RAM.
3. `company_id` is `float64` (cast it; expect NaN). And `(symbol, year, quarter)`
   uniqueness across 33,362 rows is **unverified** — if it collides, the plan's ID
   format silently overwrites records. Must be asserted before ingest.

---

## 2. The big one: bulk import, not upsert (~16× cheaper)

The plan's Phase 2/3 upserts 13.5M vectors in 350–500-record batches. Pricing makes
that the wrong choice by a wide margin.

Per-record size = `ID + metadata + dims × 4 bytes`. With text in metadata
(~1,620 B text + ~200 B other fields) and a ~26 B ID:

| Path | 1536 dims | 768 dims |
|---|---|---|
| Index size | 108.5 GB | 66.8 GB |
| **Ingest via upsert** (WU ≈ 1/KB @ $4/M) | **~$424** | **~$261** |
| **Ingest via bulk import** ($0.25/GB) | **~$27** | **~$17** |
| Storage / month ($0.33/GB) | $35.80 | $22.00 |

Bulk import is **~16× cheaper** and far faster. Constraints that follow from choosing it:

- **Imports only target namespaces that don't yet exist.** No partial top-ups. A
  year must be *fully* staged to parquet before its import starts, and a re-run means
  `delete_namespace(year)` → re-import. This directly rewrites the plan's §6
  idempotency test (the unit of idempotency is a year namespace, not a 500-chunk batch).
- **S3 → AWS-hosted index only.** GCS/Azure can reach any cloud. Pick the index region
  to match the bucket.
- Requires a storage integration configured in the Pinecone console.
- Limits are comfortable: 10,000 namespaces/import, 100,000 files, 10 GB/file, 1 TB
  total. Shard staged parquet to ~250–500 MB for parallelism and resumability.

**Corollary: integrated embedding is off the table at this scale.** Import requires
pre-computed vectors ("records must contain vectors, not text"), so integrated
embedding forces the upsert path *and* hits the 1M passage-tokens/minute cap →
5.5B tokens ≈ **92 hours** of embedding alone. Embed externally.

---

## 3. Year namespaces are the right call — here's the proof

Read units per query scale with **namespace** size. At 768 dims the SEC index is
66.8 GB over 22 years ≈ 3.0 GB per year-namespace.

| Query shape | Year namespaces | One flat namespace |
|---|---|---|
| Single year | **3 RU** | 67 RU (**22× worse**) |
| All 22 years (fan-out) | 66 RU | 67 RU (a wash) |

So partitioning by year is 22× cheaper for year-scoped queries and costs nothing for
full-corpus queries. Keep it. Two caveats:

- At 66 RU per cross-year query and a 2,000 RU/s per-index ceiling, full-corpus
  search tops out around **~30 queries/sec**. Fine for a demo, worth knowing.
- The plan's `query_namespaces` example omits `pool_threads` and
  `connection_pool_maxsize`. Without them the fan-out serializes and the latency win
  evaporates. Set both to ≥ the namespace count.

---

## 4. Batch-size guidance in Phase 2 is only accidentally right

"350–500 vectors per payload to stay under 2 MB" depends entirely on transport and
dimension, which the plan doesn't state:

| Transport | 768 dims | 1536 dims |
|---|---|---|
| gRPC (protobuf, 4 B/float) | ~425 records ✅ | ~215 records |
| REST (JSON, ~13 B/float) | **~170 records** ❌ | ~90 records |

So 350–500 is safe *only* on gRPC at ≤768 dims. The fix isn't a better constant —
it's computing batch size from measured serialized bytes and asserting it. (Hard
caps for reference: 2 MB or 1,000 records per batch; 40 KB filterable metadata per
record; 512-char IDs. Our ~1.8 KB metadata is comfortably clear of 40 KB.)

---

## 5. Hard gate: this needs a Standard plan

Storage caps are **2 GB (Starter)** and **10 GB (Builder)**. The SEC index alone is
67–108 GB. Full-corpus ingest **requires Standard** ($50/mo minimum). Also:
`PINECONE_API_KEY` is not set in this shell.

---

## 6. Missing from the plan entirely

1. **Resumability.** A 13.5M-record embed-and-stage job *will* be interrupted. Needs
   per-shard checkpointing, atomic temp-write-then-rename, and skip-if-done.
2. **Observability.** Flushed progress with a denominator + rate + ETA, plus a
   tailable `status.json` for background runs.
3. **A profiling / dry-run gate.** Before spending ~$120 on embeddings, measure real
   token counts, the **`is_boilerplate` fraction** (unmeasured — if it's 40% of the
   corpus, excluding it saves ~$50 of embedding and ~25 GB of storage), and actual
   metadata bytes. Print a projected cost table and require confirmation.
4. **Metric choice is a one-way door.** Hybrid dense+sparse in a single index requires
   `dotproduct`. With normalized embeddings `dotproduct == cosine`, so choosing
   `dotproduct` now costs nothing and keeps hybrid on the table. Financial retrieval
   wants lexical matching for tickers and figures — don't foreclose it.
5. **No reranking.** A `cohere-rerank-4-fast` pass over top-50 → top-10 is the
   cheapest quality win available.
6. **No retrieval eval.** No golden query set, so there's no way to know if any of
   this works. Even 30 hand-labeled queries would do.
7. **Public-repo hygiene.** License, dataset attribution/licensing, `.env.example`,
   and a `.gitignore` that keeps 20 GB of staged parquet out of git.
8. **Reproducibility for strangers.** Nobody will spend $165 to try your repo. The
   default should be a demo subset (~20 tickers, a few dollars); full corpus behind
   an explicit flag.

Typos: §2 "Deterministic Deterministic Vector IDs"; §1 transcript count.

---

## 7. Projected cost (768 dims, import path, full corpus)

| Item | One-time | Monthly |
|---|---|---|
| OpenAI embeddings (~6B tokens @ $0.02/M) | ~$120 | — |
| Bulk import (74 GB @ $0.25/GB) | ~$19 | — |
| Storage (74 GB @ $0.33/GB) | — | ~$25 |
| Standard plan minimum | — | $50 |
| **Total** | **~$139** | **~$75** |

Same build via upsert at 1536 dims: **~$560 one-time, ~$88/mo**. The upsert-vs-import
decision is worth more than every other optimization in this document combined.

---

## 8. Measured findings (supersedes the estimates above)

`finvec profile --shards 8` sampled 79,250 real rows across 8 evenly-spaced shards.
Three of the numbers in §1–§7 were wrong, and one of them was wrong by 4×.

### 8a. The dataset's `token_count` column is a character count

This is the big one. Verified against `cl100k_base` on three independent shards:

| Shard | Source `token_count` (mean) | Real tokens (mean) | Text bytes (mean) |
|---|---|---|---|
| 7 | 357.2 | **86.7** | 357.2 |
| 345 | 401.9 | **112.7** | 420.0 |
| 690 | 381.1 | **103.9** | 381.6 |

`token_count` tracks *byte length*, not tokens, and its maximum is 511–512 — the corpus
is chunked to a **512-character** cap. So the dataset card's "~300–500 tokens" and the
plan's Phase 1 assumption are both off by ~4×. Real chunks average **~95 tokens /
~380 bytes**: a sentence or two.

Consequences:

- Corpus is **~1.29B tokens**, not 5.5B. Embeddings cost **~$26**, not ~$120.
- Average metadata is **600 bytes**, not the ~1,800 estimated, so the SEC index is
  **91.9 GB**, not 108.5 GB.
- **`token_count` must not be propagated to Pinecone metadata under that name.** A
  mislabeled field is a trap for anyone who later filters on it. Store it as
  `char_count` and compute a real `token_count` if one is wanted.

### 8b. ~95-token chunks are too small for this use case — merge them

`chunk_id` is sequential within `(ticker, fiscal_year)` and consecutive chunks flow
into each other as continuous prose, so they can be merged. Consecutive chunks sharing
the same `(is_boilerplate, is_table)` flags run **3.3 chunks** long on average, which
means merging within same-flag runs preserves filter semantics *exactly* — no blurring
of the boolean flags — while landing at a useful chunk size:

| | As published | Merged within same-flag runs |
|---|---|---|
| Records | 13,578,263 | **4,096,266** (3.3× fewer) |
| Avg tokens/chunk | 95 | **314** |
| Index size | 91.9 GB | **33.4 GB** |
| Import (one-time) | $22.96 | **$8.35** |
| Storage (per month) | $30.31 | **$11.02** |
| Embeddings | $25.73 | **$25.73** (same text) |

Merging is cheaper on every axis *and* better for retrieval, because the per-record
cost is dominated by the fixed 6,144-byte vector — which you pay once per fragment at
95 tokens and once per useful passage at 314. The only thing lost is the ability to
retrieve a single 95-token fragment, which nobody wants.

### 8c. Pinecone bills in decimal GB, not GiB

Its own worked example — 500,000 × (8 B ID + 768 × 4 B + 500 B) = "1.79 GB" — only
resolves at 10⁹. Using GiB anywhere in the cost math understates the bill by 7%. This
is asserted in `tests/test_limits.py` so it cannot drift.

### 8d. Measured facts worth keeping

- **Boilerplate is 27.7%** of chunks; tables are 25.8%. Excluding boilerplate saves
  ~$7 of embedding and ~25 GB of storage — real, but no longer the big lever now that
  the token estimate has come down 4×. Keeping it filterable remains the right call.
- Safe batch sizes at 1536 dims and 600-byte metadata, measured: **278 records over
  gRPC, 91 over REST**. The plan's 350–500 is unsafe on both.
- Read units confirm the namespace design: **4.2 RU** for a single-year query vs
  **91.9 RU** flat — 22× — and a full 22-year fan-out costs the same as one flat query.

### 8e. Revised cost (as published vs merged, 1536 dims, import path)

| | As published | Merged |
|---|---|---|
| Embeddings (one-time) | $25.73 | $25.73 |
| Bulk import (one-time) | $22.96 | $8.35 |
| **One-time total** | **~$49** | **~$34** |
| Storage / month | $30.31 | $11.02 |
| Standard plan minimum | $50 | $50 |
| **Monthly total** | **~$80** | **~$61** |

The full build is **~$49 one-time**, not the ~$139 projected in §7. Still needs a
Standard plan: 92 GB (or 33 GB merged) is far past Builder's 10 GB cap.
