# RCA — staging pipeline, 2026-08-20

A 1,380-shard embedding job failed at 31% after five hours. Separately, two runs
embedded the same data concurrently for 33 minutes, and the monitor reported a
dead run as live for 52 minutes.

No data was lost or corrupted. The cost was time, roughly $3–5 of duplicate
embeddings, and confidence.

## Impact

| | |
|---|---|
| Run outcome | failed at shard 424 of 1,380 (31%) |
| Time to failure | ~5 hours of wall clock across restarts |
| Duplicate spend | ~$3–5 (33 minutes of two runs embedding the same shards) |
| Wasted monitoring | 52 minutes reading a frozen progress bar as live |
| Data integrity | intact — 1,686,792 documents, 0 duplicate IDs |
| Work preserved | 422 shards; nothing re-embedded on restart |

## Timeline

| Time | Event |
|---|---|
| 17:55 | Run 1 starts. Per-namespace embedding; no concurrency guard exists yet. |
| 18:23 | Killed to pick up retry logging. |
| 18:24 | Run 2 starts (`--concurrency 24`), still on pre-lock code. |
| 18:35 | Throughput measured at 3.63M tok/min — *slower* than run 1's 4.4M. Cause found: embedding per namespace capped in-flight requests at ~2. |
| 18:50 | Run 3 starts with the fix and the new lock. The lock cannot see run 2, which holds none. **Two runs now embed the same shards.** |
| 19:23 | Run 3 dies: `400 input cannot be an empty string`. |
| 19:56 | Run 2 exits. |
| 20:15 | `watch.sh` still shows 30.7% and an ETA, as though live. Run has been dead 52 minutes. |

---

## Part 1 — The outage

### 1. One empty string killed a 250-record batch and the run

Eleven chunks in the source dataset have empty or whitespace-only text. Merging
them produced a merged record with no text, and embedding providers reject an
empty string by **failing the entire request** — so one bad record in 1.69 million
took down a 250-record batch and the process with it.

The tempting lesson is "smoke test more." The data says otherwise:

| Sample | Blank chunks |
|---|---|
| Shard 424 (ticker FDBC, FY2014–2019) | 11 |
| 34 other shards, 337,000 chunks | **0** |

Every blank is in one company's filings. A 20-shard smoke run would almost
certainly have missed it, and so would a 100-shard one.

**Rule.** For rare, clustered bad data, sampling is not a control — boundary
validation is. Assume every field from a third-party dataset is empty, null, or
malformed *somewhere* in the corpus, and assert it at the point of use. The guard
is one line and would have failed deterministically on record 1 of shard 424.

**Rule.** Batch APIs convert a per-record data problem into a per-batch outage.
When one bad element fails N records, validate all N locally before the call and
name the offending index in the error.

### 2. A deterministic 400 was retried eight times

The retry predicate read `retry_if_exception_type((RateLimitError, APITimeoutError,
APIError))`. `APIError` is the **base class** of `BadRequestError`, so the 400 was
retried with exponential backoff — 56 retry lines, roughly ten minutes of pointless
waiting per doomed batch, and the real cause buried under retry noise.

**Rule.** Enumerate retryable exceptions explicitly and never retry on a base
class. Transient means 429, timeout, connection reset, 5xx. A 4xx that is not 429
will fail identically forever; retrying it only delays the diagnosis.

---

## Part 2 — Systemic failures

These cost more than the bug did.

### 3. Killing the parent did not kill the job

`kill <pid>` on the wrapper script left its child shell and the Python process
alive and still spending money. The checkpoint advanced 26 shards *after* the run
was believed dead.

**Rule.** Kill the process group (`kill -- -<pgid>`), or verify with `ps` that
nothing survived. Never treat a kill as complete because it returned 0.

### 4. A lock added mid-flight protects nothing already running

A concurrency guard was added after run 2 had started. Run 3 acquired the lock
successfully — because run 2 held none — and both embedded the same shards for 33
minutes.

Data survived only by luck of design: both processes wrote identical content to
identical paths via atomic rename, so last-writer-wins was harmless. The money did
not survive.

**Rule.** Ship the mutual-exclusion guard *before* the first long run, not after
the first collision. A lock is only as good as the oldest process that respects it.

**Rule.** Never recommend a restart without first proving the previous process is
gone. "Kill it and re-run" is incomplete advice; "confirm it's dead, then re-run"
is the instruction.

### 5. Verification commands were run against a live pipeline

Test invocations of the same orchestrator repointed the shared `latest.log`
symlink, which would have pointed the operator's monitor at a finished test run
instead of their live job.

**Rule.** A long job owns its shared mutable state — checkpoints, symlinks,
staging directories, status files. Exercise changes in a scratch directory or with
a distinct state path, never against the live one.

### 6. The monitor reported a dead run as live

`watch.sh` rendered `status.json` faithfully and drew a progress bar at 30.7% with
a rate and an ETA. The process had been dead for 52 minutes. A frozen bar and a
slow job are visually identical.

**Rule.** A progress display that cannot distinguish *stalled* from *slow* is
worse than no display, because it manufactures confidence. Every monitor must
report **liveness** (is the process alive) and **freshness** (how old is this
number) before it reports progress.

### 7. A knob that existed but did nothing

`--concurrency` was plumbed end to end and looked correct. But embedding ran
per namespace, and a shard averages ~4,000 records across ~9 fiscal years, so each
call saw ~445 records — two batches — and never had more than **2 requests in
flight** regardless of the setting.

The evidence was visible and ignored for an hour: concurrency 24 ran at 3.63M
tokens/min against concurrency 8's 4.4M. Raising the knob made it *slower*.

**Rule.** Verify a tuning parameter changes the thing it names, by measuring
achieved throughput. A parameter that plumbs correctly and does nothing is worse
than a missing one.

**Rule.** When a change should improve a metric and doesn't, stop and find out
why. That inversion was the diagnosis, sitting in plain sight.

### 8. Silent retries hid the real ceiling

Backoff was silent until fixed mid-incident. The moment retries were logged, they
revealed **1,650 rate-limit retries** at concurrency 24 — the actual TPM ceiling,
invisible until then, and the answer to what the right concurrency is.

**Rule.** Log every retry with attempt number, elapsed time, and cause. A silent
retry storm is indistinguishable from a hang, and retry volume is the cheapest
capacity signal you will ever get.

---

## Part 3 — Trusting third-party data

### 9. A dataset's own column was wrong by 4×

The source's `token_count` column is a **character** count, capped at 512. Verified
against `cl100k_base` on three independent shards: it reported ~380 where the real
count was ~95.

Everything derived from it was wrong — the embedding budget (~$120 projected vs
~$26 actual), the index size, and the belief that chunks were retrieval-sized when
they averaged a sentence and a half.

**Rule.** Independently measure any third-party field you are about to build a
cost model, a chunking strategy, or a capacity plan on. Dataset cards describe
intent; the bytes describe reality.

---

## Part 4 — Pinecone facts worth writing down

Non-obvious behaviours that shaped the design. Each cost time to establish.

| Area | Behaviour |
|---|---|
| **FTS schema** | Declares **ranking fields only** — FTS `string`, `dense_vector`, `sparse_vector`. Declaring `float`/`boolean`/`string_list`/plain `string` is **rejected at index creation**. All other fields are auto-indexed as filterable metadata with no declaration. |
| **Schema mutability** | Immutable in `2026-01.alpha`. A wrong field means a new index and a full re-ingest. Validate the contract on ~50 real records before a large load. |
| **FTS bulk import** | Supported, via **JSONL** (`.jsonl` / `.jsonl.gz`), not Parquet — and **REST-only**, not in any SDK. Needs `X-Pinecone-Api-Version: 2026-01.alpha`. |
| **Undeclared fields** | Parquet import **ignores** them; JSONL import **stores** them. An array of numbers in an undeclared field is **rejected**, not stored. |
| **Namespaces** | Imports only create namespaces that **do not exist**. No partial top-up, ever. This single rule dictates directory layout, retry granularity, and whether two corpora can share an index. |
| **`describe_index_stats`** | **Not supported** on document-schema indexes. Completeness checks must come from each import's own `records_imported`. |
| **`query_namespaces`** | Does not exist on the FTS documents API. It was never a server feature anyway — the classic one is a client-side fan-out plus merge. |
| **Cross-namespace merge** | Cosine scores are comparable across namespaces; **BM25 scores are not** (per-namespace IDF and average document length). A raw score merge over year namespaces is silently biased toward whichever years make the query terms look rarest. Use rank fusion or a cross-encoder rerank. |
| **Billing units** | Decimal **GB (10⁹)**, not GiB. Using GiB understates the bill by 7%. Pinecone's own worked example only resolves at 10⁹. |
| **Public buckets** | An integration ID is **not required** to import from a public bucket — which removes the IAM policy, the cross-account trust role, and the storage integration entirely. |
| **SDK pinning** | `pinecone.preview` is explicitly outside SemVer. Pin exactly (`==9.1.0`); `documents.fetch` and `documents.delete` lost a parameter between 9.0.0 and 9.1.0. |
| **Verify the source** | A packaged skill claimed document-schema indexes had no S3 bulk import. The live docs had a dedicated section for it. Cross-check docs *and* SDK introspection before designing around a limitation. |

---

## Checklist

Before starting any long, paid, multi-unit job:

- [ ] **Mutual exclusion exists and is in place before the first run.** Not added later.
- [ ] **Boundary validation over the whole input domain.** Empty, null, malformed — asserted, not sampled.
- [ ] **Retryable exceptions enumerated explicitly.** No base classes. No retrying 4xx.
- [ ] **Retries logged** with attempt, elapsed, and cause.
- [ ] **Monitor reports liveness and data freshness**, not just the last known numbers.
- [ ] **Every tuning knob measured** to confirm it moves the metric it names.
- [ ] **Third-party fields independently verified** before any cost or capacity model depends on them.
- [ ] **Checkpointing at a granularity where losing one unit is cheap**, with atomic writes.
- [ ] **A kill is verified with `ps`**, not assumed from an exit code.
- [ ] **The immutable decisions identified up front** — schema, dimension, metric, namespace scheme — and validated on a small live sample first.

## What worked

Worth keeping, because these are why the incident was survivable rather than
expensive:

- **Per-shard checkpointing with atomic writes.** Three kills and a crash cost
  zero re-embedding. This is the single highest-value thing in the pipeline.
- **Atomic rename everywhere.** It is the only reason two concurrent writers
  produced 1,686,792 rows with zero duplicates instead of corrupt Parquet.
- **A live contract probe before bulk load.** Round-tripping 60 real documents
  through `documents.upsert` validated the schema, every filter type, and all three
  query shapes for pennies — against an immutable schema, this is the cheapest
  insurance available.
- **Tests written against the specific failure.** The compaction numbering bug
  would have silently dropped records; a test caught it before it ran on real data.
