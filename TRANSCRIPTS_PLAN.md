# Plan — adding S&P 500 earnings transcripts

Every number below is measured from the real dataset, not estimated. That is the first
lesson applied: the SEC build's cost model was wrong by 4× because it trusted a column
name.

## 1. Measured facts

Source: [`Bose345/sp500_earnings_transcripts`](https://huggingface.co/datasets/Bose345/sp500_earnings_transcripts)

| | |
|---|---|
| Calls | 33,362 · 685 symbols · 2005–2025 |
| File | one 1.82 GB parquet, **2 row groups** (32,768 + 594 rows) |
| `content` column | **918 MB — 50% of the file, and we don't need it** |
| `structured_content` | 898 MB · already segmented into speaker turns |
| Turns per call | 69 |
| Year namespaces | **21** (2005–2025), 67–2,210 calls each |
| `(year, quarter)` combinations | 82 |

Chunking the real data through the existing speaker-turn chunker:

| | |
|---|---|
| Chunks per call | 79.2 |
| Tokens per chunk | mean 139 · **median 89** · p75 220 · p90 400 |
| Projected documents | **2,642,270** |
| Projected tokens | 358 M |
| Chunking throughput | ~400 calls/s single core (~1.4 min total) |

## 2. The physical-read problem, and why it isn't one

Hugging Face's own datasets-server **refuses to scan this file** — it attempts 1.79 GB
against a 300 MB limit — which is why the original plan's "stream the dataset" was
wrong. Row group 0 alone is 3.47 GB uncompressed.

Measured, though, `iter_batches(batch_size=64)` with `content` excluded peaks at
**0.53 GB RSS** and reads 3,400 calls/s. pyarrow streams within the row group rather
than materialising it. So:

- **Read only the seven columns we use.** Excluding `content` halves the file.
- **Never call `.to_pylist()` on a row group.** Batches of 64 keep memory flat.
- No reshard-to-smaller-files step is needed. That was the fallback if memory had
  blown up; it doesn't.

## 3. Decisions

| Decision | Choice | Why |
|---|---|---|
| Index | **new `sp500-transcripts-fts`** | Namespaces 2005–2025 already exist in `sec-10k-fts`, and imports only create namespaces that don't exist. A shared index is now foreclosed as a matter of fact. |
| Schema | identical: `text` (FTS, en, stemming) + `embedding` (1536, cosine) | Proven by the SEC build and its live probe. Declare nothing else — metadata-only declarations are rejected at index creation. |
| Checkpoint unit | **a block of 250 consecutive calls** (134 units) | Mirrors the SEC shard pattern that survived three kills and a crash. Resume streams-and-discards to the offset, which costs ~7s at 3,400 calls/s — measured, not assumed. |
| Namespace | `str(year)` | Same year-partitioned design; 21 namespaces. |
| Short-turn floor | **drop chunks under 25 tokens** | See §4 — this is the one decision worth a second look. |
| Layout | `import-{year}/{year}/*.jsonl.gz`, one import per year | Unchanged. Per-year isolation is what makes a failed year retryable. |

## 4. The one real quality decision: short turns

Median chunk is **89 tokens**, because most speaker turns are short. The shortest are
pure conversational filler:

```
[ 1 tok] 'But'
[ 1 tok] 'Okay'
[ 2 tok] 'Absolutely.'
```

Measured effect of a minimum-token floor:

| Floor | Documents kept | Text tokens kept | Mean chunk |
|---|---|---|---|
| none | 100.0% | 100.0% | 139 |
| 15 | 82.2% | 99.0% | 168 |
| **25** | **71.1%** | **97.5%** | **191** |
| 40 | 66.0% | 96.4% | 203 |
| 60 | 60.8% | 94.5% | 216 |

**Recommendation: 25.** It removes 28.9% of documents — every one of them a vector you
pay to store, import, and search — for 2.5% of the corpus's actual words, all of it
filler. Below 25 tokens there is nothing a search can usefully return.

This is deliberately *not* the SEC approach. There we merged adjacent fragments, because
they were consecutive prose from one document. Merging here would splice different
speakers together and destroy the exact speaker attribution that makes this corpus
worth having.

With the floor: **~1,878,700 documents**.

## 5. Build

Five pieces. Everything after staging is already built and dataset-agnostic.

**5.1 `sources/transcripts.py` — the reader** (chunker already exists and is tested)

```python
COLUMNS = [symbol, quarter, year, date, structured_content, company_name, company_id]
def iter_calls(offset, limit) -> Iterator[dict]   # iter_batches(batch_size=64), skip to offset
def call_blocks(size=250) -> list[tuple[int,int]] # (offset, limit) units
```

Field renames are already handled (`symbol`→`ticker`, `year`→`fiscal_year`,
`date`→`call_date`) and NaN `company_id` is already dropped.

**5.2 `stage.py` — a transcripts branch**

Reuses `Embedder`, `limits.validate_metadata`, `UniquenessGuard`, `_write_part`, the
`Checkpoint`, and `Progress` unchanged. Per unit: chunk → apply the token floor →
validate → **embed the whole unit in one call** (the fix that took SEC staging from 5h
to 20 min) → split by namespace → write `shard-{offset:06d}.parquet` per year.

**5.3 Boundary validation, before any embedding**

Not sampling. Asserted over every record, because the two failures that killed SEC runs
were single anomalous units invisible to sampling:

- non-empty text after stripping (the blank-string 400 that killed a 5-hour run)
- `company_id` finite, or absent
- no metadata field starting with `_` or `$`
- no array-of-numbers metadata values
- `(symbol, year, quarter, chunk_index)` unique within a namespace
- token count ≤ 8191 per input
- speaker present, defaulting to `"Unknown"`

**5.4 Schema survey first**

One pass over the parquet schema before anything else, the way all 1,380 SEC shard
schemas were surveyed after `chunk_index` broke a run at 86%. One file here, so it is
cheap and absolute.

**5.5 Parallelise the uploader**

SEC's upload was **99.6% CPU on one core** — 1h35m for 26.74 GB. Profile per part:
parquet read 32%, gzip 34%, doc building 19%, `json.dumps` 14%. Two changes:
`orjson` (4.5× on the serialize step) and a process pool across parts. Keep gzip at
level 6 — level 1 is 5.4× faster but 21% larger, a bad trade for a public dataset that
gets downloaded repeatedly.

## 6. Gates, in order

Each of these exists because something got past its absence.

1. `finvec preflight-import transcripts` — all 17 documented limits, computed from real staged data.
2. `finvec probe-schema transcripts` — round-trip ~60 real documents through `documents.upsert`, which the docs guarantee is the same validation path as import. Immutable schema, so this runs *before* the upload.
3. `finvec verify transcripts` — reconcile against **Pinecone's own import records**, not a local checkpoint.
4. `finvec import-status transcripts` — the command that made the SEC import diagnosable in one shot.

Ordering in `run.sh` is already fixed: **prune runs last**, after verify, and the manifest is written while staging still exists.

## 7. Projected cost and time

| | One-time | Monthly |
|---|---|---|
| Embeddings (349 M tokens) | **$6.99** | — |
| Bulk import (~12.4 GB) | ~$3.10 | — |
| Storage (~12.4 GB) | — | ~$4.10 |

| Stage | Time |
|---|---|
| Chunk + embed (at the measured 10 M tok/min, concurrency 12) | ~35 min |
| Upload (~8.5 GB; ~30 min single-core, ~8 min parallelised) | 8–30 min |
| Import (21 imports, ≥10 min each) | 10–30 min |
| **Total** | **~1–1.5 h** |

Roughly a quarter of the SEC corpus's cost, on a pipeline that is now debugged.

## 8. What could still bite

Honest list, with the mitigation already in place.

| Risk | Mitigation |
|---|---|
| A single anomalous call — empty turn, null `structured_content`, absurd `quarter` | Boundary validation over every record; the failure mode that cost two runs |
| `(symbol, year, quarter)` not unique — **unverified**, and a duplicate silently overwrites | `UniquenessGuard` per namespace, asserted before upsert |
| Import concurrency capacity is undocumented | 21 imports; observed 20 SEC imports finish in 8m12s |
| Rate limits | `--concurrency 12` measured at 10 M tok/min with a trickle of retries; retries are logged |
| Disk | ~8.5 GB staged against 51 GB free; requirement is computed, not a constant |
| Preview API drift | `pinecone==9.1.0` pinned exactly; `pinecone.preview` is outside SemVer |

## 9. Sequence

```bash
uv run finvec stage transcripts --calls 0-999      # smoke: 1,000 calls, cents
uv run finvec probe-schema transcripts             # immutable schema — gate before upload
uv run finvec preflight-import transcripts
./run.sh --detach --yes --dataset transcripts --concurrency 12
uv run finvec import-status transcripts            # authoritative progress
uv run finvec verify transcripts
```

`run.sh` currently hardcodes `sec`; it needs a `--dataset` flag. That is a change to a
script that must not be edited while running — so it lands before the run starts, not
during.

## 10. What this unlocks

Both corpora queryable by year, ticker, and speaker:

```python
# What NVDA filed, versus what management said, about the same thing
sec  = search(sec_index,  "supply chain risk", years=["2024"], ticker="NVDA")
call = search(txn_index,  "supply chain risk", years=["2024"], ticker="NVDA")
```

Cross-corpus comparison is a client-side merge over two indexes rather than one query —
the cost of the unified-index option having been foreclosed. Both sides are dense,
BM25, and hybrid capable, and `search.py` already fans out across year namespaces with
the right merge per scoring type.
