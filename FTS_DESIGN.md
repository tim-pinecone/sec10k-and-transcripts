# FTS index design and bulk-import alignment

Verified against the live FTS docs (`2026-01.alpha`) and SDK 9.1.0 on 2026-08-20.
Reference implementation: [`tim-pinecone/sec-dense-fts`](https://github.com/tim-pinecone/sec-dense-fts).

## 1. Bulk import into an FTS index — yes, but not through the SDK

The `pinecone:full-text-search` skill states document-schema indexes have **"no S3 bulk
import"**. That is stale. The live docs have a dedicated
[Bulk import](https://docs.pinecone.io/guides/search/full-text-search#bulk-import)
section for document schemas:

> "the `start_import`, `describe_import`, `list_imports`, and `cancel_import` operations
> are identical, and Pinecone selects the file format automatically from the index type."

What is genuinely different from the Parquet path:

| | Vector index (what we built) | FTS document-schema index |
|---|---|---|
| File format | Parquet | **JSONL** (`.jsonl` / `.jsonl.gz`) |
| File contents | 3 columns: `id`, `values`, `metadata` | One JSON document per line |
| Undeclared fields | **Silently ignored** | **Stored and auto-indexed as metadata** |
| Driven from | `index.start_import(...)` in the SDK | **REST only** — "Bulk import is not yet supported in any Pinecone SDK" |
| API version header | `2026-04` | `X-Pinecone-Api-Version: 2026-01.alpha` |
| Namespace rules | must not already exist | **identical** |
| Directory layout | namespace subdirectories under a prefix | **identical** |
| Public bucket | no integration ID needed | **identical** ("not required for public data sources") |

The two constraints that shaped our whole S3 design — namespace-per-subdirectory and
imports-only-into-nonexistent-namespaces — carry over **unchanged**. So the per-year
isolated import root layout survives as-is:

```
{prefix}/sec/import-2004/2004/part-00000.jsonl.gz
{prefix}/sec/import-2005/2005/part-00000.jsonl.gz

POST https://$INDEX_HOST/bulk/imports
  X-Pinecone-Api-Version: 2026-01.alpha
  {"uri": "s3://.../sec/import-2004/", "errorMode": {"onError": "continue"}}
    -> creates namespace "2004"
```

Import validates each document "through the same code path as a live upsert, so any
document that upserts cleanly imports cleanly" — which means the JSONL writer can be
tested against `documents.upsert` on a handful of records before committing to a
50 GB upload.

## 2. The schema

Schemas declare **ranking fields only**. This is not a style preference — the docs are
explicit that declaring metadata-only fields is *rejected at index creation*:

> "Metadata-only field declarations (`string` without `full_text_search`, `string_list`,
> `float`, `boolean`) are rejected at index creation; metadata is auto-indexed at upsert
> time."

So even though `SchemaBuilder` exposes `add_float_field` / `add_boolean_field` /
`add_string_list_field`, using them for `fiscal_year` or `is_boilerplate` would fail.
Every one of our metadata fields is auto-indexed at upsert with the full operator set
(`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$exists`, `$and`, `$or`,
`$not`) with no declaration at all.

### `sec-10k-fts`

```python
schema = (
    SchemaBuilder()
    .add_string_field("text", full_text_search={"language": "en", "stemming": True})
    .add_dense_vector_field("embedding", dimension=1536, metric="cosine")
    .build()
)
```

That's the whole schema. Two fields.

- **`stemming: True`** — the corpus is long-form regulatory prose, where "recognizes /
  recognized / recognition" should collide. Exact-token needs (tickers, CIKs) are served
  by metadata `$eq`, not BM25, so stemming costs us nothing there.
- **`stop_words`** left at its default (`False`). Removing stop words would degrade
  `$match_phrase` fidelity on phrases like "risk of loss", and precision gain on BM25
  over this corpus is marginal.
- **`metric: "cosine"`** — our embeddings are L2-normalized (measured norm 1.0001), so
  cosine and dotproduct are numerically identical. The dotproduct argument that applied
  to the classic index (keeping sparse-in-same-index open) does not apply here: FTS has
  a dedicated `sparse_vector` field type, independent of the dense metric.

### Auto-indexed metadata (declared nowhere, filterable everywhere)

| Field | Type | Purpose |
|---|---|---|
| `ticker` | string | `$eq` / `$in` — the dominant filter |
| `cik` | string | exact issuer identity |
| `fiscal_year` | number | redundant with the namespace, kept for cross-namespace queries |
| `sic` | string | industry code, `$in` for sector queries |
| `sic_description` | string | human-readable sector |
| `chunk_id` | number | first source fragment; document order |
| `is_boilerplate` | boolean | `{"is_boilerplate": {"$eq": false}}` |
| `is_table` | boolean | separate tabular from narrative |
| `char_count` | number | source char count (the source's mislabeled `token_count`) |
| `token_count` | number | real `cl100k_base` count of the merged text |
| `merged_from` | number | how many source fragments were merged |
| `source_chunk_ids` | array of strings | provenance back to source rows |

Two encoding rules that bite here:

- Numbers are **stored as floating point**, so `fiscal_year` comes back as `2024.0`.
  Filters still take integers (`{"fiscal_year": {"$eq": 2024}}`).
- **"An array of numbers in an undeclared field is rejected rather than stored as
  metadata."** `source_chunk_ids` is already a list of *strings* — that was luck, and it
  must stay that way.

Field names may not start with `_` or `$` and are capped at 64 bytes. Ours comply.

### `sp500-transcripts-fts`

Same two declared fields. Metadata: `ticker`, `company_name`, `fiscal_year`, `quarter`,
`call_date`, `speaker`, `turn_index`, `chunk_index`, `token_count`. `speaker` stays
undeclared metadata rather than a second FTS field — the real query is "everything Tim
Cook said", which is `$eq`, not BM25.

## 3. What this buys over the classic dense index

One index serves three retrieval shapes, all in a single call:

```python
# BM25 keyword
score_by=[{"type": "text", "field": "text", "query": "supply chain disruption"}]

# Dense semantic
score_by=[{"type": "dense_vector", "field": "embedding", "values": q}]

# Hybrid: semantic ranking, hard lexical requirement
score_by=[{"type": "dense_vector", "field": "embedding", "values": q}],
filter={"text": {"$match_all": "Azure"}, "ticker": {"$eq": "MSFT"}}
```

For a financial corpus the hybrid shape is the valuable one: "what does MSFT say about
Azure capex" needs semantic intent *and* a guarantee the term is present. The classic
dense index cannot express that at all. **One scoring type per request** — hard
requirements go in `filter`, never as a second `score_by` clause.

## 4. What breaks in the code we already wrote

| Component | Status |
|---|---|
| `layout.py` (per-year import roots) | **unchanged** — same namespace rules |
| `s3_setup.py` (public bucket) | **unchanged** — public buckets need no integration ID either way |
| `merge.py`, `embed.py` | **unchanged** |
| `stage.py` parquet writer | needs a JSONL emitter (see §5) |
| `pinecone_ops.start_year_imports` | **rewrite** — `index.start_import()` is the classic SDK path; FTS import is REST-only |
| `pinecone_ops.ensure_index` | **rewrite** — `pc.preview.indexes.create(name, schema=...)` |
| `existing_namespaces()` | **broken** — `describe_index_stats` is not supported on document-schema indexes |
| `namespace_counts()` / `verify` | **broken** — same reason |

Replacements for the two broken pieces:

- **Namespace existence**: stop pre-checking. Attempt the import and treat the
  "namespace already exists" error as *skipped*. Self-correcting, and needs no stats
  endpoint.
- **Completeness**: use `describe_import`'s `records_imported` per year, compared against
  the staged line count per namespace. That is a stronger check than the old one — it is
  the server's own count for that specific import, not an index-wide aggregate.

Also worth knowing: **no backup/restore** for document-schema indexes, schemas are
**immutable**, and `pinecone.preview` is explicitly outside SemVer — pin
`pinecone==9.1.0` exactly rather than `>=`.

## 5. JSONL is bigger than Parquet — generate it during upload

Measured on a real staged part (585 documents), not estimated:

| | Per document | 4.1M documents |
|---|---|---|
| Parquet (float32 + zstd) | 3,387 B | **13.9 GB** |
| JSONL, uncompressed | 15,695 B | 64 GB |
| **`jsonl.gz`** (floats at 6 dp) | **5,069 B** | **20.8 GB** |

`jsonl.gz` is **1.50×** Parquet. Rounding floats to 6 decimal places is numerically
free — a 1e-6 absolute step against components of order 1e-2, far below anything that
moves a cosine ranking — and removes about a third of the raw bytes.

Both formats together are ~35 GB, which does fit the 52 GB free. But there is no reason
to hold two copies of the same data: **Parquet stays the durable local artifact and the
`.jsonl.gz` is generated per-part during upload**, into a temp file that is deleted after
the PUT. Peak extra disk is one part (~600 MB), and conversion is pure CPU, so retrying
an upload costs nothing. This is the "final artifact is derived" rule — Parquet is
typed, columnar and cheap to re-read; JSONL is transport.

Resumability changes shape: the old uploader compared local and remote byte sizes, which
cannot work when the local file is Parquet and the remote one is gzipped JSONL. The S3
key's **existence** is the signal instead — multipart uploads are atomic, so an object
appears only once complete. Conversion is also deterministic (`mtime=0`), so a
re-converted part is byte-identical to what was uploaded.

Import is billed at $0.25/GB; the docs don't say whether it meters compressed or
uncompressed bytes, so budget **$5 to $16** for the SEC corpus.

## 6. Validated live, not just on paper

`finvec probe-schema` creates the index, round-trips real staged documents through
`documents.upsert`, and exercises every query and filter shape. The docs guarantee this
is a faithful proxy for import: *"Import validates every document through the same code
path as a live upsert, so any document that upserts cleanly imports cleanly."*

Result on 60 real staged documents:

```
upsert accepted with no per-document errors
searchable after 11s
  OK   BM25 text
  OK   dense vector
  OK   hybrid: dense rank + lexical filter
  OK   metadata filter ticker==ABCP
  OK   boolean filter is_boilerplate==True
  OK   numeric filter fiscal_year $gte 2000
  OK   string-array filter source_chunk_ids $exists
  note  probe docs carry both is_boilerplate values {False, True}
```

Zero matches counts as a **failure**, not a pass — the probe filters are built from
values known to be present in the loaded documents, because a filter that silently
matches nothing looks identical to a broken one if you only check for exceptions.
