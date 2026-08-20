# 📐 System Architecture & Implementation Plan: Dual-Index SEC & Earnings Transcripts Pinecone Vector Store

## 1. Executive Summary & Objectives

The goal of this project is to construct a production-grade, dual-index vector search architecture on **Pinecone Serverless** leveraging two comprehensive financial datasets:

1. **`astr010/sec-10k-lsh-chunks`**: 13.56 million SEC Form 10-K text chunks across 1,380 companies (2004–2025).
2. **`Bose345/sp500_earnings_transcripts`**: ~20,000 S&P 500 earnings call transcripts (2005–2025).

---

## 2. Key Architectural Decisions

* **Dual-Index Partitioning**: 
  * `sec-10k-index`: Holds 10-K regulatory disclosures.
  * `sp500-transcripts-index`: Holds earnings call presentations and Q&A.
* **Year-Based Namespaces (`namespace="2024"`)**:
  * Enables fast, zero-overhead single-year searches via `index.query(namespace="2024")`.
  * Enables cross-year trend queries via `index.query_namespaces(namespaces=["2022", "2023", "2024"], metric="cosine")`.
* **Retained LSH Boilerplate Metadata**:
  * All 13.56M SEC chunks are indexed into Pinecone.
  * `is_boilerplate` (boolean) is preserved in metadata.
  * Search queries can dynamically include or exclude boilerplate using Pinecone metadata filters (`filter={"is_boilerplate": False}`).
* **Deterministic Deterministic Vector IDs**:
  * Ensures idempotent upserts (re-running the pipeline overwrites existing records instead of duplicating).
  * Format: `{TICKER}_{YEAR}_10K_CHUNK_{CHUNK_ID}` and `{TICKER}_{YEAR}_Q{QUARTER}_TRANSCRIPT_{CHUNK_ID}`.

---

## 3. Detailed Data Schemas

### 3.1 `sec-10k-index` Vector Payload & Metadata

```json
{
  "id": "AAPL_2024_10K_CHUNK_42",
  "values": [0.0123, -0.0456, "... (768 or 1536 dims)"],
  "metadata": {
    "ticker": "AAPL",
    "cik": "0000320193",
    "fiscal_year": 2024,
    "sic": "3571",
    "sic_description": "Electronic Computers",
    "chunk_id": 42,
    "is_table": false,
    "is_boilerplate": false,
    "token_count": 481,
    "text": "The Company's business strategy leverages its unique ability to design and develop..."
  }
}
```

### 3.2 `sp500-transcripts-index` Vector Payload & Metadata

```json
{
  "id": "AAPL_2024_Q4_TRANSCRIPT_15",
  "values": [0.0321, -0.0987, "... (768 or 1536 dims)"],
  "metadata": {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "fiscal_year": 2024,
    "quarter": 4,
    "call_date": "2024-10-31",
    "speaker": "Tim Cook",
    "chunk_index": 15,
    "token_count": 340,
    "text": "We are pleased to report revenue of $94.9 billion for the fiscal fourth quarter..."
  }
}
```

---

## 4. End-to-End Pipeline Implementation Steps

### Phase 1: Data Preparation & Transcripts Chunking
1. **SEC 10-K Data**: Stream directly from `astr010/sec-10k-lsh-chunks` via Hugging Face `datasets` (pre-chunked into ~300-500 tokens).
2. **Earnings Transcripts Data**: Stream `Bose345/sp500_earnings_transcripts`.
   * Apply a recursive character / token chunker (~400 tokens per chunk with 50-token overlap) while preserving speaker attribution (`speaker`) and quarter (`quarter`).

### Phase 2: Embedding Generation & Safe Batching
1. **Embedding Engine**: Utilize OpenAI `text-embedding-3-small` (1,536 dimensions) or local `bge-small-en-v1.5` / `nomic-embed-text` (768 dimensions).
2. **Safe Payload Batching**:
   * Limit batch sizes to **350–500 vectors per API payload** to guarantee keeping payload size under Pinecone's **2 MB hard limit**.
3. **Concurrency & Rate Limits**:
   * Implement retry decorators (`tenacity`) to catch and back off on `HTTP 429 Too Many Requests` (staying under the 100 RPS per namespace rate limit).

### Phase 3: Pinecone Index & Namespace Ingestion
1. Create `sec-10k-index` and `sp500-transcripts-index` in **Pinecone Serverless**.
2. Stream vector batches to their respective year namespace (`namespace=str(fiscal_year)`).

---

## 5. Query & Retrieval Engine

### Single-Year Query Example
```python
# Search 2024 SEC 10-K narrative signal only (excluding boilerplate)
response = sec_index.query(
    vector=query_embedding,
    namespace="2024",
    top_k=10,
    include_metadata=True,
    filter={"ticker": "NVDA", "is_boilerplate": False}
)
```

### Multi-Year Trend Search Example
```python
# Search across 2021-2024 namespaces for AI commentary across transcripts
response = transcripts_index.query_namespaces(
    vector=query_embedding,
    namespaces=["2021", "2022", "2023", "2024"],
    metric="cosine",
    top_k=10,
    include_metadata=True,
    filter={"ticker": "NVDA"}
)
```

---

## 6. Verification Plan

### Automated Tests & Pipeline Checks
- **Schema & Payload Validator**: Test batch payload sizes (<2MB) and metadata size (<40KB) on sample batches before full upload.
- **Idempotency Test**: Re-run a sample 500-chunk batch and confirm Pinecone record count remains identical.
- **Namespace Coverage Verification**: Verify vector counts for all year namespaces (`2004` through `2025`) using `index.describe_index_stats()`.
- **Query Verification**: Execute test queries on single-year (`query()`) and multi-year (`query_namespaces()`) with metadata filters (`is_boilerplate = False`).
