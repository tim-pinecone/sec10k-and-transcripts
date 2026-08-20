"""Search across year namespaces.

The FTS documents API takes exactly one namespace per request — `documents.search(*,
namespace: str, ...)` — and there is no `query_namespaces` equivalent. So multi-year
search is a client-side fan-out. That is what the classic `Index.query_namespaces` has
always been too: a thread-pool fan-out plus a merge, not a server capability.

The merge is where the care is needed, and it depends on the scoring type:

- **Dense** (`dense_vector`): cosine scores are absolute and comparable across
  namespaces, so sorting the union by score is exactly correct. This is why the classic
  helper requires a `metric` argument.
- **Hybrid** (dense `score_by` + a `$match_*` filter): ranking is still dense, so the
  same applies. The lexical part is a filter, not a score.
- **BM25** (`type: "text"`): scores are **not** comparable. BM25 depends on
  corpus statistics — inverse document frequency and average document length — computed
  within each namespace. The same sentence scores differently in the 2008 namespace than
  in 2021 because the query terms have different document frequency there. Sorting the
  union by raw score systematically over-selects from whichever years make the query
  terms look rarest, which over a 22-year financial corpus is the dominant effect for
  exactly the queries worth running ("COVID", "AI", "supply chain").

Two scale-free ways to merge BM25 results, in order of preference:

1. **Rerank the union** with a cross-encoder. It scores (query, document) pairs
   directly, with no corpus statistics involved, so the ordering is globally consistent.
   This is a fix rather than a workaround, and it improves dense results too.
2. **Reciprocal rank fusion.** Note what RRF degenerates to here: the per-namespace
   result lists are *disjoint*, so every document at rank r gets the same fused score
   regardless of namespace, and the merge becomes round-robin interleaving by rank. That
   removes the IDF bias, but substitutes a uniformity bias — each year gets equal
   representation even when one year genuinely holds all the relevant material. It is
   the right default without a reranker, and worth understanding rather than trusting.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from .config import settings
from .fts import TEXT_FIELD, VECTOR_FIELD

Mode = Literal["text", "dense", "hybrid"]
Merge = Literal["score", "rrf", "rerank"]

RRF_K = 60
DEFAULT_RERANK_MODEL = "cohere-rerank-4-fast"

# Fields worth returning. Explicit rather than ["*"] so payloads stay small — the
# embedding is 1536 floats and is never needed on read.
RESULT_FIELDS = [
    TEXT_FIELD, "ticker", "cik", "fiscal_year", "sic", "sic_description",
    "chunk_id", "is_table", "is_boilerplate", "token_count", "merged_from",
    "company_name", "quarter", "call_date", "speaker",
]


@dataclass
class Hit:
    id: str
    namespace: str
    score: float
    rank: int
    fields: dict[str, Any] = field(default_factory=dict)
    fused: float | None = None

    @property
    def ordering_score(self) -> float:
        return self.fused if self.fused is not None else self.score

    @property
    def text(self) -> str:
        return str(self.fields.get(TEXT_FIELD, ""))


def year_namespaces(spec: str | None, default: Sequence[str] = ()) -> list[str]:
    """`'2024'`, `'2021-2024'`, `'2019,2021-2023'` -> namespace list."""
    if not spec:
        return list(default)
    out: list[str] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lo, hi = piece.split("-", 1)
            out.extend(str(y) for y in range(int(lo), int(hi) + 1))
        else:
            out.append(piece)
    return sorted(set(out))


def build_filter(
    ticker: str | None = None,
    tickers: Sequence[str] | None = None,
    include_boilerplate: bool = False,
    must_contain: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Assemble the metadata and text-match filter.

    Note what is absent: no `fiscal_year` clause. Year is the namespace here, not a
    filter — that is the whole point of the partitioning, and filtering on it as well
    would just cost read units for nothing.
    """
    filt: dict[str, Any] = {}
    if ticker:
        filt["ticker"] = {"$eq": ticker.upper()}
    elif tickers:
        filt["ticker"] = {"$in": [t.upper() for t in tickers]}
    if not include_boilerplate:
        filt["is_boilerplate"] = {"$eq": False}
    if must_contain and must_contain.strip():
        # A hard lexical requirement belongs in `filter`, never as a second score_by
        # clause — the server rejects mixed scoring types, and a BM25 term in score_by
        # would only influence ranking rather than guarantee presence.
        filt[TEXT_FIELD] = {"$match_all": must_contain.strip()}
    if extra:
        filt.update(extra)
    return filt or None


def score_by_for(
    mode: Mode, query: str, embedding: list[float] | None
) -> list[dict[str, Any]]:
    """One scoring type per request — the server rejects mixed types."""
    if mode == "text":
        return [{"type": "text", "field": TEXT_FIELD, "query": query}]
    if embedding is None:
        raise ValueError(f"mode {mode!r} needs a query embedding")
    return [{"type": "dense_vector", "field": VECTOR_FIELD, "values": embedding}]


def search_namespace(
    idx: Any,
    namespace: str,
    score_by: list[dict[str, Any]],
    top_k: int,
    filter: dict[str, Any] | None = None,
) -> list[Hit]:
    kwargs: dict[str, Any] = {
        "namespace": namespace,
        "top_k": top_k,
        "score_by": score_by,
        # Required on every call; some SDK builds 400 without it.
        "include_fields": RESULT_FIELDS,
    }
    if filter:
        kwargs["filter"] = filter
    response = idx.documents.search(**kwargs)
    hits: list[Hit] = []
    for rank, match in enumerate(response.matches, start=1):
        payload = match.to_dict()
        hits.append(
            Hit(
                id=payload.get("_id", getattr(match, "_id", "")),
                namespace=namespace,
                score=float(getattr(match, "_score", getattr(match, "score", 0.0)) or 0),
                rank=rank,
                fields={k: v for k, v in payload.items() if not k.startswith("_")},
            )
        )
    return hits


def fan_out(
    idx: Any,
    namespaces: Sequence[str],
    score_by: list[dict[str, Any]],
    top_k: int,
    filter: dict[str, Any] | None = None,
    max_workers: int = 16,
) -> list[list[Hit]]:
    """One search per namespace, concurrently. Returns one ranked list per namespace.

    Each namespace is asked for the full `top_k` because any single year could supply
    all of the best results; trimming per-namespace would cap that.
    """
    if not namespaces:
        return []
    if len(namespaces) == 1:
        return [search_namespace(idx, namespaces[0], score_by, top_k, filter)]

    def run(namespace: str) -> list[Hit]:
        return search_namespace(idx, namespace, score_by, top_k, filter)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(namespaces))) as pool:
        return list(pool.map(run, namespaces))


def merge_by_score(per_namespace: list[list[Hit]], top_k: int) -> list[Hit]:
    """Correct for dense and hybrid: cosine is comparable across namespaces."""
    union = [hit for hits in per_namespace for hit in hits]
    union.sort(key=lambda h: h.score, reverse=True)
    return union[:top_k]


def merge_rrf(per_namespace: list[list[Hit]], top_k: int, k: int = RRF_K) -> list[Hit]:
    """Rank-based merge, so per-namespace score scales cannot bias the result.

    Because the lists are disjoint this is round-robin interleaving by rank. Ties are
    broken by the original score, which keeps the ordering deterministic and prefers
    the stronger match within a rank tier.
    """
    union: list[Hit] = []
    for hits in per_namespace:
        for hit in hits:
            hit.fused = 1.0 / (k + hit.rank)
            union.append(hit)
    union.sort(key=lambda h: (h.fused or 0.0, h.score), reverse=True)
    return union[:top_k]


def rerank(
    pc: Any,
    query: str,
    per_namespace: list[list[Hit]],
    top_k: int,
    model: str = DEFAULT_RERANK_MODEL,
) -> list[Hit]:
    """Order the union with a cross-encoder.

    This is the principled answer to cross-namespace BM25 comparability: a cross-encoder
    scores each (query, document) pair on its own, so per-namespace corpus statistics
    never enter the ranking.
    """
    union = [hit for hits in per_namespace for hit in hits]
    if not union:
        return []
    result = pc.inference.rerank(
        model=model,
        query=query,
        documents=[{"id": str(i), TEXT_FIELD: h.text} for i, h in enumerate(union)],
        rank_fields=[TEXT_FIELD],
        top_n=min(top_k, len(union)),
        return_documents=False,
    )
    ordered: list[Hit] = []
    for row in result.data:
        hit = union[int(getattr(row, "index", 0))]
        hit.fused = float(getattr(row, "score", 0.0) or 0.0)
        ordered.append(hit)
    return ordered


def default_merge(mode: Mode, namespaces: Sequence[str]) -> Merge:
    """Score merge is exact for dense ranking; BM25 needs a scale-free merge."""
    if len(namespaces) <= 1:
        return "score"
    return "rrf" if mode == "text" else "score"


def embed_query(text: str) -> list[float]:
    from openai import OpenAI

    from .config import EMBED_DIMS, EMBED_MODEL

    s = settings()
    s.require("openai_api_key")
    client = OpenAI(api_key=s.openai_api_key)
    response = client.embeddings.create(
        model=EMBED_MODEL, input=[text], dimensions=EMBED_DIMS
    )
    return response.data[0].embedding


def search(
    pc: Any,
    idx: Any,
    query: str,
    namespaces: Sequence[str],
    mode: Mode = "hybrid",
    top_k: int = 10,
    must_contain: str | None = None,
    ticker: str | None = None,
    tickers: Sequence[str] | None = None,
    include_boilerplate: bool = False,
    merge: Merge | None = None,
    rerank_model: str = DEFAULT_RERANK_MODEL,
) -> tuple[list[Hit], Merge]:
    """Search one or many year namespaces. Returns (hits, merge strategy used)."""
    embedding = None if mode == "text" else embed_query(query)
    # In hybrid mode the query doubles as the must-contain terms unless overridden.
    lexical = must_contain if must_contain is not None else (
        query if mode == "hybrid" else None
    )
    filt = build_filter(
        ticker=ticker,
        tickers=tickers,
        include_boilerplate=include_boilerplate,
        must_contain=lexical if mode == "hybrid" else must_contain,
    )
    per_namespace = fan_out(
        idx, namespaces, score_by_for(mode, query, embedding), top_k, filt
    )
    strategy = merge or default_merge(mode, namespaces)
    if strategy == "rerank":
        return rerank(pc, query, per_namespace, top_k, rerank_model), strategy
    if strategy == "rrf":
        return merge_rrf(per_namespace, top_k), strategy
    return merge_by_score(per_namespace, top_k), strategy
