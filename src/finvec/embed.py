"""OpenAI embeddings, batched by token budget and retried on rate limits.

Batches are bounded by tokens as well as by count, because the request cap that
actually bites at ~400-token chunks is the per-request token limit, not the input
count. A fixed batch size of N inputs would be fine for short chunks and rejected for
long ones.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator, Sequence

from openai import APIError, APITimeoutError, RateLimitError
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .config import EMBED_DIMS, EMBED_MODEL, settings

# Conservative against the per-request limits; the binding constraint in practice is
# tokens-per-minute, which the retry/backoff handles.
MAX_INPUTS_PER_REQUEST = 256
MAX_TOKENS_PER_REQUEST = 80_000
# Hard model limit. An input above this is rejected by the API outright — it is not
# truncated — so it must be caught before a request goes out, not discovered halfway
# through a multi-hour job.
MAX_TOKENS_PER_INPUT = 8191


def _log_retry(state: RetryCallState) -> None:
    """Announce every backoff.

    A silent retry loop is indistinguishable from a hang, and rate-limit backoff is
    exactly when a long embedding run looks frozen. This also tells you whether raising
    concurrency would help or just generate more 429s.
    """
    exc = state.outcome.exception() if state.outcome else None
    print(
        f"  embed retry {state.attempt_number} after "
        f"{state.seconds_since_start:.0f}s: {type(exc).__name__}: "
        f"{str(exc)[:120]}",
        flush=True,
    )


@dataclass
class EmbedBatch:
    texts: list[str]
    tokens: int


def batch_texts(
    items: Sequence[tuple[str, int]],
    max_inputs: int = MAX_INPUTS_PER_REQUEST,
    max_tokens: int = MAX_TOKENS_PER_REQUEST,
) -> Iterator[list[int]]:
    """Yield lists of indices into `items`, bounded by both count and token budget.

    `items` is (text, token_count) pairs. Indices are yielded rather than the texts
    themselves so the caller can keep vectors aligned with their records.
    """
    batch: list[int] = []
    tokens = 0
    for i, (text, count) in enumerate(items):
        if not text or not text.strip():
            # The provider rejects the whole request for one empty string, so a single
            # blank input kills a 250-record batch. Fail here, naming the position, so
            # the cause is obvious instead of arriving as a bare 400 mid-run.
            raise ValueError(
                f"input {i} is empty or whitespace-only; embedding providers reject "
                f"empty strings and fail the entire batch. Filter blanks upstream "
                f"(see merge.merge_records)."
            )
        if count > MAX_TOKENS_PER_INPUT:
            # Chunking caps every unit far below this, so hitting it means the merge
            # or chunk step is broken. Failing here beats a 400 mid-run.
            raise ValueError(
                f"input {i} is {count} tokens, over the model's "
                f"{MAX_TOKENS_PER_INPUT}-token limit; the chunker should have "
                f"prevented this"
            )
        if batch and (len(batch) >= max_inputs or tokens + count > max_tokens):
            yield batch
            batch, tokens = [], 0
        batch.append(i)
        tokens += count
    if batch:
        yield batch


class Embedder:
    """Thread-pooled embedding client.

    Concurrency exists because 4M chunks is ~16,000 requests, and serially that is
    hours of pure round-trip latency. Rate limits are handled by backoff rather than by
    throttling, so throughput self-adjusts to whatever tier the key has.
    """

    def __init__(self, concurrency: int = 8, model: str = EMBED_MODEL,
                 dims: int = EMBED_DIMS) -> None:
        from openai import OpenAI

        s = settings()
        s.require("openai_api_key")
        self.client = OpenAI(api_key=s.openai_api_key)
        self.model = model
        self.dims = dims
        self.concurrency = concurrency
        self.tokens_embedded = 0
        self.retries = 0

    @retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
        wait=wait_random_exponential(multiplier=2, max=120),
        stop=stop_after_attempt(8),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _embed_once(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model, input=texts, dimensions=self.dims
        )
        # The API returns results in request order, but sorting by index makes the
        # alignment guarantee explicit rather than assumed.
        ordered = sorted(response.data, key=lambda d: d.index)
        self.tokens_embedded += response.usage.total_tokens
        return [d.embedding for d in ordered]

    def embed(self, items: Sequence[tuple[str, int]]) -> list[list[float]]:
        """Embed every item, preserving input order.

        Returns one vector per input. A failure after retries propagates rather than
        yielding a short list — a silently missing vector would corrupt the parquet
        alignment between IDs and values.
        """
        batches = list(batch_texts(items))
        vectors: list[list[float] | None] = [None] * len(items)

        def run(indices: list[int]) -> tuple[list[int], list[list[float]]]:
            return indices, self._embed_once([items[i][0] for i in indices])

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            for indices, result in pool.map(run, batches):
                for i, vector in zip(indices, result):
                    vectors[i] = vector

        missing = [i for i, v in enumerate(vectors) if v is None]
        if missing:
            raise RuntimeError(
                f"{len(missing)} of {len(items)} inputs came back without an "
                f"embedding; refusing to write a misaligned parquet part"
            )
        return [v for v in vectors if v is not None]
