import pytest

from finvec.config import EMBED_DIMS, MAX_UPSERT_BYTES
from finvec.limits import (
    batched_by_bytes,
    index_size_bytes,
    max_batch_records,
    metadata_bytes,
    record_wire_bytes,
    validate_metadata,
)


def test_rest_batch_is_far_smaller_than_grpc():
    """The plan's '350-500 records' is only safe on gRPC; REST is much tighter."""
    grpc = max_batch_records(1800, dims=1536, grpc=True)
    rest = max_batch_records(1800, dims=1536, grpc=False)
    assert rest < grpc
    assert rest < 200, rest
    assert grpc >= 200, grpc


def test_batches_respect_the_2mb_cap():
    meta = {"text": "x" * 1600, "ticker": "AAPL", "fiscal_year": 2024}
    values = [0.01] * EMBED_DIMS
    records = [(f"AAPL_2024_10K_CHUNK_{i}", values, meta) for i in range(5000)]
    batches = list(batched_by_bytes(records, dims=EMBED_DIMS, grpc=True))
    assert sum(len(b) for b in batches) == 5000
    for batch in batches:
        total = sum(record_wire_bytes(r, m, dims=EMBED_DIMS) for r, _, m in batch)
        assert total <= MAX_UPSERT_BYTES
        assert len(batch) <= 1000


def test_metadata_over_40kb_rejected():
    with pytest.raises(ValueError, match="over the 40"):
        validate_metadata({"text": "x" * 41_000}, "REC_1")


def test_metadata_rejects_nested_objects():
    # Pinecone metadata values cannot be objects; catching it here beats a 400
    # halfway through a multi-hour ingest.
    with pytest.raises(ValueError, match="must be string, number"):
        validate_metadata({"speaker": {"name": "Tim Cook"}})


def test_metadata_rejects_non_string_lists():
    with pytest.raises(ValueError, match="list of non-strings"):
        validate_metadata({"years": [2023, 2024]})


def test_index_size_matches_pinecones_documented_example():
    """Pinecone's own worked example: 500k x (8B id + 768x4B + 500B) = 1.79 GB.

    Note the unit: matching this example only works with decimal GB (10^9). Pinecone
    bills in decimal GB, so using GiB anywhere in the cost math understates it by 7%.
    """
    size = index_size_bytes(500_000, 500, avg_id_bytes=8, dims=768)
    assert round(size / 1_000_000_000, 2) == 1.79
    assert round(size / 1024**3, 2) == 1.67  # GiB, for contrast


def test_metadata_bytes_counts_utf8_not_characters():
    assert metadata_bytes({"a": "é"}) > metadata_bytes({"a": "e"})


def test_empty_input_is_rejected_before_a_request_goes_out():
    from finvec.embed import batch_texts

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        list(batch_texts([("real", 10), ("   ", 1)]))


def test_only_transient_errors_are_retried():
    """A deterministic 400 must fail immediately, not after 8 backoffs.

    `APIError` is the base class of `BadRequestError`, so retrying on `APIError` meant
    an empty-input 400 was retried with exponential backoff — ~10 minutes of pointless
    waiting per doomed batch, and 56 retry lines burying the real cause.
    """
    from openai import (
        APIConnectionError,
        APITimeoutError,
        BadRequestError,
        InternalServerError,
        RateLimitError,
    )

    from finvec.embed import Embedder

    predicate = Embedder._embed_once.retry.retry
    retried = predicate.exception_types

    assert RateLimitError in retried
    assert APITimeoutError in retried
    assert APIConnectionError in retried
    assert InternalServerError in retried
    # The ones that must NOT be retried, because retrying cannot change the outcome.
    assert BadRequestError not in retried
    assert not any(issubclass(BadRequestError, t) for t in retried), (
        "BadRequestError is reachable through a retried base class"
    )
