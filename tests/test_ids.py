import pytest

from finvec.ids import (
    UniquenessGuard,
    sec_chunk_id,
    transcript_chunk_id,
    validate_id,
)


def test_sec_id_format_and_determinism():
    a = sec_chunk_id("AAPL", 2024, 42)
    assert a == "AAPL_2024_10K_CHUNK_42"
    assert a == sec_chunk_id("aapl", "2024", "42")  # normalizes, stays stable


def test_transcript_id_format():
    assert transcript_chunk_id("AAPL", 2024, 4, 15) == "AAPL_2024_Q4_TRANSCRIPT_15"


def test_ids_survive_awkward_tickers():
    # Tickers like BRK.B and share-class suffixes must not break the ID grammar.
    assert sec_chunk_id("BRK.B", 2020, 1) == "BRK.B_2020_10K_CHUNK_1"
    assert sec_chunk_id("BF B", 2020, 1) == "BF-B_2020_10K_CHUNK_1"


def test_validate_id_rejects_overlong():
    with pytest.raises(ValueError, match="over the 512"):
        validate_id("X" * 513)
    with pytest.raises(ValueError, match="empty"):
        validate_id("")


def test_uniqueness_guard_catches_silent_overwrite():
    # Two rows for the same (symbol, year, quarter) would otherwise overwrite
    # each other in Pinecone with no error at all.
    guard = UniquenessGuard("2024")
    assert guard.add("AAPL_2024_Q4_TRANSCRIPT_0") is True
    assert guard.add("AAPL_2024_Q4_TRANSCRIPT_0") is False
    with pytest.raises(ValueError, match="duplicate record IDs"):
        guard.raise_if_collisions()
