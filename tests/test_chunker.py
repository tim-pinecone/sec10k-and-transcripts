"""Transcript chunking. Uses a whitespace encoder so tests stay offline."""

import pytest

from finvec.sources.transcripts import chunk_turn, iter_call_records


class WordEncoder:
    """Stand-in for tiktoken: one token per whitespace-delimited word."""

    def encode(self, text):
        return list(range(len(text.split())))

    def decode(self, tokens):
        return " ".join(f"w{t}" for t in tokens)


def test_short_turn_yields_one_chunk():
    assert list(chunk_turn(list(range(10)), size=400, overlap=50)) == [(0, 10)]


def test_windows_overlap_and_cover_everything():
    windows = list(chunk_turn(list(range(1000)), size=400, overlap=50))
    assert windows[0] == (0, 400)
    assert windows[1][0] == 350  # 50-token overlap
    assert windows[-1][1] == 1000  # nothing dropped off the end


def test_no_empty_or_duplicate_trailing_window():
    for n in (399, 400, 401, 700, 701):
        windows = list(chunk_turn(list(range(n)), size=400, overlap=50))
        assert all(e > s for s, e in windows)
        assert windows[-1][1] == n
        assert len(windows) == len(set(windows))


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError, match="must be smaller"):
        list(chunk_turn(list(range(10)), size=50, overlap=50))


def _call(turns):
    return {
        "symbol": "AAPL",
        "quarter": 4,
        "year": 2024,
        "date": "2024-10-31",
        "company_name": "Apple Inc.",
        "company_id": float("nan"),
        "structured_content": turns,
    }


def test_chunks_never_cross_a_speaker_boundary():
    turns = [
        {"speaker": "Tim Cook", "text": " ".join(["alpha"] * 500)},
        {"speaker": "Luca Maestri", "text": " ".join(["beta"] * 500)},
    ]
    records = list(iter_call_records(_call(turns), WordEncoder()))
    speakers = {r.metadata["speaker"] for r in records}
    assert speakers == {"Tim Cook", "Luca Maestri"}
    # Every chunk belongs to exactly one turn, so turn_index is single-valued per chunk.
    assert all(isinstance(r.metadata["turn_index"], int) for r in records)


def test_chunk_index_is_unique_across_the_whole_call():
    turns = [
        {"speaker": "A", "text": " ".join(["x"] * 900)},
        {"speaker": "B", "text": " ".join(["y"] * 900)},
        {"speaker": "C", "text": "short answer"},
    ]
    records = list(iter_call_records(_call(turns), WordEncoder()))
    ids = [r.id for r in records]
    assert len(ids) == len(set(ids))
    assert records[0].id == "AAPL_2024_Q4_TRANSCRIPT_0"
    assert all(r.namespace == "2024" for r in records)


def test_nan_company_id_is_dropped_not_serialized():
    # NaN is not valid JSON and Pinecone would reject the metadata.
    records = list(iter_call_records(_call([{"speaker": "A", "text": "hello"}]),
                                     WordEncoder()))
    assert "company_id" not in records[0].metadata


def test_empty_and_whitespace_turns_are_skipped():
    turns = [
        {"speaker": "A", "text": "   "},
        {"speaker": "B", "text": ""},
        {"speaker": None, "text": "real content here"},
    ]
    records = list(iter_call_records(_call(turns), WordEncoder()))
    assert len(records) == 1
    assert records[0].metadata["speaker"] == "Unknown"
