"""Merging must never blur filter flags or splice unrelated passages together."""

from finvec.merge import merge_records
from finvec.sources.base import Record


def chunk(chunk_id, text, *, year=2024, boiler=False, table=False, ticker="AAPL"):
    return Record(
        id=f"{ticker}_{year}_10K_CHUNK_{chunk_id}",
        namespace=str(year),
        text=text,
        metadata={
            "ticker": ticker, "cik": "0000320193", "fiscal_year": year,
            "sic": "3571", "sic_description": "Electronic Computers",
            "chunk_id": chunk_id, "is_table": table, "is_boilerplate": boiler,
            "token_count": len(text), "text": text,
        },
        token_count=len(text),
    )


def test_flags_stay_exactly_boolean_across_a_merge():
    # A merge that crossed the boilerplate boundary would make
    # filter={"is_boilerplate": False} silently return boilerplate text.
    records = [
        chunk(0, "a" * 400, boiler=False),
        chunk(1, "b" * 400, boiler=True),
        chunk(2, "c" * 400, boiler=True),
    ]
    merged = list(merge_records(records))
    assert len(merged) == 2
    assert merged[0].metadata["is_boilerplate"] is False
    assert merged[1].metadata["is_boilerplate"] is True
    assert merged[1].metadata["merged_from"] == 2


def test_tables_never_merge_with_prose():
    merged = list(merge_records([
        chunk(0, "x" * 300, table=False),
        chunk(1, "| 1 | 2 |", table=True),
    ]))
    assert len(merged) == 2
    assert [m.metadata["is_table"] for m in merged] == [False, True]


def test_years_never_merge_together():
    merged = list(merge_records([
        chunk(0, "old" * 100, year=2023),
        chunk(1, "new" * 100, year=2024),
    ]))
    assert {m.namespace for m in merged} == {"2023", "2024"}


def test_long_run_is_split_at_the_target_not_merged_whole():
    # 17-chunk runs exist in the corpus; merging one whole would yield a
    # ~1,600-token blob instead of a retrieval-sized chunk.
    records = [chunk(i, "z" * 400) for i in range(17)]
    merged = list(merge_records(records, target_chars=1600))
    assert len(merged) > 1
    assert all(len(m.text) <= 1600 + 400 for m in merged)


def test_out_of_order_input_is_sorted_before_merging():
    shuffled = [chunk(2, "third "), chunk(0, "first "), chunk(1, "second ")]
    merged = list(merge_records(shuffled))
    assert merged[0].text.startswith("first")
    assert merged[0].metadata["source_chunk_ids"] == ["0", "1", "2"]


def test_merged_id_uses_the_first_fragment_and_is_deterministic():
    records = [chunk(7, "a" * 200), chunk(8, "b" * 200)]
    once = list(merge_records(records))
    twice = list(merge_records(records))
    assert once[0].id == "AAPL_2024_10K_CHUNK_7"
    assert [m.id for m in once] == [m.id for m in twice]


def test_source_char_count_is_renamed_not_propagated_as_token_count():
    merged = list(merge_records([chunk(0, "a" * 300), chunk(1, "b" * 300)]))
    meta = merged[0].metadata
    assert meta["char_count"] == 600
    # `token_count` must be absent here — staging fills it with a real count.
    assert "token_count" not in meta


def test_metadata_carries_the_company_fields_through():
    merged = list(merge_records([chunk(0, "a" * 100)]))[0].metadata
    assert merged["ticker"] == "AAPL"
    assert merged["cik"] == "0000320193"
    assert merged["sic_description"] == "Electronic Computers"
