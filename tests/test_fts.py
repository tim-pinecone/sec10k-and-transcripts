"""FTS schema and JSONL document shape — the two things import validates strictly."""

import gzip
import json

import pytest

from finvec.config import EMBED_DIMS
from finvec.fts import TEXT_FIELD, VECTOR_FIELD, build_schema
from finvec.jsonl import document_from_row, jsonl_key_for, write_jsonl_gz


def test_schema_declares_only_ranking_fields():
    """Metadata-only declarations are rejected at index creation.

    The docs are explicit: declaring a `float` / `boolean` / plain `string` /
    `string_list` field fails. So the schema must contain exactly the FTS text field
    and the dense vector — never ticker, fiscal_year, is_boilerplate and friends.
    """
    fields = build_schema()["fields"]
    assert set(fields) == {TEXT_FIELD, VECTOR_FIELD}


def test_text_field_is_full_text_with_stemming():
    text = build_schema()["fields"][TEXT_FIELD]
    assert text["type"] == "string"
    assert text["full_text_search"]["language"] == "en"
    assert text["full_text_search"]["stemming"] is True
    # stop_words left unset: removing them degrades $match_phrase fidelity.
    assert "stop_words" not in text["full_text_search"]


def test_vector_field_matches_the_embedder():
    vector = build_schema()["fields"][VECTOR_FIELD]
    assert vector["type"] == "dense_vector"
    assert vector["dimension"] == EMBED_DIMS
    assert vector["metric"] == "cosine"


def _row(metadata=None):
    meta = {
        "ticker": "AAPL", "fiscal_year": 2024, "is_boilerplate": False,
        "source_chunk_ids": ["0", "1"], "text": "revenue grew",
    }
    meta.update(metadata or {})
    return "AAPL_2024_10K_CHUNK_0", [0.0123456789, -0.987654321], meta


def test_text_is_lifted_to_a_top_level_schema_field():
    doc = document_from_row(*_row())
    assert doc["_id"] == "AAPL_2024_10K_CHUNK_0"
    assert doc[TEXT_FIELD] == "revenue grew"
    # Metadata stays top-level too — FTS auto-indexes undeclared fields.
    assert doc["ticker"] == "AAPL"
    assert doc["fiscal_year"] == 2024


def test_floats_are_rounded_to_six_places():
    doc = document_from_row(*_row())
    assert doc[VECTOR_FIELD] == [0.012346, -0.987654]


def test_reserved_field_prefixes_are_rejected():
    # Only `_id` may start with an underscore; `$` is reserved for filter operators.
    with pytest.raises(ValueError, match="reserved character"):
        document_from_row(*_row({"_score": 1}))
    with pytest.raises(ValueError, match="reserved character"):
        document_from_row(*_row({"$gt": 1}))


def test_arrays_of_numbers_are_rejected_not_silently_stored():
    """"An array of numbers in an undeclared field is rejected rather than stored."""
    with pytest.raises(ValueError, match="array of non-strings"):
        document_from_row(*_row({"years": [2023, 2024]}))
    # Arrays of strings are fine, which is why source_chunk_ids are strings.
    assert document_from_row(*_row())["source_chunk_ids"] == ["0", "1"]


def test_key_extension_maps_parquet_to_jsonl_gz():
    assert jsonl_key_for("sec/import-2024/2024/part-00000.parquet") == (
        "sec/import-2024/2024/part-00000.jsonl.gz"
    )
    with pytest.raises(ValueError, match="expected a .parquet"):
        jsonl_key_for("sec/import-2024/2024/part-00000.jsonl.gz")


def test_written_jsonl_is_one_document_per_line_and_reproducible(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    src = tmp_path / "part-00000.parquet"
    pq.write_table(
        pa.table({
            "id": pa.array(["a", "b"], pa.string()),
            "values": pa.array([[0.1, 0.2], [0.3, 0.4]], pa.list_(pa.float32())),
            "metadata": pa.array(
                [json.dumps({"text": "one", "ticker": "X"}),
                 json.dumps({"text": "two", "ticker": "Y"})], pa.string()),
        }),
        src,
    )
    out = tmp_path / "part-00000.jsonl.gz"
    count, size = write_jsonl_gz(src, out)
    assert count == 2

    lines = gzip.open(out, "rt").read().splitlines()
    assert len(lines) == 2
    docs = [json.loads(line) for line in lines]
    assert [d["_id"] for d in docs] == ["a", "b"]
    assert docs[0][TEXT_FIELD] == "one"
    assert "text" not in {k for d in docs for k in d} or docs[0]["text"] == "one"

    # mtime=0 keeps output byte-identical, so a re-converted part matches what was
    # already uploaded.
    assert write_jsonl_gz(src, out) == (count, size)
