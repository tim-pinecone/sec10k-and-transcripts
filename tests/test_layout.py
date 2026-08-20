"""The per-year import layout, and the guard on pointing `uri` at the wrong level."""

import pytest

from finvec.layout import (
    assert_import_uri,
    import_root,
    import_uri,
    namespace_dir,
    namespace_from_import_uri,
    part_key,
    s3_key,
)


def test_namespace_dir_nests_the_year_inside_its_own_import_root():
    # The inner directory name is what Pinecone reads as the namespace.
    assert namespace_dir("sec", "2024") == "sec/import-2024/2024"
    assert import_root("sec", "2024") == "sec/import-2024"


def test_part_keys_are_zero_padded_for_stable_ordering():
    assert part_key("sec", "2024", 0).endswith("part-00000.parquet")
    assert part_key("sec", "2024", 42).endswith("part-00042.parquet")


def test_import_uri_ends_in_a_slash_and_names_the_import_root():
    uri = import_uri("my-bucket", "datasets/v1", "sec", "2024")
    assert uri == "s3://my-bucket/datasets/v1/sec/import-2024/"
    assert namespace_from_import_uri(uri) == "2024"


def test_uri_pointing_at_the_dataset_dir_is_rejected():
    """The expensive mistake: this would create namespaces named 'import-2024'."""
    with pytest.raises(ValueError, match="per-year import root"):
        namespace_from_import_uri("s3://my-bucket/datasets/v1/sec/")


def test_uri_pointing_at_the_namespace_dir_is_rejected():
    """One level too low — Pinecone would report 'No namespace detected'."""
    with pytest.raises(ValueError, match="per-year import root"):
        namespace_from_import_uri("s3://my-bucket/p/sec/import-2024/2024/")


def test_non_s3_scheme_rejected():
    with pytest.raises(ValueError, match="must start with s3://"):
        namespace_from_import_uri("https://my-bucket.s3.amazonaws.com/import-2024/")


def test_assert_import_uri_catches_a_year_mismatch():
    uri = import_uri("b", "p", "sec", "2024")
    assert_import_uri(uri, "2024")  # no raise
    with pytest.raises(ValueError, match="would create namespace '2024'"):
        assert_import_uri(uri, "2023")


def test_s3_key_tolerates_stray_slashes_and_empty_prefix():
    assert s3_key("/p/v1/", "sec/x.parquet") == "p/v1/sec/x.parquet"
    assert s3_key("", "sec/x.parquet") == "sec/x.parquet"
