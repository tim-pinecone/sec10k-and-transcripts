"""Staging and S3 path layout.

The layout is dictated by one Pinecone constraint: `start_import` treats the
*immediate subdirectories* of its `uri` as namespace names, and it refuses to import
into a namespace that already exists. Those two facts together decide everything here.

The obvious layout — `{dataset}/{year}/*.parquet`, importing once at `{dataset}/` —
creates all 22 year namespaces in a single call. That works exactly once. If it fails
partway, some namespaces exist and some don't, and retrying the same prefix fails on
the ones that got created; recovery means deleting every created namespace and
re-importing 33 GB.

So each year gets its own isolated import root instead:

    {prefix}/{dataset}/import-2004/2004/part-00000.parquet
                                       part-00001.parquet
    {prefix}/{dataset}/import-2005/2005/part-00000.parquet

    start_import(uri="s3://bucket/{prefix}/{dataset}/import-2004/")
        -> creates exactly one namespace, "2004"

One import per year, each independently retryable: a failed year is deleted and redone
on its own, and the other 21 are untouched. The data is stored once — the extra path
level costs nothing.

The footgun this creates is pointing `uri` one level too high, at `{dataset}/`, which
would create 22 namespaces literally named `import-2004` … `import-2025`. Hence
`assert_import_uri`, which every code path that calls `start_import` goes through.
"""

from __future__ import annotations

import re
from pathlib import Path

IMPORT_ROOT_RE = re.compile(r"^import-(?P<ns>\d{4})/?$")

# Parquet part files within a namespace directory.
PART_TEMPLATE = "part-{index:05d}.parquet"


def import_root(dataset: str, namespace: str) -> str:
    """Relative path of a year's isolated import root (the `uri` target)."""
    return f"{dataset}/import-{namespace}"


def namespace_dir(dataset: str, namespace: str) -> str:
    """Relative path of the namespace subdirectory Pinecone reads the name from."""
    return f"{import_root(dataset, namespace)}/{namespace}"


def part_key(dataset: str, namespace: str, index: int) -> str:
    return f"{namespace_dir(dataset, namespace)}/{PART_TEMPLATE.format(index=index)}"


def staging_namespace_dir(staging_dir: Path, dataset: str, namespace: str) -> Path:
    return Path(staging_dir) / namespace_dir(dataset, namespace)


def s3_key(prefix: str, relative: str) -> str:
    """Join an S3 key prefix with a relative path, tolerating stray slashes."""
    return f"{prefix.strip('/')}/{relative.lstrip('/')}" if prefix.strip("/") else relative.lstrip("/")


def import_uri(bucket: str, prefix: str, dataset: str, namespace: str) -> str:
    """The `uri` to hand `start_import` for one year. Always ends in a slash."""
    return f"s3://{bucket}/{s3_key(prefix, import_root(dataset, namespace))}/"


def namespace_from_import_uri(uri: str) -> str:
    """Extract the namespace an import URI will create, or raise.

    This is the guard against pointing `uri` at the dataset directory instead of a
    year's import root — a mistake that silently produces namespaces named
    `import-2004` rather than `2004`, and is only visible after the import succeeds.
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"import URI must start with s3:// — got {uri!r}")
    tail = uri.rstrip("/").rsplit("/", 1)[-1]
    match = IMPORT_ROOT_RE.match(tail + "/")
    if not match:
        raise ValueError(
            f"import URI {uri!r} does not point at a per-year import root.\n"
            f"Expected the last path segment to look like 'import-2024'; got "
            f"{tail!r}.\n"
            f"Pointing at the dataset directory would create namespaces named "
            f"'import-2004' … 'import-2025' instead of '2004' … '2025'."
        )
    return match.group("ns")


def assert_import_uri(uri: str, expected_namespace: str) -> None:
    got = namespace_from_import_uri(uri)
    if got != expected_namespace:
        raise ValueError(
            f"import URI {uri!r} would create namespace {got!r}, "
            f"not {expected_namespace!r}"
        )
