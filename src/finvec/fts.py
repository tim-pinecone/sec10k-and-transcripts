"""FTS document-schema index: schema, creation, and REST-driven bulk import.

Two things about the FTS surface shape this module.

First, **schemas declare ranking fields only.** `SchemaBuilder` exposes
`add_float_field` / `add_boolean_field` / `add_string_list_field`, but the docs are
explicit that metadata-only declarations are *rejected at index creation*. Every one of
our filterable fields — ticker, fiscal_year, is_boilerplate and the rest — is
auto-indexed at upsert with the full operator set and must not be declared.

Second, **bulk import is REST-only.** The docs state plainly that "bulk import is not
yet supported in any Pinecone SDK", so `index.start_import(...)` from the classic
data-plane client is not the path here. Requests go to the index host with
`X-Pinecone-Api-Version: 2026-01.alpha`.

Schemas are immutable in `2026-01.alpha` and `pinecone.preview` is outside SemVer, so
the SDK version is pinned exactly rather than floated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator

import requests
from pinecone import Pinecone
from pinecone.preview import SchemaBuilder

from .config import CLOUD, EMBED_DIMS, FTS_API_VERSION, FTS_METRIC, settings

TEXT_FIELD = "text"
VECTOR_FIELD = "embedding"


def build_schema(dimension: int = EMBED_DIMS, metric: str = FTS_METRIC) -> dict:
    """The whole schema: one full-text field and one dense vector field.

    `stemming` is on because the corpus is long-form regulatory prose where
    recognizes/recognized/recognition should collide. Exact-token needs — tickers, CIKs
    — are served by metadata `$eq`, not BM25, so stemming costs nothing there.

    `stop_words` is left at its default (off): removing them would degrade
    `$match_phrase` on phrases like "risk of loss" for a marginal precision gain.
    """
    return (
        SchemaBuilder()
        .add_string_field(
            TEXT_FIELD, full_text_search={"language": "en", "stemming": True}
        )
        .add_dense_vector_field(VECTOR_FIELD, dimension=dimension, metric=metric)
        .build()
    )


def client() -> Pinecone:
    s = settings()
    s.require("pinecone_api_key")
    return Pinecone(api_key=s.pinecone_api_key)


def ensure_index(pc: Pinecone, name: str, region: str) -> str:
    """Create the FTS index if absent and return its host.

    Schemas cannot be migrated, so an existing index is checked for compatibility
    rather than quietly reused: a dimension or metric mismatch here would otherwise
    surface as every document failing deep inside a 4.5M-record import.
    """
    if not pc.preview.indexes.exists(name):
        pc.preview.indexes.create(
            name=name,
            schema=build_schema(),
            deployment={
                "deployment_type": "managed",
                "cloud": CLOUD,
                "region": region,
            },
            read_capacity={"mode": "OnDemand"},
        )

    described = pc.preview.indexes.describe(name)
    while not described.status.ready:
        time.sleep(5)
        described = pc.preview.indexes.describe(name)

    _assert_schema_matches(described)
    return described.host


def _assert_schema_matches(described: Any) -> None:
    fields = getattr(getattr(described, "schema", None), "fields", None) or {}
    vector = fields.get(VECTOR_FIELD)
    if vector is None:
        raise SystemExit(
            f"index {described.name!r} has no {VECTOR_FIELD!r} field. Its schema is "
            f"{sorted(fields)}. Schemas are immutable — delete the index and recreate "
            f"it, or point at a different name."
        )
    dimension = getattr(vector, "dimension", None)
    if dimension not in (None, EMBED_DIMS):
        raise SystemExit(
            f"index {described.name!r} declares {VECTOR_FIELD!r} with dimension "
            f"{dimension}, but embeddings are {EMBED_DIMS}-dimensional. Schemas cannot "
            f"be changed; recreate the index."
        )
    if TEXT_FIELD not in fields:
        raise SystemExit(
            f"index {described.name!r} has no {TEXT_FIELD!r} full-text field; BM25 and "
            f"$match_* filters would be unavailable. Schema is {sorted(fields)}."
        )


# ── Bulk import over REST ─────────────────────────────────────────────────────

# Import statuses, matching the vector-index import lifecycle.
TERMINAL_OK = {"Completed"}
TERMINAL_BAD = {"Failed", "Cancelled"}


@dataclass
class ImportStatus:
    id: str
    status: str
    # Carried because it is the only link back to the namespace an import targeted:
    # document-schema indexes have no describe_index_stats, so the URI is how imported
    # counts get attributed to a year.
    uri: str = ""
    percent_complete: float = 0.0
    records_imported: int = 0
    error: str | None = None

    @property
    def done(self) -> bool:
        return self.status in TERMINAL_OK | TERMINAL_BAD

    @property
    def ok(self) -> bool:
        return self.status in TERMINAL_OK


class NamespaceExists(Exception):
    """Raised when an import targets a namespace that already exists.

    Not an error in a resumed run — it is how a completed year announces itself, given
    `describe_index_stats` is unavailable on document-schema indexes and there is no
    other way to ask which namespaces exist.
    """


class ImportClient:
    """Minimal REST client for the four import operations on an FTS index."""

    def __init__(self, host: str, api_key: str | None = None,
                 api_version: str = FTS_API_VERSION) -> None:
        self.base = f"https://{host.rstrip('/')}" if "://" not in host else host
        key = api_key or settings().pinecone_api_key
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Api-Key": key,
                "Content-Type": "application/json",
                "X-Pinecone-Api-Version": api_version,
            }
        )

    def start(self, uri: str, on_error: str = "continue",
              integration_id: str | None = None) -> str:
        """Start an import and return its id.

        `integration_id` is omitted for public buckets — the docs note an integration
        "is not required for public data sources", which is the entire reason the
        staging bucket is public.
        """
        payload: dict[str, Any] = {"uri": uri, "errorMode": {"onError": on_error}}
        if integration_id:
            payload["integrationId"] = integration_id
        response = self.session.post(f"{self.base}/bulk/imports", json=payload)
        if response.status_code >= 400:
            body = response.text
            if "already exists" in body:
                raise NamespaceExists(body)
            raise RuntimeError(
                f"start_import failed ({response.status_code}) for {uri}: {body}"
            )
        return str(response.json()["id"])

    def describe(self, import_id: str) -> ImportStatus:
        response = self.session.get(f"{self.base}/bulk/imports/{import_id}")
        response.raise_for_status()
        body = response.json()
        return ImportStatus(
            id=str(body.get("id", import_id)),
            status=body.get("status", "Unknown"),
            uri=body.get("uri") or "",
            percent_complete=float(body.get("percentComplete") or 0.0),
            records_imported=int(body.get("recordsImported") or 0),
            error=body.get("error"),
        )

    def list(self) -> Iterator[ImportStatus]:
        token: str | None = None
        while True:
            params = {"paginationToken": token} if token else {}
            response = self.session.get(f"{self.base}/bulk/imports", params=params)
            response.raise_for_status()
            body = response.json()
            for item in body.get("data", []):
                yield ImportStatus(
                    id=str(item.get("id")),
                    status=item.get("status", "Unknown"),
                    uri=item.get("uri") or "",
                    percent_complete=float(item.get("percentComplete") or 0.0),
                    records_imported=int(item.get("recordsImported") or 0),
                    error=item.get("error"),
                )
            token = (body.get("pagination") or {}).get("next")
            if not token:
                return

    def cancel(self, import_id: str) -> None:
        response = self.session.delete(f"{self.base}/bulk/imports/{import_id}")
        response.raise_for_status()
