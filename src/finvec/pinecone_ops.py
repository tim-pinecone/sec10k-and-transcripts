"""Index creation and per-year bulk imports.

The import model here is one `start_import` per year namespace, run concurrently and
polled together. That shape is forced by two Pinecone rules:

- `start_import` reads namespace names from the immediate subdirectories of its `uri`.
- A namespace that already exists cannot be imported into.

So a single import over all 22 years is a one-shot operation with no partial retry: if
it fails halfway, the namespaces it already created block any re-run of the same
prefix. Per-year imports make failure local — delete that one year's namespace and
redo it, leaving the rest untouched. See `layout.py` for the directory scheme.

Because the bucket is public, `integration_id` is omitted entirely: Pinecone's docs
state an Integration ID isn't needed to import from a public bucket, which is the
whole reason the bucket is public.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pinecone import ImportErrorMode, Pinecone, ServerlessSpec

from .config import CLOUD, EMBED_DIMS, METRIC, settings
from .layout import assert_import_uri, import_uri
from .progress import Checkpoint

TERMINAL_OK = {"Completed"}
TERMINAL_BAD = {"Failed", "Cancelled"}


def client() -> Pinecone:
    s = settings()
    s.require("pinecone_api_key")
    return Pinecone(api_key=s.pinecone_api_key)


def ensure_index(
    pc: Pinecone,
    name: str,
    region: str,
    dimension: int = EMBED_DIMS,
    metric: str = METRIC,
) -> str:
    """Create the index if absent, and return its host.

    Returns the host rather than the name because every data-plane call should target
    the host — index names are resolved through an extra round trip and are not stable
    identifiers across projects.
    """
    existing = {ix["name"] for ix in pc.list_indexes()}
    if name not in existing:
        pc.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=CLOUD, region=region),
        )
    described = pc.describe_index(name)
    _assert_compatible(described, dimension, metric)
    return described.host


def _assert_compatible(described: Any, dimension: int, metric: str) -> None:
    """Refuse to import into an index whose shape does not match the embeddings.

    Dimension and metric are one-way doors: a mismatch here would either be rejected
    per-record deep inside a multi-hour import, or silently score with the wrong
    similarity function.
    """
    actual_dim = getattr(described, "dimension", None)
    actual_metric = getattr(described, "metric", None)
    if actual_dim not in (None, dimension):
        raise SystemExit(
            f"index {described.name!r} has dimension {actual_dim}, but embeddings are "
            f"{dimension}-dimensional. Delete the index or change EMBED_DIMS."
        )
    if actual_metric is not None and str(actual_metric) != metric:
        raise SystemExit(
            f"index {described.name!r} uses metric {actual_metric!r}, expected "
            f"{metric!r}."
        )


def existing_namespaces(index: Any) -> set[str]:
    stats = index.describe_index_stats()
    return set((stats.get("namespaces") or {}).keys())


@dataclass
class ImportJob:
    namespace: str
    uri: str
    id: str | None = None
    status: str = "NotStarted"
    records_imported: int = 0
    percent_complete: float = 0.0
    error: str | None = None

    @property
    def done(self) -> bool:
        return self.status in TERMINAL_OK | TERMINAL_BAD

    @property
    def ok(self) -> bool:
        return self.status in TERMINAL_OK


@dataclass
class ImportRun:
    jobs: list[ImportJob] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def failed(self) -> list[ImportJob]:
        return [j for j in self.jobs if j.done and not j.ok]

    @property
    def records(self) -> int:
        return sum(j.records_imported for j in self.jobs)


def start_year_imports(
    index: Any,
    dataset: str,
    namespaces: list[str],
    bucket: str,
    prefix: str,
    abort_on_error: bool = False,
) -> ImportRun:
    """Kick off one import per year, skipping years that already exist.

    Years already present in the index are skipped rather than attempted: importing
    into an existing namespace is a guaranteed failure, and reporting it as "skipped"
    makes a resumed run readable instead of a wall of errors.
    """
    present = existing_namespaces(index)
    run = ImportRun()
    error_mode = ImportErrorMode.ABORT if abort_on_error else ImportErrorMode.CONTINUE

    for namespace in namespaces:
        if namespace in present:
            run.skipped.append(namespace)
            continue
        uri = import_uri(bucket, prefix, dataset, namespace)
        # Guards against pointing one level too high, which would create namespaces
        # named "import-2024" instead of "2024".
        assert_import_uri(uri, namespace)
        response = index.start_import(uri=uri, error_mode=error_mode)
        run.jobs.append(
            ImportJob(namespace=namespace, uri=uri, id=str(response.id),
                      status="Pending")
        )
        print(f"started import {response.id} -> namespace {namespace}", flush=True)

    if run.skipped:
        print(
            f"skipped {len(run.skipped)} namespace(s) that already exist: "
            f"{', '.join(run.skipped)}\n"
            f"  to redo one, delete it first: finvec drop-namespace "
            f"{dataset} <year>",
            flush=True,
        )
    return run


def poll_imports(
    index: Any,
    run: ImportRun,
    interval: float = 30.0,
    state_dir: Path | None = None,
    timeout: float | None = None,
) -> ImportRun:
    """Poll every running import until all finish, reporting progress as they go.

    Each import takes at least 10 minutes, so the loop is slow by nature — which makes
    printing something on every pass essential rather than optional.
    """
    checkpoint = (
        Checkpoint("imports", state_dir, flush_every=1) if state_dir else None
    )
    started = time.time()

    while any(not job.done for job in run.jobs):
        if timeout is not None and time.time() - started > timeout:
            pending = [j.namespace for j in run.jobs if not j.done]
            print(
                f"timed out after {timeout:.0f}s with {len(pending)} import(s) still "
                f"running: {', '.join(pending)}. They continue server-side — re-run "
                f"`finvec import` to resume polling.",
                flush=True,
            )
            break

        for job in run.jobs:
            if job.done or not job.id:
                continue
            model = index.describe_import(id=job.id)
            job.status = model.status
            job.percent_complete = model.percent_complete or 0.0
            job.records_imported = model.records_imported or 0
            job.error = model.error
            if job.done and checkpoint:
                checkpoint.mark(
                    job.namespace,
                    import_id=job.id,
                    status=job.status,
                    records=job.records_imported,
                )

        finished = sum(1 for j in run.jobs if j.done)
        summary = " ".join(
            f"{j.namespace}:{j.percent_complete:.0f}%" for j in run.jobs if not j.done
        )
        print(
            f"imports {finished}/{len(run.jobs)} done · "
            f"{run.records:,} records · {int(time.time() - started)}s elapsed"
            + (f" · pending {summary}" if summary else ""),
            flush=True,
        )
        if all(job.done for job in run.jobs):
            break
        time.sleep(interval)

    if checkpoint:
        checkpoint.flush()

    for job in run.failed:
        print(
            f"FAILED {job.namespace}: {job.status} — {job.error}\n"
            f"  the namespace may have been partially created; delete it before "
            f"retrying: finvec drop-namespace <dataset> {job.namespace}",
            flush=True,
        )
    return run


def drop_namespace(index: Any, namespace: str) -> None:
    """Delete a namespace so its year can be re-imported.

    The only route to re-running a year: imports refuse to target an existing
    namespace, so this is a required part of the retry path rather than a cleanup
    convenience.
    """
    index.delete_namespace(name=namespace)


def namespace_counts(index: Any) -> dict[str, int]:
    stats = index.describe_index_stats()
    return {
        name: info.get("vector_count", 0)
        for name, info in (stats.get("namespaces") or {}).items()
    }
