"""Drive per-year FTS bulk imports and check them for completeness.

One import per year namespace, started concurrently and polled together. The layout
reason is unchanged from the vector path — imports only create namespaces that don't
exist, so a single all-years import has no partial retry — and FTS adds a second
reason: `describe_index_stats` is not supported on document-schema indexes, so there is
no way to ask which namespaces exist. Per-year imports make each year's state
observable through its own import record instead.

Existence is therefore discovered rather than queried: start the import and treat a
"namespace already exists" rejection as "this year is already done".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .fts import ImportClient, ImportStatus, NamespaceExists
from .layout import assert_import_uri, import_uri
from .progress import Checkpoint


@dataclass
class YearImport:
    namespace: str
    uri: str
    import_id: str | None = None
    status: str = "NotStarted"
    records_imported: int = 0
    percent_complete: float = 0.0
    error: str | None = None

    @property
    def done(self) -> bool:
        return self.status in {"Completed", "Failed", "Cancelled"}

    @property
    def ok(self) -> bool:
        return self.status == "Completed"

    def absorb(self, status: ImportStatus) -> None:
        self.status = status.status
        self.records_imported = status.records_imported
        self.percent_complete = status.percent_complete
        self.error = status.error


@dataclass
class ImportRun:
    jobs: list[YearImport] = field(default_factory=list)
    already_present: list[str] = field(default_factory=list)

    @property
    def failed(self) -> list[YearImport]:
        return [j for j in self.jobs if j.done and not j.ok]

    @property
    def records(self) -> int:
        return sum(j.records_imported for j in self.jobs)


def start_year_imports(
    api: ImportClient,
    dataset: str,
    namespaces: list[str],
    bucket: str,
    prefix: str,
    on_error: str = "continue",
    integration_id: str | None = None,
) -> ImportRun:
    run = ImportRun()
    for namespace in namespaces:
        uri = import_uri(bucket, prefix, dataset, namespace)
        # Guards against pointing one level too high, which would create namespaces
        # named "import-2024" instead of "2024".
        assert_import_uri(uri, namespace)
        try:
            import_id = api.start(uri, on_error=on_error,
                                  integration_id=integration_id)
        except NamespaceExists:
            run.already_present.append(namespace)
            continue
        run.jobs.append(
            YearImport(namespace=namespace, uri=uri, import_id=import_id,
                       status="Pending")
        )
        print(f"started import {import_id} -> namespace {namespace}", flush=True)

    if run.already_present:
        print(
            f"skipped {len(run.already_present)} namespace(s) that already exist: "
            f"{', '.join(run.already_present)}\n"
            f"  imports cannot add to an existing namespace. To redo one:\n"
            f"    uv run finvec drop-namespace {dataset} <year>",
            flush=True,
        )
    return run


def poll(
    api: ImportClient,
    run: ImportRun,
    interval: float = 30.0,
    state_dir: Path | None = None,
    timeout: float | None = None,
) -> ImportRun:
    """Poll until every import finishes, printing on each pass.

    Each import takes at least ten minutes, so a silent loop here is indistinguishable
    from a hang.
    """
    checkpoint = Checkpoint("imports", state_dir, flush_every=1) if state_dir else None
    started = time.time()

    while any(not job.done for job in run.jobs):
        if timeout is not None and time.time() - started > timeout:
            pending = [j.namespace for j in run.jobs if not j.done]
            print(
                f"stopped polling after {timeout:.0f}s with {len(pending)} import(s) "
                f"still running: {', '.join(pending)}. They continue server-side — "
                f"re-run `finvec import` to resume polling.",
                flush=True,
            )
            break

        for job in run.jobs:
            if job.done or not job.import_id:
                continue
            job.absorb(api.describe(job.import_id))
            if job.done and checkpoint:
                checkpoint.mark(
                    job.namespace,
                    import_id=job.import_id,
                    status=job.status,
                    records=job.records_imported,
                )

        finished = sum(1 for j in run.jobs if j.done)
        pending = " ".join(
            f"{j.namespace}:{j.percent_complete:.0f}%" for j in run.jobs if not j.done
        )
        print(
            f"imports {finished}/{len(run.jobs)} done · {run.records:,} records · "
            f"{int(time.time() - started)}s elapsed"
            + (f" · pending {pending}" if pending else ""),
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
            f"  the namespace may have been partially created; drop it before "
            f"retrying that year",
            flush=True,
        )
    return run


def reconcile(
    expected: dict[str, int], state_dir: Path
) -> tuple[dict[str, tuple[int, int]], list[str]]:
    """Compare staged document counts against what each import reported.

    This replaces the `describe_index_stats` completeness check, which document-schema
    indexes do not support. It is arguably the better check: `records_imported` is the
    server's own count for that specific import, rather than an index-wide aggregate
    that cannot distinguish a short year from a complete one.
    """
    checkpoint = Checkpoint("imports", state_dir)
    rows: dict[str, tuple[int, int]] = {}
    problems: list[str] = []
    for namespace in sorted(set(expected) | set()):
        info = checkpoint.info(namespace)
        imported = info.get("records", 0) if isinstance(info, dict) else 0
        rows[namespace] = (expected[namespace], imported)
        if imported != expected[namespace]:
            problems.append(namespace)
    return rows, problems
