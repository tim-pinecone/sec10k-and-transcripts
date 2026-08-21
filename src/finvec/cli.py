"""`finvec` command line.

The pipeline is deliberately staged rather than one command: each stage is resumable on
its own, and the expensive stages sit behind the free `profile` gate.

    profile -> stage -> s3-setup -> upload -> import -> verify -> search
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer

from . import profile as profile_mod
from . import compact as compact_mod
from . import fts, import_run, jsonl as jsonl_mod, preflight as preflight_mod, s3_setup
from . import search as search_mod, stage as stage_mod
from . import upload as upload_mod
from .config import SEC_INDEX, TRANSCRIPTS_INDEX, settings

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Dual-index Pinecone vector store over SEC 10-K filings and earnings calls.",
)

DATASETS = {"sec": SEC_INDEX, "transcripts": TRANSCRIPTS_INDEX}


def _state_dir() -> Path:
    d = settings().state_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _require_dataset(dataset: str) -> None:
    if dataset not in DATASETS:
        raise typer.BadParameter(
            f"unknown dataset {dataset!r}; expected one of {', '.join(DATASETS)}"
        )


def _index_host(dataset: str) -> str:
    """Create the FTS index if needed and return its host.

    Data-plane calls target the host, not the name — and FTS bulk import is REST-only,
    so the host is what the import client needs.
    """
    pc = fts.client()
    return fts.ensure_index(pc, DATASETS[dataset], region=settings().index_region)


@app.command()
def profile(
    shards: int = typer.Option(12, help="Number of SEC shards to sample (of 1380)."),
) -> None:
    """Measure the corpus and project cost. Spends nothing, needs no API keys."""
    state = _state_dir()
    p = profile_mod.profile_sec(shards, status_path=state / "status.json")
    typer.echo(profile_mod.render(p))
    out = state / "profile.sec.json"
    profile_mod.save(p, out)
    typer.echo(f"Saved profile to {out}")


@app.command()
def stage(
    dataset: str = typer.Argument("sec", help="'sec' or 'transcripts'."),
    shards: Optional[str] = typer.Option(
        None, help="Shard range for a smoke run, e.g. '0-4' or '0-2,10'. Default: all."
    ),
    concurrency: int = typer.Option(8, help="Concurrent embedding requests."),
) -> None:
    """Merge, embed, and write staging/{dataset}/import-{year}/{year}/*.parquet.

    Resumable per source shard: kill it and re-run the same command, and it re-embeds
    at most one shard's worth of work.
    """
    _require_dataset(dataset)
    if dataset != "sec":
        typer.secho(
            "only `sec` staging is implemented; the transcripts chunker is built but "
            "its shard reader is not wired up yet.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(1)
    s = settings()
    s.require("openai_api_key")
    stage_mod.stage(
        s.staging_dir,
        _state_dir(),
        stage_mod.parse_shard_range(shards),
        concurrency=concurrency,
        dataset=dataset,
        status_path=_state_dir() / "status.json",
    )


@app.command()
def compact(
    dataset: str = typer.Argument("sec", help="'sec' or 'transcripts'."),
    target_mb: int = typer.Option(400, help="Target size per output part, in MB."),
    keep_inputs: bool = typer.Option(
        False, help="Keep the per-shard files instead of deleting them."
    ),
) -> None:
    """Coalesce per-shard parquet parts into fewer, larger ones. Optional, free."""
    _require_dataset(dataset)
    compact_mod.compact(
        settings().staging_dir,
        dataset,
        target_bytes=target_mb * 1024 * 1024,
        keep_inputs=keep_inputs,
        status_path=_state_dir() / "status.json",
    )


@app.command("probe-schema")
def probe_schema(
    dataset: str = typer.Argument("sec", help="'sec' or 'transcripts'."),
    documents: int = typer.Option(50, help="Real staged documents to round-trip."),
    namespace: str = typer.Option("schema-probe", help="Throwaway namespace."),
    keep: bool = typer.Option(False, help="Leave the probe documents in place."),
) -> None:
    """Validate the schema and document shape against the live index, cheaply.

    Worth doing before any large import for two reasons. Schemas are immutable, so a
    wrong field costs a full rebuild. And the docs note import "validates every document
    through the same code path as a live upsert, so any document that upserts cleanly
    imports cleanly" — which makes a 50-document upsert a faithful proxy for a 50 GB
    import.

    Runs BM25, dense, and hybrid queries against the probe documents, then deletes them.
    """
    _require_dataset(dataset)
    s = settings()
    parts = upload_mod.staged_files(s.staging_dir, dataset)
    marker = _state_dir() / f"probe-{dataset}.json"
    if not parts:
        # Staging may legitimately be gone: prune deletes it once the bytes are in S3.
        # Skipping is only honest if the contract was actually validated before, so it
        # is gated on the marker rather than assumed.
        if marker.exists():
            info = json.loads(marker.read_text())
            typer.secho(
                f"skipped: nothing staged locally to probe, and the schema contract "
                f"was already validated at {info.get('validated_at', 'an earlier run')}"
                f" ({info.get('documents', '?')} documents).",
                fg=typer.colors.YELLOW,
            )
            return
        typer.secho(
            f"nothing staged under {s.staging_dir}/{dataset} and no previous "
            f"validation on record — run `finvec stage {dataset}` first (a single "
            f"shard is enough).",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(1)

    docs = []
    for part in parts:
        for doc in jsonl_mod.iter_documents(part):
            docs.append(doc)
            if len(docs) >= documents:
                break
        if len(docs) >= documents:
            break
    typer.echo(f"loaded {len(docs)} real staged documents")

    pc = fts.client()
    _index_host(dataset)  # creates the index and waits for Ready
    idx = pc.preview.index(name=DATASETS[dataset])

    result = idx.documents.batch_upsert(namespace=namespace, documents=docs)
    if getattr(result, "has_errors", False):
        for err in result.errors:
            typer.secho(f"  UPSERT ERROR: {err.error_message}", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.echo("upsert accepted with no per-document errors")

    # Indexing is asynchronous: batch_upsert returning is not the same as searchable.
    sentinel = docs[0][fts.TEXT_FIELD].split()[0]
    deadline = time.time() + 180
    while time.time() < deadline:
        probe = idx.documents.search(
            namespace=namespace, top_k=1,
            score_by=[{"type": "text", "field": fts.TEXT_FIELD, "query": sentinel}],
            include_fields=[],
        )
        if probe.matches:
            typer.echo(f"searchable after {time.time() - deadline + 180:.0f}s")
            break
        time.sleep(5)
    else:
        typer.secho("documents never became searchable within 180s",
                    fg=typer.colors.RED)
        raise typer.Exit(1)

    # Filter probes are built from values known to be present in the loaded docs, and
    # a zero-match result counts as a failure: a filter that silently matches nothing
    # is indistinguishable from a broken one if you only check for exceptions.
    ticker = docs[0].get("ticker")
    boilerplate_values = {bool(d.get("is_boilerplate")) for d in docs}
    known_boilerplate = next(
        (bool(d["is_boilerplate"]) for d in docs if "is_boilerplate" in d), False
    )
    checks = [
        ("BM25 text", dict(
            score_by=[{"type": "text", "field": fts.TEXT_FIELD, "query": sentinel}])),
        ("dense vector", dict(
            score_by=[{"type": "dense_vector", "field": fts.VECTOR_FIELD,
                       "values": docs[0][fts.VECTOR_FIELD]}])),
        ("hybrid: dense rank + lexical filter", dict(
            score_by=[{"type": "dense_vector", "field": fts.VECTOR_FIELD,
                       "values": docs[0][fts.VECTOR_FIELD]}],
            filter={fts.TEXT_FIELD: {"$match_all": sentinel}})),
        (f"metadata filter ticker=={ticker}", dict(
            score_by=[{"type": "dense_vector", "field": fts.VECTOR_FIELD,
                       "values": docs[0][fts.VECTOR_FIELD]}],
            filter={"ticker": {"$eq": ticker}})),
        (f"boolean filter is_boilerplate=={known_boilerplate}", dict(
            score_by=[{"type": "dense_vector", "field": fts.VECTOR_FIELD,
                       "values": docs[0][fts.VECTOR_FIELD]}],
            filter={"is_boilerplate": {"$eq": known_boilerplate}})),
        ("numeric filter fiscal_year $gte 2000", dict(
            score_by=[{"type": "dense_vector", "field": fts.VECTOR_FIELD,
                       "values": docs[0][fts.VECTOR_FIELD]}],
            filter={"fiscal_year": {"$gte": 2000}})),
        ("string-array filter source_chunk_ids $exists", dict(
            score_by=[{"type": "dense_vector", "field": fts.VECTOR_FIELD,
                       "values": docs[0][fts.VECTOR_FIELD]}],
            filter={"source_chunk_ids": {"$exists": True}})),
    ]
    failures = 0
    for label, kwargs in checks:
        try:
            resp = idx.documents.search(
                namespace=namespace, top_k=3, include_fields=["*"], **kwargs
            )
        except Exception as exc:  # noqa: BLE001 - the server message is the point
            typer.secho(f"  FAIL {label}: {type(exc).__name__} {exc}",
                        fg=typer.colors.RED)
            failures += 1
            continue
        if resp.matches:
            typer.secho(f"  OK   {label}: {len(resp.matches)} match(es)",
                        fg=typer.colors.GREEN)
        else:
            typer.secho(
                f"  FAIL {label}: 0 matches, but the probe documents contain values "
                f"that should match — the field may not be indexed as expected",
                fg=typer.colors.RED,
            )
            failures += 1

    if len(boilerplate_values) > 1:
        typer.echo(
            f"  note  probe docs carry both is_boilerplate values {boilerplate_values},"
            f" so the boolean filter is genuinely discriminating"
        )

    if keep:
        typer.echo(f"left {len(docs)} probe documents in namespace {namespace!r}")
    else:
        idx.documents.delete(namespace=namespace, delete_all=True)
        typer.echo(f"deleted probe documents from namespace {namespace!r}")

    if failures:
        typer.secho(f"{failures} check(s) failed", fg=typer.colors.RED)
        raise typer.Exit(1)
    # Recorded so a later run whose staging has been pruned can honestly skip this
    # gate instead of failing on it.
    marker.write_text(
        json.dumps(
            {"dataset": dataset, "documents": len(docs),
             "index": DATASETS[dataset], "validated_at": time.strftime("%Y-%m-%d %H:%M:%S")},
            indent=1,
        )
    )
    typer.secho("schema and document contract validated", fg=typer.colors.GREEN)


@app.command("s3-setup")
def s3_setup_cmd(
    apply: bool = typer.Option(
        False, "--apply", help="Actually create and configure. Default is a dry run."
    ),
    bucket: Optional[str] = typer.Option(None, help="Overrides S3_BUCKET."),
) -> None:
    """Create the public staging bucket and make it anonymously readable.

    Public read is what lets anyone — including us — import without an IAM role or a
    Pinecone storage integration. The policy grants GetObject and ListBucket only:
    no write actions, and ACL-based public access stays blocked.

    Defaults to a dry run, because this creates a globally-named, publicly readable
    AWS resource. Pass --apply to go through with it.
    """
    s = settings()
    name = bucket or s.s3_bucket
    if not name:
        raise typer.BadParameter("set S3_BUCKET in .env or pass --bucket")

    with s3_setup.friendly_aws_errors():
        typer.echo(f"AWS identity: {s3_setup.caller_identity(s.aws_region)}")
        report = s3_setup.create_public_dataset_bucket(
            name, region=s.aws_region, dry_run=not apply
        )
    typer.echo(report.render())
    if not apply:
        typer.secho("Dry run. Re-run with --apply to create it.", fg=typer.colors.CYAN)
        return
    with s3_setup.friendly_aws_errors():
        verified = s3_setup.verify_public_read(name, region=s.aws_region)
    typer.echo("Verified live configuration:")
    for key, value in verified.items():
        typer.echo(f"  {key}: {value}")
    if verified.get("grants_write"):
        typer.secho(
            "REFUSE: the live bucket policy grants write access to the public. "
            "Fix the policy before uploading anything.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)


@app.command("s3-verify")
def s3_verify_cmd(
    bucket: Optional[str] = typer.Option(None, help="Overrides S3_BUCKET."),
) -> None:
    """Read back the bucket's live public-access configuration."""
    s = settings()
    name = bucket or s.s3_bucket
    if not name:
        raise typer.BadParameter("set S3_BUCKET in .env or pass --bucket")
    with s3_setup.friendly_aws_errors():
        verified = s3_setup.verify_public_read(name, region=s.aws_region)
    for key, value in verified.items():
        typer.echo(f"  {key}: {value}")


@app.command("upload")
def upload_cmd(
    dataset: str = typer.Argument(..., help="'sec' or 'transcripts'."),
) -> None:
    """Convert staged parquet to gzipped JSONL and upload it to S3.

    FTS import reads JSONL, so conversion happens on the way out rather than being kept
    on disk in both formats. Resumable: an existing S3 key means done, and S3 multipart
    uploads are atomic so presence implies completeness.
    """
    _require_dataset(dataset)
    s = settings()
    s.require("s3_bucket")
    # Written first, while staging still exists: prune deletes the files that import
    # and verify would otherwise have to glob.
    manifest = upload_mod.write_manifest(s.staging_dir, dataset, _state_dir())
    typer.echo(
        f"manifest: {len(manifest['namespaces'])} namespaces, "
        f"{manifest['total_documents']:,} documents"
    )
    with s3_setup.friendly_aws_errors():
        upload_mod.upload(
            s.staging_dir,
            dataset,
            s.s3_bucket,
            s.s3_prefix,
            region=s.aws_region,
            status_path=_state_dir() / "status.json",
        )


@app.command()
def prune(
    dataset: str = typer.Argument("sec", help="'sec' or 'transcripts'."),
    apply: bool = typer.Option(
        False, "--apply", help="Actually delete. Default is a dry run."
    ),
) -> None:
    """Delete local staged parts already confirmed in S3, to reclaim disk.

    Each file is checked individually against S3 by key and size before removal;
    anything not confirmed is kept. Staging the full corpus is ~33 GB.
    """
    _require_dataset(dataset)
    s = settings()
    s.require("s3_bucket")
    with s3_setup.friendly_aws_errors():
        upload_mod.prune_uploaded(
            s.staging_dir, dataset, s.s3_bucket, s.s3_prefix,
            region=s.aws_region, dry_run=not apply,
        )


@app.command("preflight-import")
def preflight_import(
    dataset: str = typer.Argument("sec", help="'sec' or 'transcripts'."),
    sample: int = typer.Option(
        400, help="Documents sampled per parquet part for field-size checks."
    ),
) -> None:
    """Audit staged data against every documented import and document limit.

    Reads no Pinecone state and costs nothing. Worth running before an import
    because the recovery path on the far side is expensive: imports only create
    namespaces that do not exist, so a rejected import leaves a partially created
    namespace that blocks its own retry until it is dropped.
    """
    _require_dataset(dataset)
    s = settings()
    import json as _json

    checkpoint = _state_dir() / f"stage-{dataset}.checkpoint.json"
    shards_done = len(_json.loads(checkpoint.read_text())) if checkpoint.exists() else 0

    measured = preflight_mod.measure(s.staging_dir, dataset, sample_docs_per_part=sample)
    if not measured["files"]:
        typer.secho(f"nothing staged under {s.staging_dir}/{dataset}",
                    fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    report = preflight_mod.audit(measured, shards_done=shards_done)
    typer.echo(preflight_mod.render(report, measured, shards_done))
    if report.failures:
        raise typer.Exit(1)


@app.command("import")
def import_cmd(
    dataset: str = typer.Argument("sec", help="'sec' or 'transcripts'."),
    namespaces: Optional[str] = typer.Option(
        None, help="Comma-separated years. Default: every staged year."
    ),
    interval: float = typer.Option(30.0, help="Seconds between status polls."),
    abort_on_error: bool = typer.Option(
        False, help="Fail the import on the first bad document instead of skipping it."
    ),
) -> None:
    """Start one bulk import per year namespace and poll them all to completion.

    One import per year, not one for the whole corpus: a namespace cannot be imported
    into twice, so a single multi-year import that fails partway leaves created
    namespaces blocking any retry.

    Years that already exist are discovered, not queried — document-schema indexes do
    not support `describe_index_stats`, so the import is attempted and a
    "namespace already exists" rejection is reported as skipped.
    """
    _require_dataset(dataset)
    s = settings()
    s.require("s3_bucket", "pinecone_api_key")

    if namespaces:
        years, source = [n.strip() for n in namespaces.split(",") if n.strip()], "flag"
    else:
        with s3_setup.friendly_aws_errors():
            years, source = upload_mod.resolve_namespaces(
                s.staging_dir, dataset, _state_dir(),
                bucket=s.s3_bucket, prefix=s.s3_prefix, region=s.aws_region,
            )
    if not years:
        typer.secho(
            f"no namespaces found in the manifest, in "
            f"s3://{s.s3_bucket}/{s.s3_prefix}/{dataset}, or under "
            f"{s.staging_dir}/{dataset} — run `finvec stage {dataset}` and "
            f"`finvec upload {dataset}` first.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(1)
    typer.echo(f"{len(years)} namespace(s) to import, resolved from {source}")

    api = fts.ImportClient(_index_host(dataset))
    run = import_run.start_year_imports(
        api,
        dataset,
        years,
        s.s3_bucket,
        s.s3_prefix,
        on_error="abort" if abort_on_error else "continue",
        # Omitted for a public bucket: the docs note an integration "is not required
        # for public data sources".
        integration_id=s.pinecone_storage_integration_id or None,
    )
    if not run.jobs:
        typer.echo("nothing to import")
        return
    run = import_run.poll(api, run, interval=interval, state_dir=_state_dir())
    typer.echo(
        f"\nimported {run.records:,} documents across "
        f"{len(run.jobs) - len(run.failed)}/{len(run.jobs)} namespaces"
    )
    typer.echo("documents are indexed asynchronously and may not be searchable yet")
    if run.failed:
        raise typer.Exit(1)


@app.command("import-status")
def import_status(
    dataset: str = typer.Argument("sec", help="'sec' or 'transcripts'."),
) -> None:
    """List every import on the index, straight from Pinecone.

    Reads the server's own import records rather than a local status file, so it works
    while a run is polling, after one has exited, and regardless of what any local
    state says.
    """
    _require_dataset(dataset)
    api = fts.ImportClient(_index_host(dataset))
    rows = list(api.list())
    if not rows:
        typer.echo("no imports on this index")
        return

    by_status: dict[str, int] = {}
    total = 0
    typer.echo(f"  {'id':>4}  {'status':<12}{'pct':>7}{'records':>14}  uri")
    for row in sorted(rows, key=lambda r: int(r.id) if r.id.isdigit() else 0):
        by_status[row.status] = by_status.get(row.status, 0) + 1
        total += row.records_imported
        colour = (
            typer.colors.GREEN if row.status == "Completed"
            else typer.colors.RED if row.status in ("Failed", "Cancelled")
            else typer.colors.CYAN
        )
        tail = row.error or ""
        typer.secho(
            f"  {row.id:>4}  {row.status:<12}{row.percent_complete:>6.0f}%"
            f"{row.records_imported:>14,}  {tail[:60]}",
            fg=colour,
        )
    typer.echo("")
    typer.echo("  " + " · ".join(f"{k}: {v}" for k, v in sorted(by_status.items())))
    typer.echo(f"  {total:,} documents imported so far")


@app.command("drop-namespace")
def drop_namespace_cmd(
    dataset: str = typer.Argument(..., help="'sec' or 'transcripts'."),
    namespace: str = typer.Argument(..., help="Year to delete, e.g. '2024'."),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
) -> None:
    """Delete every document in a year namespace so it can be re-imported.

    Required for any import retry, since `start_import` refuses to write into a
    namespace that already exists. Document-schema indexes have no namespace-delete
    endpoint yet, so this deletes the documents via `delete_all` on that namespace.
    """
    _require_dataset(dataset)
    if not yes:
        typer.confirm(
            f"Delete all documents in namespace {namespace!r} of "
            f"{DATASETS[dataset]}?",
            abort=True,
        )
    pc = fts.client()
    host = _index_host(dataset)
    idx = pc.preview.index(name=DATASETS[dataset])
    idx.documents.delete(namespace=namespace, delete_all=True)
    typer.echo(
        f"deleted all documents in namespace {namespace!r}\n"
        f"note: the namespace itself may linger until empty; if the re-import still "
        f"reports 'already exists', wait and retry."
    )
    del host


@app.command()
def verify(
    dataset: str = typer.Argument("sec", help="'sec' or 'transcripts'."),
) -> None:
    """Compare staged document counts against what each import reported.

    `describe_index_stats` is not supported on document-schema indexes, so completeness
    is checked against each import's own `records_imported` rather than an index-wide
    aggregate. That is the stronger comparison anyway: an aggregate cannot tell a short
    year from a complete one.

    The failure this exists to catch is silent incompleteness — a namespace created by
    an earlier partial import is skipped by every later import, keeping its subset
    forever with no error raised anywhere.
    """
    _require_dataset(dataset)
    s = settings()
    manifest = upload_mod.load_manifest(_state_dir(), dataset)
    if manifest:
        expected = {
            ns: v["documents"] for ns, v in manifest["namespaces"].items()
        }
        baseline = "manifest"
    else:
        expected = upload_mod.expected_counts(s.staging_dir, dataset)
        baseline = "local staging"
    if not expected:
        typer.secho(
            "no manifest and nothing staged, so imported counts cannot be checked "
            "for completeness. Reporting what each import reported instead.",
            fg=typer.colors.YELLOW,
        )
        expected = {}

    # Pinecone's own import records are the authoritative account of what landed, and
    # they do not depend on a local checkpoint surviving.
    try:
        api = fts.ImportClient(_index_host(dataset))
        actual = import_run.imported_by_namespace(api)
        source = "pinecone import records"
    except Exception as exc:  # noqa: BLE001 - fall back rather than fail the check
        typer.secho(f"  could not read imports from Pinecone ({exc}); "
                    f"falling back to the local checkpoint", fg=typer.colors.YELLOW)
        actual, source = {}, "local checkpoint"

    if actual:
        namespaces = sorted(set(expected) | set(actual))
        rows = {
            ns: (expected.get(ns, -1), actual.get(ns, 0)) for ns in namespaces
        }
        problems = [
            ns for ns, (w, g) in rows.items() if w >= 0 and w != g
        ]
    else:
        rows, problems = import_run.reconcile(expected, _state_dir())
    typer.echo(f"  imported counts from: {source}")
    total_want = (
        sum(expected.values()) if expected
        else import_run.expected_total_from_stage(_state_dir(), dataset)
    )
    if not expected:
        baseline = "stage checkpoint (corpus total only)"
    typer.echo(f"  baseline: {baseline}")
    typer.echo(f"  {'ns':<6}{'staged':>14}{'imported':>14}{'diff':>12}")
    for namespace, (want, got) in rows.items():
        if want < 0:
            typer.echo(f"  {namespace:<6}{'—':>14}{got:>14,}{'—':>12}")
            continue
        flag = "  <-- MISMATCH" if got != want else ""
        typer.echo(f"  {namespace:<6}{want:>14,}{got:>14,}{got - want:>+12,}{flag}")
    total_got = sum(g for _, g in rows.values())
    typer.echo(
        f"  {'TOTAL':<6}{total_want:>14,}{total_got:>14,}"
        f"{total_got - total_want:>+12,}"
        + ("  <-- MISMATCH" if total_got != total_want else "")
    )
    if total_got != total_want:
        problems.append("TOTAL")

    if problems:
        typer.secho(
            f"\n{len(problems)} namespace(s) do not match what was staged: "
            f"{', '.join(problems)}\n"
            f"A year showing 0 imported was never imported, or was skipped because "
            f"its namespace already existed. To redo one:\n"
            f"  uv run finvec drop-namespace {dataset} <year>\n"
            f"  uv run finvec import {dataset} --namespaces <year>",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    typer.secho(
        "\nevery namespace matches the staged document count", fg=typer.colors.GREEN
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language or keyword query."),
    dataset: str = typer.Option("sec", help="'sec' or 'transcripts'."),
    mode: str = typer.Option(
        "hybrid", help="'text' (BM25), 'dense' (semantic), or 'hybrid'."
    ),
    years: Optional[str] = typer.Option(
        None, help="Year namespaces: '2024', '2021-2024', '2019,2021-2023'."
    ),
    ticker: Optional[str] = typer.Option(None, help="Filter to one ticker."),
    must_contain: Optional[str] = typer.Option(
        None, help="Hard lexical requirement ($match_all). Defaults to the query in hybrid mode."
    ),
    include_boilerplate: bool = typer.Option(
        False, help="Include LSH-flagged boilerplate chunks."
    ),
    merge: Optional[str] = typer.Option(
        None, help="'score', 'rrf', or 'rerank'. Default depends on mode."
    ),
    top_k: int = typer.Option(10),
) -> None:
    """Search one year namespace, or fan out across several and merge.

    The FTS API takes one namespace per request and has no `query_namespaces`, so
    multi-year search fans out client-side. How the results merge matters: cosine scores
    are comparable across namespaces, BM25 scores are not, so `text` mode defaults to a
    rank-based merge instead of a score sort. `--merge rerank` orders the union with a
    cross-encoder, which sidesteps comparability entirely.
    """
    _require_dataset(dataset)
    if mode not in ("text", "dense", "hybrid"):
        raise typer.BadParameter("mode must be 'text', 'dense', or 'hybrid'")
    if merge is not None and merge not in ("score", "rrf", "rerank"):
        raise typer.BadParameter("merge must be 'score', 'rrf', or 'rerank'")

    namespaces = search_mod.year_namespaces(years, default=[])
    if not namespaces:
        namespaces = upload_mod.staged_namespaces(settings().staging_dir, dataset)
    if not namespaces:
        raise typer.BadParameter(
            "no year namespaces given and none inferable from staging; pass --years"
        )

    pc = fts.client()
    idx = pc.preview.index(name=DATASETS[dataset])
    hits, strategy = search_mod.search(
        pc, idx, query, namespaces,
        mode=mode, top_k=top_k, must_contain=must_contain,
        ticker=ticker, include_boilerplate=include_boilerplate,
        merge=merge,  # type: ignore[arg-type]
    )

    typer.echo(
        f"\n{mode} over {len(namespaces)} namespace(s) "
        f"[{namespaces[0]}..{namespaces[-1]}] · merged by {strategy} · "
        f"{len(hits)} result(s)\n"
    )
    if strategy == "rrf":
        typer.secho(
            "  note: BM25 scores are not comparable across namespaces, so results are "
            "interleaved by rank. Use --merge rerank for a single global ordering.",
            fg=typer.colors.YELLOW,
        )
        typer.echo("")
    for i, hit in enumerate(hits, start=1):
        f = hit.fields
        label = f.get("ticker", "?")
        when = f.get("call_date") or f.get("fiscal_year")
        extra = f" Q{int(f['quarter'])}" if f.get("quarter") else ""
        speaker = f" · {f['speaker']}" if f.get("speaker") else ""
        flags = " · table" if f.get("is_table") else ""
        flags += " · boilerplate" if f.get("is_boilerplate") else ""
        typer.secho(
            f"{i:>3}. {label} {when}{extra}{speaker}  "
            f"[{hit.ordering_score:.4f}] ns={hit.namespace} rank={hit.rank}{flags}",
            fg=typer.colors.CYAN,
        )
        text = hit.text.replace("\n", " ")
        typer.echo(f"     {text[:240]}{'…' if len(text) > 240 else ''}\n")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
