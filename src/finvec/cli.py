"""`finvec` command line.

The pipeline is deliberately staged rather than one command: each stage is resumable on
its own, and the expensive stages sit behind the free `profile` gate.

    profile -> stage -> s3-setup -> upload -> import -> verify -> search
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from . import profile as profile_mod
from . import pinecone_ops, s3_setup, upload as upload_mod
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


def _open_index(dataset: str):
    """Resolve the index host and return a data-plane handle.

    `pool_threads` matters here: `query_namespaces` fans out one request per namespace
    and serializes without a pool big enough to hold them, which erases the whole
    point of the year-partitioned design.
    """
    pc = pinecone_ops.client()
    host = pinecone_ops.ensure_index(
        pc, DATASETS[dataset], region=settings().index_region
    )
    return pc.Index(host=host, pool_threads=32, connection_pool_maxsize=32)


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
    dataset: str = typer.Argument(..., help="'sec' or 'transcripts'."),
    shards: Optional[str] = typer.Option(
        None, help="Shard range for a smoke run, e.g. '0-4'. Default: all."
    ),
) -> None:
    """Merge, embed, and write staging/{dataset}/import-{year}/{year}/*.parquet."""
    _require_dataset(dataset)
    typer.secho(
        "`stage` is not implemented yet. The S3 and import stages below are ready; "
        "staging (merge + embed + parquet) is next.",
        fg=typer.colors.YELLOW,
    )
    raise typer.Exit(1)


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
    """Upload staged parquet to S3. Resumable — S3 itself is the source of truth."""
    _require_dataset(dataset)
    s = settings()
    s.require("s3_bucket")
    with s3_setup.friendly_aws_errors():
        upload_mod.upload(
            s.staging_dir,
            dataset,
            s.s3_bucket,
            s.s3_prefix,
            region=s.aws_region,
            status_path=_state_dir() / "status.json",
        )


@app.command("import")
def import_cmd(
    dataset: str = typer.Argument(..., help="'sec' or 'transcripts'."),
    namespaces: Optional[str] = typer.Option(
        None, help="Comma-separated years. Default: every staged year."
    ),
    interval: float = typer.Option(30.0, help="Seconds between status polls."),
    abort_on_error: bool = typer.Option(
        False, help="Fail the import on the first bad record instead of skipping it."
    ),
) -> None:
    """Start one bulk import per year namespace and poll them all to completion.

    One import per year, not one for the whole corpus: a namespace cannot be imported
    into twice, so a single multi-year import that fails partway leaves created
    namespaces blocking any retry. Per-year, a failure costs one year.

    Years that already exist in the index are skipped — delete one with
    `drop-namespace` to redo it.
    """
    _require_dataset(dataset)
    s = settings()
    s.require("s3_bucket", "pinecone_api_key")

    years = (
        [n.strip() for n in namespaces.split(",") if n.strip()]
        if namespaces
        else upload_mod.staged_namespaces(s.staging_dir, dataset)
    )
    if not years:
        typer.secho(
            f"no staged years found under {s.staging_dir}/{dataset} — run "
            f"`finvec stage {dataset}` first.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(1)

    index = _open_index(dataset)
    run = pinecone_ops.start_year_imports(
        index, dataset, years, s.s3_bucket, s.s3_prefix, abort_on_error=abort_on_error
    )
    if not run.jobs:
        typer.echo("nothing to import")
        return
    run = pinecone_ops.poll_imports(
        index, run, interval=interval, state_dir=_state_dir()
    )
    typer.echo(
        f"\nimported {run.records:,} records across "
        f"{len(run.jobs) - len(run.failed)}/{len(run.jobs)} namespaces"
    )
    if run.failed:
        raise typer.Exit(1)


@app.command("drop-namespace")
def drop_namespace_cmd(
    dataset: str = typer.Argument(..., help="'sec' or 'transcripts'."),
    namespace: str = typer.Argument(..., help="Year to delete, e.g. '2024'."),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
) -> None:
    """Delete a year namespace so it can be re-imported.

    Required for any import retry: `start_import` refuses to write into a namespace
    that already exists.
    """
    _require_dataset(dataset)
    index = _open_index(dataset)
    counts = pinecone_ops.namespace_counts(index)
    n = counts.get(namespace)
    if n is None:
        typer.echo(f"namespace {namespace!r} does not exist in {DATASETS[dataset]}")
        return
    if not yes:
        typer.confirm(
            f"Delete namespace {namespace!r} ({n:,} records) from "
            f"{DATASETS[dataset]}?",
            abort=True,
        )
    pinecone_ops.drop_namespace(index, namespace)
    typer.echo(f"deleted namespace {namespace!r} ({n:,} records)")


@app.command()
def verify(
    dataset: str = typer.Argument(..., help="'sec' or 'transcripts'."),
) -> None:
    """Report per-namespace record counts from the live index."""
    _require_dataset(dataset)
    index = _open_index(dataset)
    counts = pinecone_ops.namespace_counts(index)
    if not counts:
        typer.echo("no namespaces yet")
        return
    total = sum(counts.values())
    for namespace in sorted(counts):
        typer.echo(f"  {namespace}  {counts[namespace]:>12,}")
    typer.echo(f"  {'TOTAL':<5} {total:>12,} across {len(counts)} namespaces")


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language query."),
    dataset: str = typer.Option("sec", help="'sec' or 'transcripts'."),
    year: Optional[int] = typer.Option(None, help="Single year namespace."),
    years: Optional[str] = typer.Option(None, help="Year range, e.g. '2021-2024'."),
    ticker: Optional[str] = typer.Option(None, help="Filter by ticker."),
    include_boilerplate: bool = typer.Option(
        False, help="SEC only: include LSH-flagged boilerplate chunks."
    ),
    top_k: int = typer.Option(10),
) -> None:
    """Query one year, or fan out across a year range via query_namespaces."""
    _require_dataset(dataset)
    typer.secho("`search` is not implemented yet.", fg=typer.colors.YELLOW)
    raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
