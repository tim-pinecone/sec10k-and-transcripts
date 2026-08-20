"""`finvec` command line.

The pipeline is deliberately staged rather than a single command: each stage is
resumable on its own, and the expensive stages sit behind the free `profile` gate.

    profile -> stage -> upload -> import -> verify -> search
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from . import profile as profile_mod
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


@app.command()
def profile(
    shards: int = typer.Option(12, help="Number of SEC shards to sample (of 1380)."),
) -> None:
    """Measure the corpus and project cost. Spends nothing, needs no API keys.

    Run this before `stage`. It reports real token counts, real metadata sizes, and
    the boilerplate fraction, then prices the run from those instead of from
    assumptions.
    """
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
    yes: bool = typer.Option(False, "--yes", help="Skip the cost confirmation."),
) -> None:
    """Embed records and write staging/{dataset}/{year}/*.parquet. Resumable.

    Checkpoints per shard, so killing this and re-running the same command skips
    completed shards instead of re-paying for their embeddings.
    """
    _require_dataset(dataset)
    raise typer.Exit(_todo("stage"))


@app.command("upload")
def upload_cmd(
    dataset: str = typer.Argument(..., help="'sec' or 'transcripts'."),
) -> None:
    """Sync staging/{dataset} to s3://$S3_BUCKET/$S3_PREFIX/{dataset}. Resumable."""
    _require_dataset(dataset)
    raise typer.Exit(_todo("upload"))


@app.command("import")
def import_cmd(
    dataset: str = typer.Argument(..., help="'sec' or 'transcripts'."),
    namespaces: Optional[str] = typer.Option(
        None, help="Comma-separated years. Default: every staged year."
    ),
) -> None:
    """Start a bulk import per year namespace and poll to completion.

    Bulk import can only target a namespace that does not yet exist, so a year must be
    fully staged first, and re-importing a year means deleting that namespace.
    """
    _require_dataset(dataset)
    raise typer.Exit(_todo("import"))


@app.command()
def verify(
    dataset: str = typer.Argument(..., help="'sec' or 'transcripts'."),
) -> None:
    """Compare per-namespace record counts against what staging says was written."""
    _require_dataset(dataset)
    raise typer.Exit(_todo("verify"))


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
    raise typer.Exit(_todo("search"))


def _require_dataset(dataset: str) -> None:
    if dataset not in DATASETS:
        raise typer.BadParameter(
            f"unknown dataset {dataset!r}; expected one of {', '.join(DATASETS)}"
        )


def _todo(stage_name: str) -> int:
    typer.secho(
        f"`{stage_name}` is not implemented yet — the scaffold, config, checkpointing,\n"
        f"ID/limit validation and the source readers are in place; the API-touching\n"
        f"stages are next. `finvec profile` works today.",
        fg=typer.colors.YELLOW,
    )
    return 1


def main() -> None:
    app()


if __name__ == "__main__":
    main()
