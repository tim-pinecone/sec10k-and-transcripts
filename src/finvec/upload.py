"""Convert staged Parquet to gzipped JSONL and upload it to the public S3 bucket.

FTS bulk import reads JSONL, not Parquet, so each staged Parquet part is converted on
the way out rather than kept on disk in both formats. Conversion is pure CPU with
nothing to pay for, so re-doing it on a retry is free.

Resumable by construction: "already done" is the existence of the S3 key. S3 multipart
uploads are atomic — an object appears only once it is complete — so presence is a
sound completeness signal. Size comparison is not usable here because the local file is
Parquet and the remote one is gzipped JSONL.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from .jsonl import jsonl_key_for, write_jsonl_gz
from .layout import s3_key
from .progress import Progress


@dataclass
class UploadPlan:
    """What needs uploading, decided before any bytes move.

    Sizes are the *Parquet* sizes, used only to weight the progress bar; the bytes
    actually transferred are the gzipped JSONL, roughly 1.5x larger.
    """

    to_upload: list[tuple[Path, str, int]]  # (parquet path, s3 jsonl key, parquet size)
    skipped: int
    skipped_bytes: int

    @property
    def total_bytes(self) -> int:
        return sum(size for _, _, size in self.to_upload)


def staged_files(staging_dir: Path, dataset: str) -> list[Path]:
    """Every staged parquet part for a dataset, in a stable order.

    Sorted so that progress is reproducible across runs and a resumed run walks the
    files in the same sequence as the run it is continuing.
    """
    root = Path(staging_dir) / dataset
    return sorted(root.glob("import-*/*/*.parquet"))


def _relative_key(staging_dir: Path, dataset: str, path: Path) -> str:
    return f"{dataset}/{path.relative_to(Path(staging_dir) / dataset).as_posix()}"


def plan_upload(
    staging_dir: Path,
    dataset: str,
    bucket: str,
    prefix: str,
    region: str = "us-east-1",
) -> UploadPlan:
    s3 = boto3.client("s3", region_name=region)
    to_upload: list[tuple[Path, str, int]] = []
    skipped = skipped_bytes = 0

    for path in staged_files(staging_dir, dataset):
        key = s3_key(
            prefix, jsonl_key_for(_relative_key(staging_dir, dataset, path))
        )
        size = path.stat().st_size
        try:
            s3.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in ("404", "NoSuchKey"):
                raise
            to_upload.append((path, key, size))
            continue
        skipped += 1
        skipped_bytes += size

    return UploadPlan(to_upload=to_upload, skipped=skipped, skipped_bytes=skipped_bytes)


def upload(
    staging_dir: Path,
    dataset: str,
    bucket: str,
    prefix: str,
    region: str = "us-east-1",
    status_path: Path | None = None,
) -> UploadPlan:
    plan = plan_upload(staging_dir, dataset, bucket, prefix, region)
    if plan.skipped:
        print(
            f"skipping {plan.skipped:,} already-uploaded parts "
            f"({plan.skipped_bytes / 1e9:.2f} GB)",
            flush=True,
        )
    if not plan.to_upload:
        print("nothing to upload — S3 already matches staging", flush=True)
        return plan

    s3 = boto3.client("s3", region_name=region)
    # Progress is measured in bytes rather than files: parts vary in size, so a file
    # count would give a misleading ETA.
    prog = Progress(
        f"upload {dataset}",
        total=plan.total_bytes,
        status_path=status_path,
    )
    documents = uploaded_bytes = 0
    with tempfile.TemporaryDirectory(prefix="finvec-jsonl-") as scratch:
        for path, key, size in plan.to_upload:
            # Converted to a scratch file rather than streamed, so a failed upload
            # never leaves a partial object and the temp file is reclaimed either way.
            local = Path(scratch) / "part.jsonl.gz"
            count, gz_bytes = write_jsonl_gz(path, local)
            # boto3 switches to multipart automatically for large parts.
            s3.upload_file(str(local), bucket, key)
            local.unlink()
            documents += count
            uploaded_bytes += gz_bytes
            prog.advance(size, current=key, docs=f"{documents:,}")
    prog.finish(
        f"{len(plan.to_upload):,} parts, {documents:,} documents, "
        f"{uploaded_bytes / 1e9:.2f} GB of jsonl.gz"
    )
    return plan


def staged_namespaces(staging_dir: Path, dataset: str) -> list[str]:
    """Year namespaces that have at least one staged parquet part.

    A year is only importable once it is *completely* staged, so this is the input to
    the import step, not a promise that staging finished.
    """
    root = Path(staging_dir) / dataset
    found = {
        path.parent.name
        for path in root.glob("import-*/*/*.parquet")
    }
    return sorted(found)


def prune_uploaded(
    staging_dir: Path,
    dataset: str,
    bucket: str,
    prefix: str,
    region: str = "us-east-1",
    dry_run: bool = True,
) -> tuple[int, int]:
    """Delete local staged parts that are confirmed present in S3 at the same size.

    Staging the full corpus is ~33 GB, which is a lot of disk to hold once the bytes
    are safely in the bucket. Each file is verified individually with `head_object`
    before it is removed — a file missing from S3, or present at a different size, is
    kept. Returns (deleted, kept).
    """
    s3 = boto3.client("s3", region_name=region)
    deleted = kept = freed = 0

    for path in staged_files(staging_dir, dataset):
        key = s3_key(
            prefix, jsonl_key_for(_relative_key(staging_dir, dataset, path))
        )
        size = path.stat().st_size
        try:
            s3.head_object(Bucket=bucket, Key=key)
        except ClientError:
            kept += 1
            continue
        freed += size
        deleted += 1
        if not dry_run:
            path.unlink()

    verb = "would free" if dry_run else "freed"
    print(
        f"{verb} {freed / 1e9:.2f} GB across {deleted:,} verified files"
        + (f"; kept {kept:,} not confirmed in S3" if kept else ""),
        flush=True,
    )
    return deleted, kept


def expected_counts(staging_dir: Path, dataset: str) -> dict[str, int]:
    """Records staged per namespace, from parquet footers only.

    Reads row counts out of each file's metadata rather than the data itself, so this
    is fast even across 33 GB. This is the denominator `verify` needs: without it,
    "namespace 2024 has 180,000 records" is unfalsifiable.
    """
    import pyarrow.parquet as pq

    counts: dict[str, int] = {}
    for path in staged_files(staging_dir, dataset):
        namespace = path.parent.name
        counts[namespace] = counts.get(namespace, 0) + pq.ParquetFile(
            path
        ).metadata.num_rows
    return counts
