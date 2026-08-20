"""Upload staged parquet to the public S3 bucket.

Resumable by construction: the check for "already done" is a `head_object` against S3
comparing size, so the bucket itself is the source of truth. A local checkpoint would
be a second, weaker copy of that same fact — and would happily skip a file that was
never actually uploaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from .layout import s3_key
from .progress import Progress


@dataclass
class UploadPlan:
    """What needs uploading, decided before any bytes move."""

    to_upload: list[tuple[Path, str, int]]  # (local path, s3 key, size)
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
        key = s3_key(prefix, _relative_key(staging_dir, dataset, path))
        size = path.stat().st_size
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in ("404", "NoSuchKey"):
                raise
            to_upload.append((path, key, size))
            continue
        # Same size means already uploaded. A truncated upload would differ in size,
        # because parquet parts are written locally by atomic rename.
        if head["ContentLength"] == size:
            skipped += 1
            skipped_bytes += size
        else:
            to_upload.append((path, key, size))

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
    for path, key, size in plan.to_upload:
        # boto3's upload_file switches to multipart automatically for large parts.
        s3.upload_file(str(path), bucket, key)
        prog.advance(size, current=key)
    prog.finish(f"{len(plan.to_upload):,} parts, {plan.total_bytes / 1e9:.2f} GB")
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
