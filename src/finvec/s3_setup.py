"""Create and configure the public S3 bucket that serves the staged parquet.

Why public: Pinecone's docs are explicit that "an Integration ID isn't needed to
import from a public bucket." A public bucket therefore removes the entire AWS setup
step — IAM policy, cross-account trust role for Pinecone's account 713131977538, and
a per-project storage integration — for us *and* for anyone else who wants to load
this corpus. They just call `start_import(uri=...)` with no AWS account at all.

What "public" means here, precisely:

- Read-only. The bucket policy grants `s3:GetObject` and `s3:ListBucket` and nothing
  else. No `PutObject`, no `DeleteObject`, no policy or ACL changes. Anonymous callers
  can read the dataset; they cannot alter or delete it.
- Policy-based, not ACL-based. `BlockPublicAcls` and `IgnorePublicAcls` stay ON, so
  the *only* route to public access is the one explicit policy written here. Object
  ACLs cannot make anything public even by accident.

The real exposure is billing, not confidentiality: the contents are embeddings derived
from public SEC filings and public Hugging Face datasets, so there is nothing secret
in the bucket. But egress is paid by the bucket owner, so anyone can generate S3
transfer charges. Two things keep that small:

- Put the bucket in the **same region as the Pinecone index**. S3 to an AWS service in
  the same region is free, so the import path itself costs no egress. Cross-region
  would be billed per GB.
- Internet downloads (someone curl-ing the parquet) are billed at the standard egress
  rate — roughly $3 for a full copy of the merged 33 GB corpus. Set a budget alarm if
  that matters.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import boto3
from botocore.exceptions import (
    ClientError,
    NoCredentialsError,
    NoRegionError,
    TokenRetrievalError,
)

# Pinecone's AWS account, for the optional private-bucket path. Not used by the
# public-bucket flow, but recorded here so the alternative is documented in one place.
PINECONE_AWS_ACCOUNT = "713131977538"


def public_read_policy(bucket: str) -> dict[str, Any]:
    """Least-privilege anonymous read. Deliberately no write actions of any kind.

    `ListBucket` is required as well as `GetObject`: the importer enumerates the
    parquet files under a namespace directory rather than being handed a file list.
    `GetBucketLocation` lets an anonymous client resolve the bucket's region for
    SigV4 without a redirect; it discloses only the region, which bucket DNS already
    reveals.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "PublicReadForBulkImport",
                "Effect": "Allow",
                "Principal": "*",
                "Action": ["s3:GetObject"],
                "Resource": f"arn:aws:s3:::{bucket}/*",
            },
            {
                "Sid": "PublicListForBulkImport",
                "Effect": "Allow",
                "Principal": "*",
                "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
                "Resource": f"arn:aws:s3:::{bucket}",
            },
        ],
    }


# Policy-based public access only: ACLs stay blocked and ignored, so the bucket policy
# above is the single, auditable source of public access.
PUBLIC_ACCESS_BLOCK = {
    "BlockPublicAcls": True,
    "IgnorePublicAcls": True,
    "BlockPublicPolicy": False,
    "RestrictPublicBuckets": False,
}


@dataclass
class SetupReport:
    bucket: str
    region: str
    dry_run: bool
    actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            "",
            f"{'PLANNED' if self.dry_run else 'APPLIED'}: s3://{self.bucket} "
            f"({self.region})",
            "",
        ]
        lines += [f"  - {a}" for a in self.actions] or ["  (nothing to do)"]
        if self.warnings:
            lines += ["", "Warnings:"] + [f"  ! {w}" for w in self.warnings]
        lines.append("")
        return "\n".join(lines)


@contextmanager
def friendly_aws_errors():
    """Turn credential failures into one actionable line instead of a traceback.

    Expired SSO tokens are the common case, and a botocore stack trace buries the one
    thing worth knowing: which command fixes it.
    """
    try:
        yield
    except TokenRetrievalError as exc:
        raise SystemExit(
            f"AWS SSO token expired or unavailable ({exc}).\n"
            f"Run `aws sso login` (add --profile if you use one), then retry."
        ) from exc
    except NoCredentialsError as exc:
        raise SystemExit(
            "No AWS credentials found. Set AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY in .env, or configure a profile and run "
            "`aws sso login`."
        ) from exc
    except NoRegionError as exc:
        raise SystemExit("No AWS region configured. Set AWS_REGION in .env.") from exc
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("ExpiredToken", "InvalidAccessKeyId", "SignatureDoesNotMatch",
                    "AccessDenied", "InvalidClientTokenId"):
            raise SystemExit(
                f"AWS rejected the request ({code}): "
                f"{exc.response['Error'].get('Message', '')}\n"
                f"Check the credentials in .env, or re-run `aws sso login`."
            ) from exc
        raise


def caller_identity(region: str = "us-east-1") -> str:
    """Which AWS identity is about to create a public bucket. Worth printing."""
    with friendly_aws_errors():
        return boto3.client("sts", region_name=region).get_caller_identity()["Arn"]


def _client(region: str):
    return boto3.client("s3", region_name=region)


def bucket_exists(bucket: str, region: str = "us-east-1") -> bool:
    try:
        _client(region).head_bucket(Bucket=bucket)
        return True
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("404", "NoSuchBucket"):
            return False
        if code == "403":
            # Exists, owned by someone else. Bucket names are globally unique.
            raise SystemExit(
                f"Bucket {bucket!r} exists but is not accessible with these "
                f"credentials — S3 bucket names are global, so someone else owns it. "
                f"Pick a different name."
            )
        raise


def create_public_dataset_bucket(
    bucket: str, region: str = "us-east-1", dry_run: bool = True
) -> SetupReport:
    """Create the bucket if needed, then make it anonymously readable.

    Idempotent: re-running against an already-configured bucket reports no changes
    rather than failing, so it is safe to call as a setup step.
    """
    report = SetupReport(bucket=bucket, region=region, dry_run=dry_run)
    s3 = _client(region)

    exists = bucket_exists(bucket, region)
    if exists:
        report.actions.append("bucket already exists — leaving it in place")
    else:
        report.actions.append(f"create bucket in {region}")
        if not dry_run:
            # us-east-1 is the one region that rejects an explicit LocationConstraint.
            kwargs: dict[str, Any] = {"Bucket": bucket}
            if region != "us-east-1":
                kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
            try:
                s3.create_bucket(**kwargs)
            except ClientError as exc:
                if exc.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
                    raise
                report.actions[-1] = "bucket already owned by this account"

    report.actions.append(
        "public access block: policies allowed, ACLs still blocked "
        "(BlockPublicAcls=True, BlockPublicPolicy=False)"
    )
    if not dry_run:
        s3.put_public_access_block(
            Bucket=bucket, PublicAccessBlockConfiguration=PUBLIC_ACCESS_BLOCK
        )

    report.actions.append("bucket policy: anonymous GetObject + ListBucket, no writes")
    if not dry_run:
        try:
            s3.put_bucket_policy(
                Bucket=bucket, Policy=json.dumps(public_read_policy(bucket))
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "AccessDenied":
                raise SystemExit(
                    "put_bucket_policy was denied. This is almost always "
                    "account-level Block Public Access overriding the bucket "
                    "setting. Turn it off for this account in the S3 console "
                    "(Account settings for Block Public Access), or use a "
                    "dedicated AWS account for public datasets, then re-run."
                ) from exc
            raise

    report.warnings.append(
        "egress is billed to the bucket owner — keep this bucket in the same region "
        "as the Pinecone index so the import itself transfers free, and set a budget "
        "alarm for anonymous internet downloads"
    )
    return report


def verify_public_read(bucket: str, region: str = "us-east-1") -> dict[str, Any]:
    """Read back the live configuration rather than trusting that the writes stuck."""
    s3 = _client(region)
    out: dict[str, Any] = {"bucket": bucket}
    try:
        out["region"] = (
            s3.get_bucket_location(Bucket=bucket).get("LocationConstraint")
            or "us-east-1"
        )
    except ClientError as exc:
        out["region_error"] = exc.response["Error"]["Code"]
    try:
        out["public_access_block"] = s3.get_public_access_block(Bucket=bucket)[
            "PublicAccessBlockConfiguration"
        ]
    except ClientError as exc:
        out["public_access_block_error"] = exc.response["Error"]["Code"]
    try:
        policy = json.loads(s3.get_bucket_policy(Bucket=bucket)["Policy"])
        out["policy_statements"] = [s.get("Sid") for s in policy["Statement"]]
        out["grants_write"] = any(
            action.startswith(("s3:Put", "s3:Delete"))
            for stmt in policy["Statement"]
            for action in _as_list(stmt.get("Action"))
        )
    except ClientError as exc:
        out["policy_error"] = exc.response["Error"]["Code"]
    return out


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]
