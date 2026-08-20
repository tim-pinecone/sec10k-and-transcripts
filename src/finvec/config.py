"""Configuration: .env for secrets, module constants for architecture decisions.

Architecture constants live here rather than in a YAML file because they are not
tuning knobs — several are one-way doors (metric, dimension) and changing one after
ingest means re-embedding or re-importing everything.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

# ── Architecture (see CLAUDE.md; changing these invalidates existing indexes) ──

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536
# dotproduct, not cosine: OpenAI embeddings are L2-normalized so the two are
# numerically identical here, but only dotproduct permits adding sparse vectors to
# the same index later for hybrid search.
METRIC = "dotproduct"
# Bulk import from S3 only reaches AWS-hosted indexes.
CLOUD = "aws"

SEC_INDEX = "sec-10k-index"
TRANSCRIPTS_INDEX = "sp500-transcripts-index"

SEC_DATASET = "astr010/sec-10k-lsh-chunks"
SEC_SHARD_COUNT = 1380
TRANSCRIPTS_DATASET = "Bose345/sp500_earnings_transcripts"

# Transcript chunking. Chunks never cross a speaker turn.
CHUNK_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50

# ── Hard limits from the Pinecone API (do not raise these) ────────────────────

MAX_METADATA_BYTES = 40 * 1024
MAX_ID_CHARS = 512
MAX_UPSERT_BYTES = 2 * 1024 * 1024
MAX_UPSERT_RECORDS = 1000
MAX_IMPORT_FILE_BYTES = 10 * 1024 * 1024 * 1024

# Target size for a staged parquet file: small enough to parallelize and to lose
# little work on a crash, large enough to stay well under the 100k-file import cap.
TARGET_STAGE_FILE_BYTES = 400 * 1024 * 1024

# ── Cost rates, for the `profile` projection (USD; verify against pinecone.io) ─

USD_PER_GB_IMPORT = 0.25
USD_PER_GB_MONTH_STORAGE = 0.33
USD_PER_MILLION_WRITE_UNITS = 4.00
USD_PER_MILLION_READ_UNITS = 16.00
USD_PER_MILLION_EMBED_TOKENS = 0.02


class Settings(BaseSettings):
    """Secrets and deployment-specific values, read from the environment."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    pinecone_api_key: str = ""
    # Not needed while the staging bucket is public: Pinecone's docs state an
    # Integration ID isn't required to import from a public bucket. Kept for the
    # private-bucket fallback, which also needs an IAM role trusting account
    # 713131977538.
    pinecone_storage_integration_id: str = ""
    openai_api_key: str = ""

    aws_region: str = "us-east-1"
    # The index region should match the bucket region: S3 to an AWS service in the
    # same region transfers free, so a mismatch adds cross-region egress to every
    # import for no benefit.
    pinecone_region: str = ""
    s3_bucket: str = ""
    s3_prefix: str = "sec10k-and-transcripts"

    staging_dir: Path = Field(default=Path("staging"))
    state_dir: Path = Field(default=Path("data/state"))

    @property
    def index_region(self) -> str:
        """Index region, defaulting to the bucket's so transfer stays free."""
        return self.pinecone_region or self.aws_region

    def require(self, *names: str) -> None:
        """Fail loudly and early, before a long job starts, not midway through."""
        missing = [n for n in names if not getattr(self, n, None)]
        if missing:
            raise SystemExit(
                "Missing required environment variables: "
                + ", ".join(n.upper() for n in missing)
                + "\nSee .env.example and copy it to .env."
            )


@lru_cache
def settings() -> Settings:
    return Settings()
