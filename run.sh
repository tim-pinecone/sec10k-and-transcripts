#!/usr/bin/env bash
#
# End-to-end: AWS login -> embed -> S3 -> Pinecone.
#
# Every stage is individually resumable, so this whole script is safe to re-run: it
# picks up where it stopped rather than redoing paid work. Kill it at any point.
#
#   ./run.sh                      full corpus, foreground, logged
#   ./run.sh --shards 0-9         smoke run over 10 companies (a few cents)
#   ./run.sh --detach --yes       background; monitor with ./watch.sh
#   ./run.sh --stages stage,upload    run only some stages
#   ./run.sh --prune              delete local staging after a verified upload
#
set -euo pipefail

cd "$(dirname "$0")"

SHARDS=""
DETACH=0
ASSUME_YES=0
PRUNE=0
CONCURRENCY=8
STAGES="preflight,stage,compact,probe,s3,upload,import,verify"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shards)      SHARDS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --stages)      STAGES="$2"; shift 2 ;;
    --detach)      DETACH=1; shift ;;
    --prune)       PRUNE=1; shift ;;
    --yes|-y)      ASSUME_YES=1; shift ;;
    -h|--help)     sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p logs data/state
LOG="logs/run-$(date +%Y%m%d-%H%M%S).log"
ln -sf "$(basename "$LOG")" logs/latest.log

# ── Detach ───────────────────────────────────────────────────────────────────
# Re-exec under nohup so the run survives the terminal closing. Requires --yes,
# because a background job cannot answer a confirmation prompt.
if [[ $DETACH -eq 1 ]]; then
  if [[ $ASSUME_YES -ne 1 ]]; then
    echo "--detach requires --yes (a background run cannot answer a prompt)" >&2
    exit 2
  fi
  args=(--yes --concurrency "$CONCURRENCY" --stages "$STAGES")
  [[ -n "$SHARDS" ]] && args+=(--shards "$SHARDS")
  [[ $PRUNE -eq 1 ]] && args+=(--prune)
  nohup "$0" "${args[@]}" >>"$LOG" 2>&1 &
  echo "started in background · pid $!"
  echo "  log:     tail -f logs/latest.log"
  echo "  monitor: ./watch.sh"
  exit 0
fi

exec > >(tee -a "$LOG") 2>&1

say()  { printf '\n\033[1m== %s\033[0m  %s\n' "$1" "$(date +%H:%M:%S)"; }
note() { printf '   %s\n' "$1"; }
has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

trap 'printf "\n\033[31mFAILED\033[0m at %s — see %s\nRe-run the same command; completed work is skipped.\n" "$(date +%H:%M:%S)" "$LOG"' ERR

# ── Load .env ────────────────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
  echo "no .env — copy .env.example to .env and fill it in" >&2; exit 1
fi
set -a; source .env; set +a
# An exported-but-empty access key makes boto3 ignore AWS_PROFILE and then fail
# with a confusing credentials error, so clear the empties.
[[ -z "${AWS_ACCESS_KEY_ID:-}" ]] && unset AWS_ACCESS_KEY_ID || true
[[ -z "${AWS_SECRET_ACCESS_KEY:-}" ]] && unset AWS_SECRET_ACCESS_KEY || true

STAGE_ARGS=(--concurrency "$CONCURRENCY")
[[ -n "$SHARDS" ]] && STAGE_ARGS+=(--shards "$SHARDS")

# ── Preflight ────────────────────────────────────────────────────────────────
if has_stage preflight; then
  say "preflight"
  for cmd in uv aws; do
    command -v "$cmd" >/dev/null || { echo "$cmd not found on PATH" >&2; exit 1; }
  done
  for var in OPENAI_API_KEY PINECONE_API_KEY S3_BUCKET AWS_REGION; do
    [[ -n "${!var:-}" ]] || { echo "$var is not set in .env" >&2; exit 1; }
  done
  note "keys present · bucket ${S3_BUCKET} · region ${AWS_REGION}"

  # Staging the full corpus needs roughly 30 GB. Finding that out at 90% would be
  # an expensive way to learn it.
  avail_gb=$(df -g . | awk 'NR==2 {print $4}')
  need_gb=$([[ -n "$SHARDS" ]] && echo 2 || echo 40)
  note "disk free: ${avail_gb} GB (need ~${need_gb} GB)"
  if (( avail_gb < need_gb )); then
    echo "not enough free disk: ${avail_gb} GB available, ~${need_gb} GB needed" >&2
    exit 1
  fi

  say "aws login"
  if aws sts get-caller-identity >/dev/null 2>&1; then
    note "already authenticated as $(aws sts get-caller-identity --query Arn --output text)"
  else
    note "SSO token missing or expired — logging in"
    aws sso login ${AWS_PROFILE:+--profile "$AWS_PROFILE"}
    note "authenticated as $(aws sts get-caller-identity --query Arn --output text)"
  fi

  if [[ $ASSUME_YES -ne 1 ]]; then
    say "cost check"
    if [[ -n "$SHARDS" ]]; then
      note "smoke run over shards ${SHARDS} — cents"
    else
      note "full corpus: ~\$26 OpenAI embeddings + ~\$8 Pinecone import,"
      note "then ~\$11/mo storage. Pinecone Standard plan required."
    fi
    read -r -p "   proceed? [y/N] " reply
    [[ "$reply" == [yY]* ]] || { echo "aborted"; exit 0; }
  fi
fi

# ── Embed ────────────────────────────────────────────────────────────────────
if has_stage stage; then
  say "stage — merge + embed + parquet"
  note "resumable per shard; progress mirrored to data/state/status.json"
  uv run finvec stage sec "${STAGE_ARGS[@]}"
fi

if has_stage compact; then
  say "compact — coalesce per-shard parts"
  uv run finvec compact sec
fi

# ── Schema gate ──────────────────────────────────────────────────────────────
# Round-trips real staged documents through documents.upsert before anything large
# is uploaded. FTS schemas are immutable, and import validates documents through the
# same code path as upsert, so a clean probe here means a clean import later.
if has_stage probe; then
  say "probe-schema — validate schema + document contract on live index"
  uv run finvec probe-schema sec
fi

# ── S3 ───────────────────────────────────────────────────────────────────────
if has_stage s3; then
  say "s3-setup — public read-only bucket"
  uv run finvec s3-setup --apply
fi

if has_stage upload; then
  say "upload — staging to s3://${S3_BUCKET}/${S3_PREFIX}"
  uv run finvec upload sec
fi

if [[ $PRUNE -eq 1 ]] && has_stage upload; then
  say "prune — reclaim disk from verified uploads"
  uv run finvec prune sec --apply
fi

# ── Pinecone ─────────────────────────────────────────────────────────────────
if has_stage import; then
  say "import — one bulk import per year namespace"
  note "each import takes at least 10 minutes; all years run concurrently"
  uv run finvec import sec
fi

if has_stage verify; then
  say "verify — per-namespace record counts"
  uv run finvec verify sec
fi

say "done"
note "log: $LOG"
