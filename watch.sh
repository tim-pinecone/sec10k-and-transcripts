#!/usr/bin/env bash
#
# Live view of a run: current stage, position, rate, ETA, and the recent log tail.
#
#   ./watch.sh          refresh every 5s
#   ./watch.sh 30       refresh every 30s
#
set -euo pipefail
cd "$(dirname "$0")"

INTERVAL="${1:-5}"
STATUS="data/state/status.json"
LOG="logs/latest.log"

while true; do
  clear
  printf '\033[1mfinvec run\033[0m   %s   (refresh %ss, ctrl-c to stop)\n\n' \
    "$(date +%H:%M:%S)" "$INTERVAL"

  if [[ -f "$STATUS" ]]; then
    python3 - "$STATUS" <<'PY'
import json, sys, datetime
d = json.load(open(sys.argv[1]))
eta = d.get("eta_seconds")
eta = "?" if eta is None else str(datetime.timedelta(seconds=int(eta)))
el  = str(datetime.timedelta(seconds=int(d.get("elapsed_seconds", 0))))
bar_w = 40
filled = int(bar_w * d.get("pct", 0) / 100)
print(f"  stage    {d.get('label','?')}")
print(f"  progress [{'#'*filled}{'.'*(bar_w-filled)}] {d.get('pct',0):.1f}%")
print(f"           {d.get('done',0):,} / {d.get('total',0):,}")
print(f"  rate     {d.get('rate_per_sec',0):.2f}/s")
print(f"  elapsed  {el}      ETA {eta}")
if d.get("current"):  print(f"  current  {d['current']}")
if d.get("records"):  print(f"  records  {d['records']}")
if d.get("tokens"):   print(f"  tokens   {d['tokens']}")
if d.get("resumed_from"): print(f"  resumed  from {d['resumed_from']:,}")
PY
  else
    printf '  no status file yet (%s)\n' "$STATUS"
  fi

  if [[ -f "logs/imports.checkpoint.json" || -f "data/state/imports.checkpoint.json" ]]; then
    printf '\n\033[1m  imports\033[0m\n'
    python3 -c "
import json,glob
for f in glob.glob('data/state/imports.checkpoint.json'):
    for ns, v in sorted(json.load(open(f)).items()):
        print(f\"    {ns}  {v.get('status','?'):<10} {v.get('records',0):>12,}\")
" 2>/dev/null || true
  fi

  printf '\n\033[1m  log tail\033[0m  (%s)\n' "$LOG"
  if [[ -f "$LOG" ]]; then
    tail -n 12 "$LOG" | sed 's/^/    /'
  else
    printf '    no log yet\n'
  fi

  sleep "$INTERVAL"
done
