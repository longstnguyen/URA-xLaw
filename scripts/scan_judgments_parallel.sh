#!/usr/bin/env bash
# Scan disjoint court-judgment ID ranges in parallel and merge the index.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

TOTAL="${1:-6000}"
WORKERS="${2:-4}"
START_TOP="${3:-}"
DATA_DIR="data/raw/congbobanan"
LOG_DIR="logs"
mkdir -p "$DATA_DIR" "$LOG_DIR"

if [[ -z "$START_TOP" ]]; then
  START_TOP=$("$PY" - <<'PY'
from ura_xlaw.acquisition.court_judgments import CongBoBanAnCrawler
crawler = CongBoBanAnCrawler()
crawler._start_browser()
try:
    print(crawler.discover_latest_id())
finally:
    crawler._stop_browser()
PY
)
fi

PER_WORKER=$(( (TOTAL + WORKERS - 1) / WORKERS ))
PIDS=()
for ((worker=0; worker<WORKERS; worker++)); do
  start=$(( START_TOP - worker * PER_WORKER ))
  "$PY" -m ura_xlaw crawl-judgments \
    --scan-only \
    --strategy probe \
    --start-id "$start" \
    --limit "$PER_WORKER" \
    --index-filename "index_part_${worker}.jsonl" \
    > "$LOG_DIR/scan_${worker}.log" 2>&1 &
  PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || failed=$((failed + 1))
done
[[ "$failed" -eq 0 ]] || echo "Warning: $failed scan workers failed" >&2

"$PY" - <<'PY'
from pathlib import Path
import json

root = Path("data/raw/congbobanan")
rows = {}
paths = sorted(root.glob("index_part_*.jsonl"))
if (root / "index.jsonl").exists():
    paths.append(root / "index.jsonl")
for path in paths:
    with path.open(encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = str(row.get("id", ""))
            if doc_id:
                rows.setdefault(doc_id, row)
with (root / "index.jsonl").open("w", encoding="utf-8") as output:
    for row in rows.values():
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Merged {len(rows):,} judgments into {root / 'index.jsonl'}")
PY
