#!/usr/bin/env bash
# URA-xLaw staged pipeline. Expensive/network stages run only when requested.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

MODE="${1:-validate}"
PROCESSED="data/processed"
mkdir -p "$PROCESSED"

generate() {
  [[ -d data/raw/congbobanan ]] || {
    echo "Missing local raw judgments: data/raw/congbobanan" >&2
    exit 1
  }
  [[ -n "${OPENAI_API_KEY:-}" ]] || {
    echo "OPENAI_API_KEY is required for QA generation" >&2
    exit 1
  }
  "$PY" -m ura_xlaw clean-judgments
  "$PY" -m ura_xlaw generate-qa \
    --provider openai \
    --model "${URA_XLAW_MODEL:-gpt-4.1}" \
    --concurrency "${URA_XLAW_CONCURRENCY:-8}"
}

build_answerable() {
  [[ -f "$PROCESSED/qa_generated_openai.jsonl" ]] || {
    echo "Run '$0 generate' first" >&2
    exit 1
  }
  "$PY" -m ura_xlaw map-citations \
    --corpus data/corpus_full.parquet \
    --supp "$PROCESSED/__no_supplement__.parquet" \
    --input "$PROCESSED/qa_generated_openai.jsonl" \
    --output "$PROCESSED/qa_mapped.jsonl" \
    --report "$PROCESSED/law_chunks_coverage.json"
  "$PY" -m ura_xlaw export-answerable \
    --input "$PROCESSED/qa_mapped.jsonl" \
    --output-jsonl "$PROCESSED/qa_answerable.jsonl" \
    --output-parquet "$PROCESSED/qa_answerable.parquet" \
    --output-corpus "$PROCESSED/law_corpus_final.parquet" \
    --output-stats "$PROCESSED/qa_answerable_stats.json" \
    --primary-corpus data/corpus_full.parquet \
    --supplemental-corpus "$PROCESSED/__no_supplement__.parquet"
  "$PY" -m ura_xlaw split \
    --input "$PROCESSED/qa_answerable.jsonl" \
    --output-dir "$PROCESSED"
}

build_unanswerable() {
  [[ -f "$PROCESSED/law_corpus_qa_only.parquet" ]] || \
    cp data/corpus_rag.parquet "$PROCESSED/law_corpus_qa_only.parquet"
  "$PY" -m ura_xlaw build-real-unanswerable \
    --pick "${URA_XLAW_UNANSWERABLE_DOCS:-100}" \
    --model "${URA_XLAW_MODEL:-gpt-4.1}" \
    --concurrency "${URA_XLAW_CONCURRENCY:-8}" \
    --merge-train
}

publish() {
  [[ -f "$PROCESSED/train.jsonl" && -f "$PROCESSED/test.jsonl" ]] || {
    echo "Processed train/test are missing; run '$0 build' first" >&2
    exit 1
  }
  # The constrained corpus is intentionally fixed by the benchmark release.
  [[ -f "$PROCESSED/law_corpus_qa_only.parquet" ]] || \
    cp data/corpus_rag.parquet "$PROCESSED/law_corpus_qa_only.parquet"
  [[ -f "$PROCESSED/law_corpus_final.parquet" ]] || \
    cp data/corpus_full.parquet "$PROCESSED/law_corpus_final.parquet"
  [[ -f "$PROCESSED/unanswerable_real.jsonl" ]] || \
    cp data/unanswerable_real.jsonl "$PROCESSED/unanswerable_real.jsonl"

  "$PY" -m ura_xlaw normalize-grounding
  "$PY" -m ura_xlaw package-release --force
  "$PY" -m ura_xlaw validate-release
}

case "$MODE" in
  validate) "$PY" -m ura_xlaw validate-release ;;
  generate) generate ;;
  build) build_answerable ;;
  unanswerable) build_unanswerable ;;
  publish) publish ;;
  all) generate; build_answerable; build_unanswerable; publish ;;
  *)
    echo "Usage: $0 {validate|generate|build|unanswerable|publish|all}" >&2
    exit 2
    ;;
esac
