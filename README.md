# URA-xLaw

URA-xLaw is a Vietnamese legal retrieval-augmented generation dataset grounded in real court judgments and Vietnamese legal documents. The repository contains both the public release and the reproducible construction/validation pipeline.

## Dataset

| Artifact | Size | Description |
| --- | ---: | --- |
| `train.jsonl` | 4,788 QA | Answerable training set |
| `test.jsonl` | 952 QA | Answerable test set |
| `unanswerable_real.jsonl` | 136 QA | Unanswerable questions derived from real judgments |
| `corpus_full.parquet` | 32,587 chunks | Full corpus covering 1,580 legal documents |
| `corpus_rag.parquet` | 1,313 chunks | Fixed RAG corpus covering 140 legal documents |

All release files are stored in [`data/`](data). SHA-256 checksums are recorded in [`data/manifest.json`](data/manifest.json).

## Quick validation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
./scripts/run_pipeline.sh validate
```

The validation command reads every JSONL and Parquet file, verifies manifest checksums and counts, checks train/test leakage, and confirms that every answerable QA resolves to its gold chunks in `corpus_rag.parquet`.

## Pipeline

Install optional crawling and generation dependencies when rebuilding data:

```bash
pip install -e ".[all]"
playwright install chromium
cp .env.example .env
```

Pipeline modes:

```bash
./scripts/run_pipeline.sh validate       # test the published release
./scripts/run_pipeline.sh generate       # raw judgments → generated QA
./scripts/run_pipeline.sh build          # citation mapping → answerable splits
./scripts/run_pipeline.sh unanswerable   # build real unanswerable QA
./scripts/run_pipeline.sh publish        # normalize, package, and validate
./scripts/run_pipeline.sh all            # run all construction stages
```

Raw crawl data remains local under `data/raw/` and is ignored by Git. Intermediate pipeline artifacts are written to the ignored `data/processed/` directory.

The package also exposes individual stages through `ura-xlaw --help`. Parallel court-index scanning is available through `scripts/parallel_scan.sh`.

## Task definition

Each answerable QA is grounded to one or more legal chunks through `positive_chunk_ids` and `positives`. The fixed RAG corpus is used for retrieval experiments.

A real QA is labeled unanswerable when at least one legal basis required by the judgment is unavailable from the fixed RAG corpus. This supports evaluation of retrieval failure detection and abstention.

## QA schema

Important fields include:

- `qa_id`, `doc_id`: QA and source-judgment identifiers
- `situation`, `question`, `answer`: legal scenario and response
- `legal_reasoning`: explanation grounded in the judgment
- `law_applied`: cited legal provisions
- `positive_chunk_ids`, `positives`: retrieval ground truth
- `legal_category`, `complexity_level`: task metadata
- `case_number`, `court`, `case_type`, `trial_level`: judgment metadata

## Recommended evaluation

- Train on `train.jsonl`.
- Evaluate answerable performance on `test.jsonl`.
- Combine `test.jsonl` with `unanswerable_real.jsonl` for a 1,088-question answerability/abstention benchmark.
- Index `corpus_rag.parquet` for the constrained RAG setting.
- Use `corpus_full.parquet` for corpus-coverage analysis or full-corpus experiments.

## Data provenance

QA contexts are derived from real Vietnamese court judgments. Legal chunks are derived from Vietnamese legal-document corpora and supplemental legal sources. Users are responsible for complying with the terms and rights associated with the original sources.

## Repository layout

```text
data/                 Public URA-xLaw release
data/prompts/         QA generation prompt
src/ura_xlaw/         Pipeline implementation
scripts/              Staged and parallel pipeline entry points
tests/                Dataset and pipeline smoke tests
```
