# URA-xLaw

URA-xLaw is a Vietnamese legal retrieval-augmented generation dataset grounded in real court judgments and Vietnamese legal documents. The repository contains both the public release and the reproducible construction/validation pipeline.

## Dataset

| Artifact | Size | Description |
| --- | ---: | --- |
| `dataset/train.jsonl` | 4,788 QA | Answerable training set |
| `dataset/test.jsonl` | 952 QA | Answerable test set |
| `dataset/unanswerable.jsonl` | 136 QA | Unanswerable questions derived from judgments |
| `dataset/corpus/full.parquet` | 32,587 chunks | Full corpus covering 1,580 legal documents |
| `dataset/corpus/retrieval.parquet` | 1,313 chunks | Fixed retrieval corpus covering 140 legal documents |

All release files are stored in [`dataset/`](dataset). SHA-256 checksums are recorded in [`dataset/manifest.json`](dataset/manifest.json).

## Quick validation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
./scripts/run_pipeline.sh validate
```

The validation command reads every JSONL and Parquet file, verifies manifest checksums and counts, checks train/test leakage, and confirms that every answerable QA resolves to its gold chunks in `dataset/corpus/retrieval.parquet`.

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
./scripts/run_pipeline.sh unanswerable   # build judgment-derived unanswerable QA
./scripts/run_pipeline.sh publish        # normalize, package, and validate
./scripts/run_pipeline.sh all            # run all construction stages
```

Raw crawl data remains local under `data/raw/` and is ignored by Git. Intermediate pipeline artifacts are written to the ignored `data/processed/` directory.

The package also exposes individual stages through `ura-xlaw --help`. Parallel court-index scanning is available through `scripts/scan_judgments_parallel.sh`.

## Dataset construction workflow

The standard pipeline reuses the two fixed corpora published in `dataset/corpus/`. This makes regeneration deterministic with respect to corpus coverage. The `all` mode starts from locally crawled judgments; it does not crawl the court portal automatically.

### 1. Acquire court judgments

This network stage is optional when `data/raw/congbobanan/` already exists.

```bash
# Scan 6,000 candidate IDs with four browser workers.
./scripts/scan_judgments_parallel.sh 6000 4

# Select documents from the scan index.
ura-xlaw select-judgments \
  --index data/raw/congbobanan/index.jsonl \
  --total 1500 \
  --out data/metadata/selected_ids.txt

# Download metadata/PDF text for the selected judgments.
ura-xlaw crawl-judgments \
  --ids-file data/metadata/selected_ids.txt \
  --batch-size 50
```

Output: JSON batches under `data/raw/congbobanan/`.

### 2. Clean judgments and generate QA

```bash
./scripts/run_pipeline.sh generate
```

This stage:

1. cleans PDF artifacts and deduplicates judgments into `data/processed/judgments_cleaned.jsonl`;
2. sends each eligible judgment to the configured LLM;
3. validates schema, source grounding, anonymized names, and question diversity;
4. writes accepted records to `data/processed/qa_generated_openai.jsonl` and rejected records to a separate JSONL file.

This stage requires `OPENAI_API_KEY` and incurs API cost. It resumes from existing generated records rather than regenerating completed document IDs.

### 3. Map citations and build answerable splits

```bash
./scripts/run_pipeline.sh build
```

The builder maps every `law_applied` citation to `dataset/corpus/full.parquet`, drops QA with unresolved citations, exports flat answerable QA, and creates a leakage-safe train/test split grouped by judgment and duplicate-question group.

Main outputs:

- `data/processed/qa_mapped.jsonl`
- `data/processed/qa_answerable.jsonl`
- `data/processed/train.jsonl`
- `data/processed/test.jsonl`

### 4. Build judgment-derived unanswerable QA

```bash
./scripts/run_pipeline.sh unanswerable
```

Candidate judgments are generated and mapped twice: once against the fixed retrieval corpus and once against the full corpus. A QA is unanswerable when at least one required citation is missing from the retrieval corpus. Answerable candidates are merged into the training split; unanswerable candidates are written to `data/processed/unanswerable.jsonl`.

This stage also uses the LLM API.

### 5. Normalize, package, and validate

```bash
./scripts/run_pipeline.sh publish
```

Publishing performs three operations:

1. normalizes split article chunks so every answerable gold ID resolves in the fixed retrieval corpus;
2. copies the five release artifacts into `dataset/` and regenerates `dataset/manifest.json`;
3. verifies checksums, schemas, row/chunk counts, train/test leakage, corpus consistency, answerable grounding, and the unanswerable contract.

After raw judgments are available, stages 2–5 can be run together:

```bash
./scripts/run_pipeline.sh all
```

Useful overrides:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | required | API key for QA generation |
| `URA_XLAW_MODEL` | `gpt-4.1` | generation model |
| `URA_XLAW_CONCURRENCY` | `8` | parallel LLM requests |
| `URA_XLAW_UNANSWERABLE_DOCS` | `100` | candidate judgments for the unanswerable stage |
| `PYTHON` | `python` | Python interpreter used by shell entry points |

### Optional: rebuild legal corpora

The published corpora are fixed benchmark artifacts and are not rebuilt by `run_pipeline.sh`. Corpus maintainers can use the lower-level commands below when updating a future release:

```bash
ura-xlaw build-corpus
ura-xlaw crawl-missing-laws --help
ura-xlaw build-supplemental-corpus --help
ura-xlaw build-precedent-corpus --help
```

## Task definition

Each answerable QA is grounded to one or more legal chunks through `positive_chunk_ids` and `positives`. The fixed RAG corpus is used for retrieval experiments.

A judgment-derived QA is labeled unanswerable when at least one required legal basis is unavailable from the fixed retrieval corpus. This supports evaluation of retrieval failure detection and abstention.

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

- Train on `dataset/train.jsonl`.
- Evaluate answerable performance on `dataset/test.jsonl`.
- Combine `dataset/test.jsonl` with `dataset/unanswerable.jsonl` for a 1,088-question answerability/abstention benchmark.
- Index `dataset/corpus/retrieval.parquet` for the constrained RAG setting.
- Use `dataset/corpus/full.parquet` for corpus-coverage analysis or full-corpus experiments.

## Data provenance

QA contexts are derived from real Vietnamese court judgments. Legal chunks are derived from Vietnamese legal-document corpora and supplemental legal sources. Users are responsible for complying with the terms and rights associated with the original sources.

## Repository layout

```text
dataset/              Public URA-xLaw release
src/ura_xlaw/prompts/ QA generation prompt packaged with the CLI
src/ura_xlaw/         Pipeline implementation
scripts/              Staged and parallel pipeline entry points
tests/                Dataset and pipeline smoke tests
data/                 Local-only raw and intermediate workspace
```
