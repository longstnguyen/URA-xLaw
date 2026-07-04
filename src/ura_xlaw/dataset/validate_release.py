"""Validate a published URA-xLaw release end to end."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

from ura_xlaw.config import PATHS


QA_FIELDS = {
    "qa_id",
    "doc_id",
    "question",
    "answer",
    "legal_reasoning",
    "law_applied",
    "positive_chunk_ids",
    "positives",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def validate_release(data_dir: Path, sample_size: int = 3) -> None:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    qa_sets: dict[str, list[dict]] = {}

    for name in ("train", "test", "unanswerable"):
        metadata = manifest["files"][name]
        path = data_dir / metadata["path"]
        rows = load_jsonl(path)
        if len(rows) != metadata["rows"]:
            raise ValueError(
                f"{name}: expected {metadata['rows']} rows, got {len(rows)}"
            )
        if sha256(path) != metadata["sha256"]:
            raise ValueError(f"{name}: checksum mismatch")
        qa_sets[name] = rows
        print(f"PASS {name}: {len(rows):,} valid JSON records")

    corpora: dict[str, pd.DataFrame] = {}
    for name in ("corpus_full", "corpus_rag"):
        metadata = manifest["files"][name]
        path = data_dir / metadata["path"]
        corpus = pd.read_parquet(path)
        if len(corpus) != metadata["chunks"]:
            raise ValueError(f"{name}: chunk count mismatch")
        if corpus["law_sig"].nunique() != metadata["legal_documents"]:
            raise ValueError(f"{name}: legal-document count mismatch")
        if corpus["chunk_id"].astype(str).nunique() != len(corpus):
            raise ValueError(f"{name}: duplicate chunk_id")
        if sha256(path) != metadata["sha256"]:
            raise ValueError(f"{name}: checksum mismatch")
        corpus = corpus.copy()
        corpus["chunk_id"] = corpus["chunk_id"].astype(str)
        corpora[name] = corpus
        print(
            f"PASS {name}: {len(corpus):,} chunks, "
            f"{corpus['law_sig'].nunique():,} legal documents"
        )

    for name in ("train", "test"):
        for index, row in enumerate(qa_sets[name]):
            missing = QA_FIELDS - set(row)
            if missing:
                raise ValueError(f"{name}[{index}]: missing fields {sorted(missing)}")

    train_ids = {row["qa_id"] for row in qa_sets["train"]}
    test_ids = {row["qa_id"] for row in qa_sets["test"]}
    if len(train_ids) != len(qa_sets["train"]) or len(test_ids) != len(qa_sets["test"]):
        raise ValueError("Duplicate QA IDs within a split")
    if train_ids & test_ids:
        raise ValueError("Train/test QA leakage detected")
    print("PASS QA IDs: unique; train/test overlap = 0")

    rag = corpora["corpus_rag"].set_index("chunk_id", drop=False)
    full = corpora["corpus_full"].set_index("chunk_id", drop=False)
    if not set(rag.index) <= set(full.index):
        raise ValueError("RAG corpus contains chunks absent from full corpus")
    for chunk_id in rag.index:
        if str(rag.at[chunk_id, "content"]) != str(full.at[chunk_id, "content"]):
            raise ValueError(f"Corpus content mismatch for {chunk_id}")

    answerable = qa_sets["train"] + qa_sets["test"]
    missing_gold = []
    for row in answerable:
        gold = {str(value) for value in row["positive_chunk_ids"]}
        if not gold or not gold <= set(rag.index):
            missing_gold.append(row["qa_id"])
    if missing_gold:
        raise ValueError(f"Invalid gold chunks for {len(missing_gold)} QA")
    print(
        f"PASS grounding: all {len(answerable):,} answerable QA resolve in corpus_rag"
    )

    unanswerable = qa_sets["unanswerable"]
    if not all(
        row.get("is_unanswerable") is True
        and not row.get("positive_chunk_ids")
        and not row.get("positives")
        for row in unanswerable
    ):
        raise ValueError("Unanswerable contract violated")
    print(f"PASS unanswerable contract: {len(unanswerable):,} QA")

    rng = random.Random(42)
    for split in ("train", "test"):
        print(f"SAMPLES {split}:")
        for row in rng.sample(qa_sets[split], min(sample_size, len(qa_sets[split]))):
            found = sum(str(value) in rag.index for value in row["positive_chunk_ids"])
            print(
                f"- {row['qa_id']}: gold_lookup={found}/{len(row['positive_chunk_ids'])}"
            )

    print("ALL URA-xLaw RELEASE TESTS PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(PATHS.release))
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    validate_release(Path(args.data_dir), sample_size=args.samples)


if __name__ == "__main__":
    main()
