"""Package processed artifacts as the public URA-xLaw release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from ura_xlaw import __version__
from ura_xlaw.config import DATASET_NAME, PATHS


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(source: Path, target: Path, force: bool) -> None:
    if target.exists() and not force:
        raise FileExistsError(f"{target} exists; pass --force to replace it")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default=str(PATHS.processed))
    parser.add_argument("--output", default=str(PATHS.release))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    processed, output = Path(args.processed), Path(args.output)
    sources = {
        "train": processed / "train.jsonl",
        "test": processed / "test.jsonl",
        "unanswerable_real": processed / "unanswerable_real.jsonl",
        "corpus_full": processed / "law_corpus_final.parquet",
        "corpus_rag": processed / "law_corpus_qa_only.parquet",
    }
    targets = {
        "train": output / "train.jsonl",
        "test": output / "test.jsonl",
        "unanswerable_real": output / "unanswerable_real.jsonl",
        "corpus_full": output / "corpus_full.parquet",
        "corpus_rag": output / "corpus_rag.parquet",
    }
    for name, source in sources.items():
        if not source.exists():
            raise FileNotFoundError(f"Missing {name}: {source}")
        copy(source, targets[name], args.force)

    manifest = {
        "dataset_name": DATASET_NAME,
        "version": __version__,
        "language": "vi",
        "task": "legal retrieval-augmented question answering",
        "files": {},
    }
    for name in ("train", "test", "unanswerable_real"):
        path = targets[name]
        with path.open(encoding="utf-8") as source:
            rows = sum(1 for line in source if line.strip())
        manifest["files"][name] = {
            "path": path.name,
            "rows": rows,
            "sha256": checksum(path),
        }
    for name in ("corpus_full", "corpus_rag"):
        path = targets[name]
        frame = pd.read_parquet(path, columns=["chunk_id", "law_sig"])
        manifest["files"][name] = {
            "path": path.name,
            "chunks": len(frame),
            "legal_documents": frame["law_sig"].nunique(),
            "sha256": checksum(path),
        }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Packaged {DATASET_NAME} in {output}")


if __name__ == "__main__":
    main()
