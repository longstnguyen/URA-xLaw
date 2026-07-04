"""Repair split article chunks so every answerable QA resolves in the RAG corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ura_xlaw.config import PATHS


def clause_numbers(value) -> list[int]:
    try:
        return [int(item) for item in list(value)]
    except (TypeError, ValueError):
        return []


def merge_article(rows: pd.DataFrame) -> dict:
    records = sorted(
        (row for _, row in rows.iterrows()),
        key=lambda row: (
            min(clause_numbers(row["clause_nums"]))
            if clause_numbers(row["clause_nums"])
            else 10**9,
            str(row["chunk_id"]),
        ),
    )
    texts: list[str] = []
    clauses: dict = {}
    numbers: set[int] = set()
    for row in records:
        text = str(row.get("article_text") or "").strip()
        if text and text not in texts:
            texts.append(text)
        numbers.update(clause_numbers(row.get("clause_nums")))
        try:
            raw = row.get("clauses_json") or "{}"
            clauses.update(json.loads(raw) if isinstance(raw, str) else dict(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    base = records[0]
    article_text = "\n\n".join(texts)
    content = (
        f"[{str(base['law_sig']).upper()}]\n\n{base['law_short_name']}\n"
        f"Điều {base['article_num']}. {base['article_title']}\n\n{article_text}"
    )
    return {
        "article_text": article_text,
        "clause_nums": sorted(numbers),
        "clauses_json": json.dumps(clauses, ensure_ascii=False),
        "content": content,
    }


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(
    train_path: Path,
    test_path: Path,
    full_path: Path,
    rag_path: Path,
) -> int:
    train, test = load_jsonl(train_path), load_jsonl(test_path)
    full, rag = pd.read_parquet(full_path), pd.read_parquet(rag_path)
    full_ids = set(full["chunk_id"].astype(str))
    rag_ids = set(rag["chunk_id"].astype(str))
    gold_ids = {
        str(chunk_id)
        for row in train + test
        for chunk_id in row.get("positive_chunk_ids", [])
    }
    missing = sorted(gold_ids - rag_ids)
    if not missing:
        print("Grounding already normalized; nothing to change.")
        return 0
    if not set(missing) <= full_ids:
        raise ValueError("Some missing gold chunks are absent from the full corpus")

    remap: dict[str, str] = {}
    for missing_id in missing:
        source = full[full["chunk_id"].astype(str) == missing_id].iloc[0]
        same_article = full[
            (full["law_sig"] == source["law_sig"])
            & (full["article_num"].astype(str) == str(source["article_num"]))
        ]
        candidates = rag[
            (rag["law_sig"] == source["law_sig"])
            & (rag["article_num"].astype(str) == str(source["article_num"]))
        ]
        if candidates.empty:
            raise ValueError(f"No canonical RAG chunk for {missing_id}")
        canonical = sorted(candidates["chunk_id"].astype(str))[0]
        remap[missing_id] = canonical
        merged = merge_article(same_article)
        for frame in (full, rag):
            indices = frame.index[frame["chunk_id"].astype(str) == canonical]
            if len(indices) != 1:
                raise ValueError(f"Canonical chunk is not unique: {canonical}")
            for field, value in merged.items():
                frame.at[indices[0], field] = value

    changed = 0
    for row in train + test:
        before = [str(value) for value in row.get("positive_chunk_ids", [])]
        after = [remap.get(value, value) for value in before]
        if before == after:
            continue
        changed += 1
        row["positive_chunk_ids"] = after
        for positive in row.get("positives", []):
            chunk_id = str(positive.get("chunk_id", ""))
            positive["chunk_id"] = remap.get(chunk_id, chunk_id)

    normalized_gold = {
        str(chunk_id)
        for row in train + test
        for chunk_id in row.get("positive_chunk_ids", [])
    }
    if not normalized_gold <= set(rag["chunk_id"].astype(str)):
        raise ValueError("Grounding normalization did not resolve all gold chunks")

    write_jsonl(train_path, train)
    write_jsonl(test_path, test)
    full.to_parquet(full_path, index=False)
    rag.to_parquet(rag_path, index=False)
    print(f"Normalized {len(remap)} split chunks across {changed} QA")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(PATHS.processed / "train.jsonl"))
    parser.add_argument("--test", default=str(PATHS.processed / "test.jsonl"))
    parser.add_argument(
        "--full-corpus", default=str(PATHS.processed / "law_corpus_final.parquet")
    )
    parser.add_argument(
        "--rag-corpus", default=str(PATHS.processed / "law_corpus_qa_only.parquet")
    )
    args = parser.parse_args()
    normalize(
        Path(args.train),
        Path(args.test),
        Path(args.full_corpus),
        Path(args.rag_corpus),
    )


if __name__ == "__main__":
    main()
