"""Leakage-safe train/test splitting for URA-xLaw.

Judgments are the unit of splitting. Duplicate-question groups that cross
judgments are merged first, then groups are stratified by legal category.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ura_xlaw.config import PATHS


class UnionFind:
    """Merge document groups connected by a duplicate-question group."""

    def __init__(self, values: set[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def split_records(
    records: list[dict],
    *,
    test_fraction: float = 0.20,
    seed: int = 42,
    min_category_size: int = 30,
) -> tuple[list[dict], list[dict]]:
    """Return train/test records without document or duplicate-group leakage."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if not records:
        return [], []

    doc_ids = {str(record["doc_id"]) for record in records}
    groups = UnionFind(doc_ids)

    duplicate_docs: dict[str, set[str]] = defaultdict(set)
    for record in records:
        duplicate_group = record.get("question_dup_group_id")
        if duplicate_group:
            duplicate_docs[str(duplicate_group)].add(str(record["doc_id"]))
    for docs in duplicate_docs.values():
        docs = list(docs)
        for doc_id in docs[1:]:
            groups.union(docs[0], doc_id)

    records_by_group: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        records_by_group[groups.find(str(record["doc_id"]))].append(record)

    group_info: list[dict] = []
    category_sizes: Counter = Counter()
    for group_id, items in records_by_group.items():
        category = Counter(
            item.get("legal_category", "UNKNOWN") for item in items
        ).most_common(1)[0][0]
        info = {"id": group_id, "size": len(items), "category": category}
        group_info.append(info)
        category_sizes[category] += len(items)

    small_categories = {
        category
        for category, size in category_sizes.items()
        if size < min_category_size
    }
    by_category: dict[str, list[dict]] = defaultdict(list)
    for group in group_info:
        category = (
            "_OTHER" if group["category"] in small_categories else group["category"]
        )
        by_category[category].append(group)

    rng = np.random.RandomState(seed)
    train_groups: set[str] = set()
    test_groups: set[str] = set()
    for category_groups in by_category.values():
        rng.shuffle(category_groups)
        total = sum(group["size"] for group in category_groups)
        if total < 5:
            train_groups.update(group["id"] for group in category_groups)
            continue
        target = total * test_fraction
        current = 0
        for group in category_groups:
            if current + group["size"] / 2 < target:
                test_groups.add(group["id"])
                current += group["size"]
            else:
                train_groups.add(group["id"])

    train = [
        record
        for group_id, items in records_by_group.items()
        if group_id in train_groups
        for record in items
    ]
    test = [
        record
        for group_id, items in records_by_group.items()
        if group_id in test_groups
        for record in items
    ]
    return train, test


def write_split(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    pd.DataFrame(records).to_parquet(path.with_suffix(".parquet"), index=False)


def validate_split(train: list[dict], test: list[dict]) -> None:
    train_docs = {str(record["doc_id"]) for record in train}
    test_docs = {str(record["doc_id"]) for record in test}
    train_dupes = {
        record.get("question_dup_group_id")
        for record in train
        if record.get("question_dup_group_id")
    }
    test_dupes = {
        record.get("question_dup_group_id")
        for record in test
        if record.get("question_dup_group_id")
    }
    if overlap := train_docs & test_docs:
        raise RuntimeError(f"Document leakage detected: {len(overlap)} documents")
    if overlap := train_dupes & test_dupes:
        raise RuntimeError(f"Duplicate-group leakage detected: {len(overlap)} groups")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=str(PATHS.processed / "qa_answerable.jsonl")
    )
    parser.add_argument("--output-dir", default=str(PATHS.processed))
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with Path(args.input).open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    train, test = split_records(
        records, test_fraction=args.test_fraction, seed=args.seed
    )
    validate_split(train, test)

    output_dir = Path(args.output_dir)
    write_split(train, output_dir / "train.jsonl")
    write_split(test, output_dir / "test.jsonl")
    print(f"URA-xLaw split: train={len(train):,}, test={len(test):,}")
    print("Leakage checks: passed")


if __name__ == "__main__":
    main()
