"""Flatten nested QA JSONL → per-Q&A flat JSONL + CSV.

Reads the canonical generator output (1 record per doc with `entries[]`)
and emits 3× rows (one per complexity level) suitable for HuggingFace SFT
loading and human review in CSV.

Usage:
    python -m ura_xlaw flatten \
        --input data/processed/qa_generated_openai.jsonl \
        --out-jsonl data/processed/legal_qa_flat.jsonl \
        --out-csv   data/processed/legal_qa_flat.csv
"""

import argparse
import csv
import json
import os
from pathlib import Path


FLAT_FIELDS = [
    "qa_id",
    "doc_id",
    "complexity_level",
    "legal_category",
    "question",
    "answer",
    "legal_reasoning",
    "law_applied",
    "situation",
    "case_type",
    "trial_level",
    "is_precedent",
    "court",
    "case_number",
    "date",
    "doc_type",
    "legal_relation",
    "precedent_applied",
    "original_source",
    "body_sha1",
    "body_chars",
]


def flatten(record: dict):
    """Yield one flat row per entry in record['entries']."""
    base = {
        k: record.get(k)
        for k in FLAT_FIELDS
        if k
        not in {
            "qa_id",
            "complexity_level",
            "legal_category",
            "question",
            "answer",
            "legal_reasoning",
            "law_applied",
        }
    }
    doc_id = record.get("doc_id", "")
    for i, entry in enumerate(record.get("entries", []) or []):
        level = entry.get("complexity_level", f"L{i}")
        row = dict(base)
        row["qa_id"] = f"{doc_id}_{level.lower()}"
        row["complexity_level"] = level
        row["legal_category"] = entry.get("legal_category", "")
        row["question"] = entry.get("question", "")
        row["answer"] = entry.get("answer", "")
        row["legal_reasoning"] = entry.get("legal_reasoning", "")
        row["law_applied"] = entry.get("law_applied", []) or []
        yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Nested JSONL from generator")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    Path(os.path.dirname(args.out_jsonl) or ".").mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.out_csv) or ".").mkdir(parents=True, exist_ok=True)

    n_doc = 0
    n_row = 0
    with open(args.input, encoding="utf-8") as f_in, open(
        args.out_jsonl, "w", encoding="utf-8"
    ) as f_jsonl, open(args.out_csv, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=FLAT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_doc += 1
            for row in flatten(rec):
                f_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
                # CSV: serialize list as " ; " joined string
                csv_row = dict(row)
                csv_row["law_applied"] = " ; ".join(row["law_applied"])
                writer.writerow(csv_row)
                n_row += 1

    print(f"docs:    {n_doc}")
    print(f"qa rows: {n_row}")
    print(f"-> {args.out_jsonl}")
    print(f"-> {args.out_csv}")


if __name__ == "__main__":
    main()
