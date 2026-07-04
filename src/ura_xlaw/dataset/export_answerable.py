"""
Export final retrieval dataset (STRICT mode).

Drops any Q&A entry whose citations did NOT all map successfully to corpus chunks.
Outputs:
  - data/processed/qa_answerable.jsonl   (1 line per Q&A; flat schema)
  - data/processed/law_corpus_final.parquet      (deduped union: primary + supplemental)
  - data/processed/qa_answerable.parquet (same as jsonl, columnar)
  - data/processed/qa_answerable_stats.json    (summary)
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter
import pandas as pd

from ura_xlaw.config import PATHS

IN_JSONL = PATHS.processed / "qa_mapped.jsonl"
OUT_JSONL = PATHS.processed / "qa_answerable.jsonl"
OUT_PARQUET = PATHS.processed / "qa_answerable.parquet"
OUT_CORPUS = PATHS.processed / "law_corpus_final.parquet"
OUT_STATS = PATHS.processed / "qa_answerable_stats.json"

PRIMARY = PATHS.processed / "law_corpus.parquet"
SUPP = PATHS.processed / "law_corpus_supplemental.parquet"


def main():
    parser = argparse.ArgumentParser(
        description="Export fully grounded answerable QA and the merged law corpus."
    )
    parser.add_argument("--input", default=str(IN_JSONL))
    parser.add_argument("--output-jsonl", default=str(OUT_JSONL))
    parser.add_argument("--output-parquet", default=str(OUT_PARQUET))
    parser.add_argument("--output-corpus", default=str(OUT_CORPUS))
    parser.add_argument("--output-stats", default=str(OUT_STATS))
    parser.add_argument("--primary-corpus", default=str(PRIMARY))
    parser.add_argument("--supplemental-corpus", default=str(SUPP))
    args = parser.parse_args()

    input_jsonl = Path(args.input)
    output_jsonl = Path(args.output_jsonl)
    output_parquet = Path(args.output_parquet)
    output_corpus = Path(args.output_corpus)
    output_stats = Path(args.output_stats)
    primary = Path(args.primary_corpus)
    supplemental = Path(args.supplemental_corpus)

    # --- Pass 1: filter Q&A entries (STRICT) ---
    n_docs = n_entries = n_kept = n_drop = 0
    kept_records = []
    used_chunk_ids: set[str] = set()
    drop_reasons = Counter()

    with input_jsonl.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n_docs += 1
            for e in rec.get("entries", []):
                n_entries += 1
                chunks = e.get("law_chunks", [])
                if not chunks or not all(c.get("matched") for c in chunks):
                    n_drop += 1
                    for c in chunks:
                        if not c.get("matched"):
                            drop_reasons[c.get("fail_reason", "unknown")] += 1
                    continue

                qa_id = f"{rec.get('doc_id')}_{n_kept}"
                positives = []
                for c in chunks:
                    cid = c["chunk_id"]
                    used_chunk_ids.add(cid)
                    positives.append(
                        {
                            "chunk_id": cid,
                            "citation": c["citation"],
                            "law_sig": c["law_sig"],
                            "article": c.get("article"),
                            "clause": c.get("clause"),
                            "point": c.get("point"),
                        }
                    )

                kept_records.append(
                    {
                        "qa_id": qa_id,
                        "doc_id": rec.get("doc_id"),
                        "case_number": rec.get("case_number"),
                        "court": rec.get("court"),
                        "case_type": rec.get("case_type"),
                        "trial_level": rec.get("trial_level"),
                        "legal_relation": rec.get("legal_relation"),
                        "situation": rec.get("situation"),
                        "legal_category": e.get("legal_category"),
                        "complexity_level": e.get("complexity_level"),
                        "question": e.get("question"),
                        "answer": e.get("answer"),
                        "legal_reasoning": e.get("legal_reasoning"),
                        "law_applied": e.get("law_applied", []),
                        "positive_chunk_ids": [p["chunk_id"] for p in positives],
                        "positives": positives,
                    }
                )
                n_kept += 1

    print(f"[export] docs={n_docs}  entries={n_entries}")
    print(f"[export] kept (strict): {n_kept}  ({n_kept / n_entries * 100:.2f}%)")
    print(f"[export] dropped:       {n_drop}  ({n_drop / n_entries * 100:.2f}%)")
    print(f"[export] drop reasons: {dict(drop_reasons)}")
    print(f"[export] unique positive chunks used: {len(used_chunk_ids)}")

    # --- Write Q&A jsonl + parquet ---
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for r in kept_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[export] wrote {output_jsonl}")

    # Parquet (flatten positives -> json string for safe storage)
    df = pd.DataFrame(kept_records)
    df["positives_json"] = df["positives"].apply(
        lambda xs: json.dumps(xs, ensure_ascii=False)
    )
    df_parq = df.drop(columns=["positives"]).copy()
    df_parq.to_parquet(output_parquet, index=False)
    print(f"[export] wrote {output_parquet}")

    # --- Build merged corpus (primary + supplemental, dedup by chunk_id) ---
    prim = pd.read_parquet(primary) if primary.exists() else pd.DataFrame()
    supp = pd.read_parquet(supplemental) if supplemental.exists() else pd.DataFrame()
    cols = sorted(set(prim.columns) | set(supp.columns))
    if not prim.empty:
        prim = prim.reindex(columns=cols)
    if not supp.empty:
        supp = supp.reindex(columns=cols)
    corpus = pd.concat([prim, supp], ignore_index=True)
    corpus = corpus.drop_duplicates(subset=["chunk_id"], keep="first")
    print(
        f"[export] merged corpus: {len(corpus)} chunks (primary={len(prim)}, supp={len(supp)})"
    )

    # Coverage check: every positive chunk_id must exist in corpus
    corpus_ids = set(corpus["chunk_id"].astype(str))
    missing = used_chunk_ids - corpus_ids
    print(f"[export] positives missing from corpus: {len(missing)}")
    if missing:
        print(f"  examples: {list(missing)[:5]}")

    corpus.to_parquet(output_corpus, index=False)
    print(f"[export] wrote {output_corpus}")

    # --- Stats ---
    n_pos_per_qa = [len(r["positive_chunk_ids"]) for r in kept_records]
    sigs_used = Counter()
    for r in kept_records:
        for p in r["positives"]:
            sigs_used[p["law_sig"]] += 1

    stats = {
        "n_docs_input": n_docs,
        "n_entries_input": n_entries,
        "n_qa_kept": n_kept,
        "n_qa_dropped": n_drop,
        "keep_rate_pct": round(n_kept / n_entries * 100, 2),
        "drop_reasons": dict(drop_reasons),
        "n_unique_positive_chunks": len(used_chunk_ids),
        "n_corpus_chunks": int(len(corpus)),
        "positives_missing_from_corpus": len(missing),
        "positives_per_qa": {
            "min": int(min(n_pos_per_qa)),
            "max": int(max(n_pos_per_qa)),
            "mean": round(sum(n_pos_per_qa) / len(n_pos_per_qa), 2),
        },
        "top_20_law_sigs_in_positives": sigs_used.most_common(20),
    }
    output_stats.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[export] wrote {output_stats}")
    print(
        f"\n[export] DONE. Final dataset: {n_kept} Q&A over {len(corpus)} corpus chunks."
    )


if __name__ == "__main__":
    main()
