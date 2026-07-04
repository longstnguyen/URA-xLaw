"""
Build unanswerable QA from real court judgments (re-run rejected docs).

Steps:
  1. Pick 100 unused source docs (previously rejected by validator)
  2. Re-run them through the URA-xLaw generation service
  3. Filter resulting QA: those whose `law_applied` cite laws NOT in the small
     qa_only corpus → mark unanswerable
  4. Output → data/processed/unanswerable_real.jsonl

Usage:
  python -m ura_xlaw build-real-unanswerable --pick 100 --model gpt-4.1
  python -m ura_xlaw build-real-unanswerable --classify-only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from ura_xlaw.config import PATHS

load_dotenv()

QA_CORPUS = str(PATHS.processed / "law_corpus_qa_only.parquet")
CLASSIFIER_QA_CORPUS = str(
    PATHS.data / "corpus_rag.parquet"
)
FULL_CORPUS = str(PATHS.data / "corpus_full.parquet")
SRC = str(PATHS.raw_judgments / "dataset_2000.jsonl")
USED_QA = str(PATHS.processed / "qa_answerable.jsonl")
RAW_BATCH_GLOB = str(PATHS.raw_judgments / "congbobanan_*_batch_*.json")

# Case types most likely to cite documents OUTSIDE the small qa_only corpus
# (Quyết định UBND, Nghị định, Thông tư, quy chế nội bộ, Luật chuyên ngành)
TARGET_CASE_TYPES = {"Hành chính", "Lao động", "Kinh doanh thương mại"}

UNANS_INPUT = str(PATHS.processed / "unanswerable_input.jsonl")
UNANS_OUT_DIR = str(PATHS.processed / "unanswerable_gen")
UNANS_RAW = f"{UNANS_OUT_DIR}/qa_generated_openai.jsonl"
UNANS_CHUNKS_SMALL = f"{UNANS_OUT_DIR}/with_chunks_qa_only.jsonl" # mapped vs qa_only (classifier)
UNANS_CHUNKS_FULL  = f"{UNANS_OUT_DIR}/with_chunks_full.jsonl"    # mapped vs full (for positives)
UNANS_FINAL = str(PATHS.processed / "unanswerable_real.jsonl")
EXTRA_TRAIN = str(PATHS.processed / "extra_train_from_judgments.jsonl")

SEED = 42


def step1_pick_docs(n: int) -> int:
    """Pick from already-crawled batch files; exclude the 2000 used + filter by case_type."""
    import glob

    used_in_2k = {str(json.loads(l)["id"]) for l in open(SRC)}

    all_docs: list[dict] = []
    for f in sorted(glob.glob(RAW_BATCH_GLOB)):
        try:
            data = json.load(open(f))
            if isinstance(data, list):
                all_docs.extend(data)
        except Exception as e:
            print(f"  skip {f}: {e}")

    # Normalize id, dedupe
    seen = set()
    unused: list[dict] = []
    for d in all_docs:
        did = str(d.get("doc_id") or d.get("id") or "")
        if not did or did in used_in_2k or did in seen:
            continue
        seen.add(did)
        # Coerce schema to the cleaned judgment-pool shape.
        d["id"] = did
        unused.append(d)
    print(f"Crawled docs not in 2000-set: {len(unused)}")

    target = [
        d for d in unused
        if d.get("case_type") in TARGET_CASE_TYPES
        and 1500 <= len(d.get("body_cleaned") or d.get("body") or "") <= 60000
    ]
    print(f"Target (case_type∈{TARGET_CASE_TYPES} & 1500≤body≤60000): {len(target)}")

    rng = __import__("random").Random(SEED)
    if n >= len(target):
        picked = target
    else:
        picked = rng.sample(target, n)
    # Sort by id for stable order
    picked = sorted(picked, key=lambda d: d["id"])

    # Stats
    ct = __import__("collections").Counter(d.get("case_type") for d in picked)
    print(f"Picked {len(picked)} docs by case_type: {dict(ct)}")

    Path(os.path.dirname(UNANS_INPUT)).mkdir(parents=True, exist_ok=True)
    with open(UNANS_INPUT, "w", encoding="utf-8") as f:
        for d in picked:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Wrote {len(picked)} docs → {UNANS_INPUT}")
    return len(picked)


def step2_run_generator(model: str, concurrency: int):
    """Invoke generator.DatasetGenerator with a custom output dir."""
    from ura_xlaw.generation import DatasetGenerator
    import asyncio

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set")

    Path(UNANS_OUT_DIR).mkdir(parents=True, exist_ok=True)
    gen = DatasetGenerator(output_dir=UNANS_OUT_DIR)
    if concurrency > 1:
        asyncio.run(
            gen.process_dataset_async(
                input_file=UNANS_INPUT,
                api_key=api_key,
                model=model,
                concurrency=concurrency,
            )
        )
    else:
        gen.process_dataset(
            input_file=UNANS_INPUT,
            provider="openai",
            api_key=api_key,
            model=model,
        )


def step2b_map_citations():
    """Map twice:
    (a) vs OLD SMALL qa_only corpus → used to decide answerable/unanswerable
    (b) vs FULL corpus           → used to build positives + later expand qa_only
    """
    import subprocess
    classifier_corpus = CLASSIFIER_QA_CORPUS if Path(CLASSIFIER_QA_CORPUS).exists() else QA_CORPUS
    for tag, corpus, out in (
        ("qa_only", classifier_corpus, UNANS_CHUNKS_SMALL),
        ("full",    FULL_CORPUS, UNANS_CHUNKS_FULL),
    ):
        cmd = [
            sys.executable, "-m", "ura_xlaw.corpus.citations",
            "--input", UNANS_RAW,
            "--output", out,
            "--corpus", corpus,
            "--supp", str(PATHS.processed / "__nonexistent_supp__.parquet"),
            "--report", f"{UNANS_OUT_DIR}/coverage_{tag}.json",
        ]
        print(f"\n[map vs {tag}]  $ {' '.join(cmd)}\n")
        subprocess.run(cmd, check=True)


def step3_classify_and_save():
    """Classify on old qa_only matching; build positives from FULL-corpus mapping.

    Unanswerable-priority rule: if ANY citation is missing from the old small
    corpus, keep the QA in unanswerable. Only all-hit QA are added to train.
    """
    if not Path(UNANS_CHUNKS_SMALL).exists() or not Path(UNANS_CHUNKS_FULL).exists():
        step2b_map_citations()

    small = [json.loads(l) for l in open(UNANS_CHUNKS_SMALL)]
    full  = [json.loads(l) for l in open(UNANS_CHUNKS_FULL)]
    assert len(small) == len(full), "mapping output size mismatch"
    print(f"\nLoaded {len(small)} doc-level records (qa_only & full)")

    extra_train: list[dict] = []
    unans: list[dict] = []
    n_skip = 0
    n_kept_total = 0

    for rec_s, rec_f in zip(small, full):
        sit = (rec_s.get("situation") or "").strip()
        doc_id = str(rec_s.get("doc_id") or "")
        entries_s = rec_s.get("entries") or []
        entries_f = rec_f.get("entries") or []
        for i, (e_s, e_f) in enumerate(zip(entries_s, entries_f)):
            citations = e_s.get("law_applied") or []
            chunks_s  = e_s.get("law_chunks") or []
            chunks_f  = e_f.get("law_chunks") or []
            if not citations or not chunks_s:
                n_skip += 1
                continue

            all_in_qa_only = all(c.get("matched") for c in chunks_s)
            qa_id = f"j_{doc_id}_{i}"

            base = {
                "qa_id": qa_id,
                "doc_id": doc_id,
                "case_number": rec_s.get("case_number", ""),
                "court": rec_s.get("court", ""),
                "case_type": rec_s.get("case_type", ""),
                "trial_level": rec_s.get("trial_level", ""),
                "legal_relation": rec_s.get("legal_relation", ""),
                "situation": sit,
                "legal_category": e_s.get("legal_category", ""),
                "complexity_level": e_s.get("complexity_level", f"L{i}"),
                "question": (e_s.get("question") or "").strip(),
                "answer": (e_s.get("answer") or "").strip(),
                "legal_reasoning": (e_s.get("legal_reasoning") or "").strip(),
                "law_applied": citations,
                "question_dup_group_id": None,
            }

            if not all_in_qa_only:
                base["positive_chunk_ids"] = []
                base["positives"] = []
                base["is_unanswerable"] = True
                unans.append(base)
            else:
                # Build positives from FULL-corpus mapping (richer)
                matched_chunks = [c for c in chunks_f if c.get("matched")]
                positives = [{
                    "chunk_id": c["chunk_id"],
                    "citation": c["citation"],
                    "law_sig": c.get("law_sig"),
                    "article": c.get("article"),
                    "clause": c.get("clause"),
                    "point": c.get("point"),
                } for c in matched_chunks]
                base["positive_chunk_ids"] = [p["chunk_id"] for p in positives]
                base["positives"] = positives
                extra_train.append(base)
            n_kept_total += 1

    n_full = sum(1 for r in extra_train if len(r["positives"]) == len(r["law_applied"]))
    n_partial = len(extra_train) - n_full
    print(f"\nQA total: {n_kept_total}  (skipped {n_skip}: empty law_applied/chunks)")
    print(f"  → Answerable (ALL cites hit old qa_only):  {len(extra_train)}")
    print(f"      • full positives from full corpus  ({n_full})")
    print(f"      • partial positives                ({n_partial})")
    print(f"  → Unanswerable (ANY cite missing old qa_only): {len(unans)}")

    Path(os.path.dirname(UNANS_FINAL) or ".").mkdir(parents=True, exist_ok=True)
    with open(UNANS_FINAL, "w", encoding="utf-8") as f:
        for r in unans:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pd.DataFrame(unans).to_parquet(UNANS_FINAL.replace(".jsonl", ".parquet"), index=False)
    print(f"\nSaved {len(unans)} unanswerable QA → {UNANS_FINAL}")

    with open(EXTRA_TRAIN, "w", encoding="utf-8") as f:
        for r in extra_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    df_extra = pd.DataFrame(extra_train)
    if len(df_extra):
        df_extra["positives_json"] = df_extra["positives"].apply(lambda xs: json.dumps(xs, ensure_ascii=False))
        df_extra.drop(columns=["positives"]).to_parquet(EXTRA_TRAIN.replace(".jsonl", ".parquet"), index=False)
    print(f"Saved {len(extra_train)} answerable QA → {EXTRA_TRAIN}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", type=int, default=100, help="Number of docs to pick & generate")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--skip-pick", action="store_true", help="Skip step 1")
    ap.add_argument("--skip-gen", action="store_true", help="Skip step 2 (generator)")
    ap.add_argument("--classify-only", action="store_true", help="Run only step 3")
    ap.add_argument("--merge-train", action="store_true", help="After classify, merge answerable into train.jsonl")
    ap.add_argument(
        "--expand-corpus",
        action="store_true",
        help="Also add newly referenced chunks to the working RAG corpus.",
    )
    args = ap.parse_args()

    if args.classify_only:
        step3_classify_and_save()
        return

    if not args.skip_pick:
        step1_pick_docs(args.pick)
    if not args.skip_gen:
        step2_run_generator(args.model, args.concurrency)
    step2b_map_citations()
    step3_classify_and_save()
    if args.merge_train:
        step4_merge_into_train()
    if args.expand_corpus:
        step5_expand_qa_corpus()


def step5_expand_qa_corpus():
    """Add chunks newly referenced by extra_train QA into law_corpus_qa_only.parquet."""
    if not Path(EXTRA_TRAIN).exists():
        print("No extra_train; skipping corpus expansion.")
        return
    extra = [json.loads(l) for l in open(EXTRA_TRAIN)]
    new_chunk_ids = {p["chunk_id"] for r in extra for p in r["positives"]}
    if not new_chunk_ids:
        print("No new chunks to add.")
        return

    qa = pd.read_parquet(QA_CORPUS)
    full = pd.read_parquet(FULL_CORPUS)
    existing_ids = set(qa["chunk_id"])
    to_add_ids = new_chunk_ids - existing_ids
    print(f"\nCorpus expansion:")
    print(f"  qa_only existing chunks:     {len(existing_ids)}")
    print(f"  new chunk ids referenced:    {len(new_chunk_ids)}")
    print(f"  truly new (not in qa_only):  {len(to_add_ids)}")
    if not to_add_ids:
        print("  Nothing to add.")
        return

    add_rows = full[full["chunk_id"].isin(to_add_ids)]
    print(f"  rows fetched from full:      {len(add_rows)}")
    missing = to_add_ids - set(add_rows["chunk_id"])
    if missing:
        print(f"  WARN: {len(missing)} chunk_ids not found in full corpus (skipped)")

    expanded = pd.concat([qa, add_rows], ignore_index=True)
    expanded = expanded.drop_duplicates(subset=["chunk_id"]).reset_index(drop=True)
    # Backup
    bak = QA_CORPUS + ".bak"
    if not Path(bak).exists():
        import shutil
        shutil.copy(QA_CORPUS, bak)
        print(f"  Backed up original → {bak}")
    expanded.to_parquet(QA_CORPUS, index=False)
    print(f"  Wrote {len(expanded)} chunks → {QA_CORPUS}  (was {len(qa)})")
    print(f"  unique laws now: {expanded['law_sig'].nunique()} (was {qa['law_sig'].nunique()})")


def step4_merge_into_train():
    """Merge answerable QA into train.jsonl + train.parquet.

    Existing generated judgment QA are replaced by qa_id so reruns can refresh
    positives after remapping against a larger corpus.
    """
    train_jsonl = str(PATHS.processed / "train.jsonl")
    train_parq = str(PATHS.processed / "train.parquet")
    if not Path(EXTRA_TRAIN).exists():
        print(f"No extra train file at {EXTRA_TRAIN}; skipping merge.")
        return

    extra = [json.loads(l) for l in open(EXTRA_TRAIN)]
    if not extra:
        print("No answerable QA to merge.")
        return

    existing = [json.loads(l) for l in open(train_jsonl)]
    extra_by_id = {r["qa_id"]: r for r in extra}
    replaced = sum(1 for r in existing if r["qa_id"] in extra_by_id)
    base = [r for r in existing if r["qa_id"] not in extra_by_id]
    new = [r for r in extra if r["qa_id"] not in {x["qa_id"] for x in existing}]
    print(
        f"\nMerge: existing train={len(existing)}, extra={len(extra)}, "
        f"replaced={replaced}, new={len(new)}"
    )

    merged = base + extra
    with open(train_jsonl, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    df = pd.DataFrame(merged)
    df["positives_json"] = df["positives"].apply(lambda xs: json.dumps(xs, ensure_ascii=False))
    df.drop(columns=["positives"]).to_parquet(train_parq, index=False)
    print(f"Wrote {len(merged)} QA \u2192 {train_jsonl} (+ parquet)")


if __name__ == "__main__":
    main()
