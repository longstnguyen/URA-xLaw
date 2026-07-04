"""
Add Án lệ documents to the supplemental corpus from `phamhoangf/legal_benchmark_anle`.

Each án lệ becomes a single chunk with chunk_id `supp_alXX_YYYY_al` and law_sig
matching the citation grammar `AL<n>/<year>/AL`.

Output: appends to data/processed/law_corpus_supplemental.parquet
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import pandas as pd
from datasets import load_dataset

from ura_xlaw.config import PATHS

OUT = PATHS.processed / "law_corpus_supplemental.parquet"
INPUT_JSONL = PATHS.processed / "qa_mapped.jsonl"

RE_AL = re.compile(r"^\s*(\d+)\s*/\s*(\d{4})\s*/\s*AL\s*$", re.I)


def collect_needed_al() -> set:
    """Find all 'AL N/YYYY/AL' citations that are unmapped."""
    out = set()
    pat = re.compile(r"AL\s*(\d+)\s*/\s*(\d{4})\s*/\s*AL", re.I)
    for line in INPUT_JSONL.open():
        rec = json.loads(line)
        for e in rec.get("entries", []):
            for ch in e.get("law_chunks", []):
                if ch.get("matched"):
                    continue
                m = pat.search(ch.get("citation", ""))
                if m:
                    out.add(f"{int(m.group(1)):02d}/{m.group(2)}/AL")
    return out


def main():
    needed = collect_needed_al()
    print(f"[al] {len(needed)} unmapped án lệ ids: {sorted(needed)[:10]}...")

    print("[al] Loading phamhoangf/legal_benchmark_anle ...")
    ds = load_dataset("phamhoangf/legal_benchmark_anle", split="anle")
    df = ds.to_pandas()
    print(f"[al]   {len(df)} QA rows; unique ids: {df['id'].nunique()}")

    # normalize id like "20/2018/AL" -> "20/2018/AL"; keep first context per id
    def norm_al(s):
        m = re.match(r"\s*(\d+)\s*/\s*(\d{4})\s*/\s*AL\s*", str(s), re.I)
        return f"{int(m.group(1)):02d}/{m.group(2)}/AL" if m else None

    df["nid"] = df["id"].map(norm_al)
    uniq = df.dropna(subset=["nid"]).drop_duplicates(subset=["nid"])
    print(f"[al]   {len(uniq)} unique án lệ in dataset")

    # Build rows
    rows = []
    matched = 0
    for _, r in uniq.iterrows():
        nid = r["nid"]
        # Build matching law_sig: e.g. "20/2018/AL" -> "20_2018_al"
        num, year, _ = nid.split("/")
        # also remove leading zero for sig (mapper resolves both)
        n_int = int(num)
        law_sig = f"{n_int}_{year}_al"
        is_needed = nid in needed
        if is_needed:
            matched += 1
        ctx = r["context"] or ""
        # Strip excessive leading whitespace lines
        ctx_clean = re.sub(r"\n{3,}", "\n\n", ctx).strip()
        title = f"Án lệ số {n_int:02d}/{year}/AL"
        rows.append({
            "chunk_id": f"supp_al_{n_int}_{year}",
            "law_sig": law_sig,
            "law_num": str(n_int),
            "law_year": year,
            "law_category": "al",
            "law_short_name": title,
            "article_num": "1",
            "article_title": title,
            "clause_nums": [],
            "content": f"[{title}]\n\n{ctx_clean}",
            "article_text": ctx_clean,
            "clauses_json": "{}",
        })
    print(f"[al] Produced {len(rows)} án lệ rows ({matched} cover unmapped citations)")

    new_df = pd.DataFrame(rows)
    if OUT.exists():
        existing = pd.read_parquet(OUT)
        # Drop any pre-existing al rows to avoid dupes
        existing = existing[existing["law_category"] != "al"]
        out_df = pd.concat([existing, new_df], ignore_index=True)
    else:
        out_df = new_df
    out_df.to_parquet(OUT, index=False)
    print(f"[al] Wrote {len(out_df)} total rows -> {OUT}")


if __name__ == "__main__":
    main()
