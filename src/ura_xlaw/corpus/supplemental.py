"""
Build SUPPLEMENTAL corpus from `hirine/dataset-van-ban-phap-luat-381K-samples`
covering documents missing from the primary `truro7/vn-law-corpus`.

For each missing doc_id (collected from coverage report), find its full text in
the hirine dataset, parse Điều/Khoản, and emit rows in the same schema as
data/processed/law_corpus.parquet so they can be merged.

Output: data/processed/law_corpus_supplemental.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from ura_xlaw.config import PATHS


HIRINE_PARQUET = Path(
    os.environ.get(
        "URA_XLAW_HIRINE_PARQUET",
        str(PATHS.raw / "hirine" / "data.parquet"),
    )
)
INPUT_JSONL = PATHS.processed / "qa_generated_openai.jsonl"
OUT_PARQUET = PATHS.processed / "law_corpus_supplemental.parquet"

RE_DOC_ID = re.compile(r"(\d+)\s*/\s*(\d{4})\s*/\s*([A-Za-zĐđ0-9\-]+)")


def norm_id(x: str) -> str:
    """Normalize doc_id for matching: strip spaces, uppercase, ND->ND (already)."""
    return re.sub(r"\s+", "", str(x).upper().replace("NĐ", "ND").replace("Đ", "D"))


def collect_missing_doc_ids(jsonl: Path) -> Dict[str, int]:
    """Collect ALL distinct doc_ids referenced in citations (regardless of map status).

    The supplemental corpus is rebuilt fresh each time, so we should add all
    extractable doc_ids from the source dataset. Pulling from the legal_dataset_*
    output of a previous mapper run would shrink the set on each iteration.
    """
    counter: Dict[str, int] = {}
    for line in jsonl.open():
        rec = json.loads(line)
        for e in rec.get("entries", []):
            # Two possible shapes: pre-mapping (law_applied list) or post (law_chunks)
            for cit in e.get("law_applied", []) or []:
                m = RE_DOC_ID.search(str(cit))
                if m:
                    key = f"{m.group(1)}/{m.group(2)}/{m.group(3).upper()}"
                    counter[key] = counter.get(key, 0) + 1
            for ch in e.get("law_chunks", []) or []:
                cit = ch.get("citation", "")
                m = RE_DOC_ID.search(cit)
                if m:
                    key = f"{m.group(1)}/{m.group(2)}/{m.group(3).upper()}"
                    counter[key] = counter.get(key, 0) + 1
    return counter


def doc_id_to_law_sig(doc_id: str) -> str:
    """326/2016/UBTVQH14 -> 326_2016_nq-ubtvqh14
       43/2014/NĐ-CP    -> 43_2014_nđ-cp
       01/2019/NQ-HĐTP  -> 01_2019_nq-hđtp
       53/2014/QH13     -> 53_2014_qh13   (no nq- prefix for QH)
    Match the naming convention of the primary corpus.
    """
    parts = doc_id.split("/")
    num, year, cat = parts[0], parts[1], parts[2]
    cat_low = cat.lower()
    # Categories that are document-type prefixed (nq-, nđ-, tt-, etc.) -- keep as-is.
    has_doctype = bool(re.match(
        r"(nq-|nđ-|nd-|tt-|qđ-|qd-|pl-|ql-|ttlt-|al)", cat_low))
    # Pure issuer code (no doctype prefix) — UBTVQH / HĐTP / HĐND are typically NQ;
    # QH is the National Assembly (Luật) -> no prefix needed.
    if not has_doctype:
        if cat_low.startswith(("ubtvqh", "hdtp", "hđtp", "hdnd", "hđnd")):
            cat_low = "nq-" + cat_low
        # else: QHxx, CTN, etc. -> keep bare
    return f"{num}_{year}_{cat_low}"


# Match "Điều N. Title-text-on-same-line" OR just "Điều N." with title on next line
RE_ARTICLE = re.compile(
    r"(?<![\w\.])Điều\s+(?P<num>\d+[a-z]?)\.\s*(?P<rest>[^\n]{0,200})",
    re.UNICODE,
)
# Inline clause: "1. ", "2. " etc. at start of sentence
RE_CLAUSE_INLINE = re.compile(r"(?:^|\n|\.\s+)(?P<n>\d{1,2})\s*\.\s+", re.UNICODE)


def extract_articles(text: str) -> List[Dict]:
    """Split flat text into articles. Returns list of dicts."""
    matches = list(RE_ARTICLE.finditer(text))
    if not matches:
        return []
    out = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        # Title heuristic: first sentence (up to first ". " followed by digit or capital)
        rest = m.group("rest").strip()
        title_match = re.match(r"^([^\n]{0,200}?)(?:\s+\d+\.\s|$)", rest)
        title = (title_match.group(1) if title_match else rest)[:200].strip()

        # Extract clauses: split by "1. ", "2. " ...
        clauses: Dict[str, str] = {}
        clause_nums: List[int] = []
        # Use the body (rest + after) for clause extraction
        full_body = (rest + "\n" + body[len(rest):] if body.startswith(rest) else body)
        clause_iter = list(RE_CLAUSE_INLINE.finditer(full_body))
        for j, cm in enumerate(clause_iter):
            cn = cm.group("n")
            cs = cm.end()
            ce = clause_iter[j + 1].start() if j + 1 < len(clause_iter) else len(full_body)
            ctext = full_body[cs:ce].strip()
            # Drop noise: keep only if reasonably long
            if len(ctext) >= 5:
                clauses[cn] = ctext
                try:
                    clause_nums.append(int(cn))
                except ValueError:
                    pass

        out.append({
            "article_num": m.group("num"),
            "article_title": title,
            "article_text": body,
            "clauses_json": json.dumps(clauses, ensure_ascii=False) if clauses else "{}",
            "clause_nums": clause_nums,
        })
    return out


RE_HEADER = re.compile(
    r"(NGHỊ\s+QUYẾT|NGHỊ\s+ĐỊNH|THÔNG\s+TƯ\s+LIÊN\s+TỊCH|THÔNG\s+TƯ|"
    r"BỘ\s+LUẬT|LUẬT|QUYẾT\s+ĐỊNH|PHÁP\s+LỆNH)\s+([^\n]{0,150})",
    re.I,
)


def extract_short_name(text: str, doc_id: str) -> str:
    head = text[:5000]
    # Look for "Nghị quyết XYZ" / "Luật ABC" line pattern just after CỘNG HÒA block
    m = RE_HEADER.search(head)
    if m:
        s = (m.group(1) + " " + m.group(2)).strip()
        s = re.split(r"\s+(?:CHÍNH PHỦ|HỘI ĐỒNG|QUỐC HỘI|UỶ BAN|ỦY BAN|Căn cứ|--+)",
                     s)[0]
        s = re.sub(r"\s+", " ", s).strip()
        return s.upper()[:200]
    return f"VĂN BẢN {doc_id}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(INPUT_JSONL))
    p.add_argument("--output", default=str(OUT_PARQUET))
    p.add_argument("--hirine", default=str(HIRINE_PARQUET))
    p.add_argument("--min-count", type=int, default=1)
    args = p.parse_args()

    print(f"[supp] Collecting referenced doc_ids from {args.input}")
    need = collect_missing_doc_ids(Path(args.input))
    need = {k: v for k, v in need.items() if v >= args.min_count}
    print(f"[supp]   {len(need)} unique doc_ids referenced in dataset")

    # Drop doc_ids already present in primary corpus with FULL coverage
    # (skip only laws that have many chunks - sparse ones still benefit from hirine's full text)
    primary_path = PATHS.processed / "law_corpus.parquet"
    if primary_path.exists():
        primary = pd.read_parquet(primary_path, columns=["law_sig"])
        sig_counts = primary["law_sig"].dropna().astype(str).value_counts()
        well_covered = set(sig_counts[sig_counts >= 20].index)  # >=20 articles -> assume complete
        before = len(need)
        need = {k: v for k, v in need.items() if doc_id_to_law_sig(k) not in well_covered}
        print(f"[supp]   {before - len(need)} well-covered in primary; {len(need)} remain to fetch")

    # Extra docs to pull regardless of citation count (referenced by name-only,
    # OR present in primary corpus but with incomplete article coverage):
    EXTRA_DOC_IDS = [
        "53/2014/QH13",   # Luật Công chứng 2014
        "82/2006/QH11",   # Luật Công chứng 2006
        "83/2015/QH13",   # Luật Ngân sách Nhà nước 2015
        "02/2011/QH13",   # Luật Khiếu nại 2011
        "62/2014/QH13",   # Luật Tổ chức TAND 2014
        # Sparse in primary corpus -> fetch full from hirine
        "58/2020/QH14",   # Luật Hòa giải, đối thoại tại Tòa án (only 6 chunks in primary)
        "54/2010/QH12",   # Luật Trọng tài thương mại 2010 (only 3 chunks)
        "11/2012/NĐ-CP",  # NĐ 11/2012 (sparse)
        "37/2015/NĐ-CP",  # NĐ 37/2015 (sparse)
        "01/2017/NĐ-CP",  # sửa đổi NĐ Đất đai (sparse)
        "14/2008/QH12",   # Luật Thuế TNDN
        "104/2015/QH13",  # Luật Phí và Lệ phí
        "52/2019/QH14",   # Luật Cán bộ công chức (sửa đổi 2019)
        "132/2020/QH14",  # Luật Người Việt Nam đi làm việc ở nước ngoài
    ]
    for doc_id in EXTRA_DOC_IDS:
        need.setdefault(doc_id, 0)

    need_norm = {norm_id(k): k for k in need}

    print(f"[supp] Loading hirine corpus: {args.hirine}")
    df = pd.read_parquet(args.hirine, columns=["id", "text"])
    df["nid"] = df["id"].map(norm_id)
    df = df.drop_duplicates(subset=["nid"])
    print(f"[supp]   {len(df):,} docs in hirine")

    found = df[df["nid"].isin(need_norm.keys())].copy()
    print(f"[supp]   {len(found)} docs matched ({len(found)/len(need)*100:.1f}% of missing)")

    rows: List[Dict] = []
    n_articles = 0
    for _, r in tqdm(found.iterrows(), total=len(found), desc="parse"):
        doc_id = need_norm[r["nid"]]
        law_sig = doc_id_to_law_sig(doc_id)
        parts = doc_id.split("/")
        short_name = extract_short_name(r["text"], doc_id)
        arts = extract_articles(r["text"])
        n_articles += len(arts)
        for art in arts:
            rows.append({
                "chunk_id": f"supp_{law_sig}_{art['article_num']}",
                "law_sig": law_sig,
                "law_num": parts[0],
                "law_year": parts[1],
                "law_category": law_sig.split("_", 2)[2],
                "law_short_name": short_name,
                "article_num": art["article_num"],
                "article_title": art["article_title"],
                "clause_nums": art["clause_nums"],
                "content": (
                    f"[{law_sig.upper()}]\n\n{short_name}\n"
                    f"Điều {art['article_num']}. {art['article_title']}\n\n"
                    f"{art['article_text']}"
                ),
                "article_text": art["article_text"],
                "clauses_json": art["clauses_json"],
            })

    print(f"[supp] Total articles produced: {n_articles}")
    if not rows:
        print("[supp] Nothing to write")
        return

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(args.output, index=False)
    print(f"[supp] Wrote {len(out_df)} rows -> {args.output}")
    # Show coverage of supp
    supp_sigs = out_df["law_sig"].nunique()
    print(f"[supp] Unique law_sigs: {supp_sigs}")
    not_found = set(need_norm.keys()) - set(found["nid"])
    print(f"[supp] Still missing in hirine: {len(not_found)}")
    if not_found:
        print("[supp]   Examples:", [need_norm[k] for k in list(not_found)[:10]])


if __name__ == "__main__":
    main()
