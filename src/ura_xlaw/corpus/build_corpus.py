"""
Build a clean, indexed law corpus from `truro7/vn-law-corpus`.

Output: data/processed/law_corpus.parquet (and .csv preview) with columns:
  - chunk_id        : original corpus_id
  - law_sig         : "<lawNum>_<year>_<category>" lowercased, e.g. "92_2015_qh13"
  - law_num         : law number, e.g. "92"
  - law_year        : 4-digit year, e.g. "2015"
  - law_category    : e.g. "qh13", "nd-cp", "tt-nhnn"
  - law_short_name  : title parsed from chunk header (e.g. "LUẬT DÂN SỰ")
  - article_num     : "Điều N" number as string (may have suffix letter, e.g. "200a")
  - article_title   : text after "Điều N." up to newline
  - clause_nums     : sorted list of clause numbers covered by this chunk (e.g. [1,2])
  - content         : raw original text
  - article_text    : content stripped of [sig] header
  - clauses         : dict {clause_num: text}  -- best-effort split on "Khoản N." or "N."
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ura_xlaw.config import PATHS

RAW_PATH = PATHS.raw / "truro7_vn-law-corpus" / "train.csv"
OUT_PARQUET = PATHS.processed / "law_corpus.parquet"
OUT_CSV_PREVIEW = PATHS.processed / "law_corpus_preview.csv"


# --- corpus_id parsing ----------------------------------------------------- #

# Format A: "<lawNum>/<year>/<category>_<articleNum>"
RE_ID_A = re.compile(
    r"^(?P<num>[\w\-]+)\s*/\s*(?P<year>\d{4})\s*/\s*(?P<cat>[a-z0-9đ\-]+)"
    r"(?:_(?P<art>\d+[a-z]?))?$",
    re.IGNORECASE,
)

# Format B: "<seqId>_<lawNum>_<year>_<category>"
RE_ID_B = re.compile(
    r"^(?P<seq>\d+)_(?P<num>[\w\-]+)_(?P<year>\d{4})_(?P<cat>[\wđ\-]+)$",
    re.IGNORECASE,
)


def parse_corpus_id(
    cid: str,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Return (law_num, law_year, law_category, article_num_from_id)."""
    cid = str(cid).strip()
    m = RE_ID_A.match(cid)
    if m:
        return m.group("num"), m.group("year"), m.group("cat").lower(), m.group("art")
    m = RE_ID_B.match(cid)
    if m:
        return m.group("num"), m.group("year"), m.group("cat").lower(), None
    return None, None, None, None


# --- content parsing ------------------------------------------------------- #

# Header shape inside content: "[<lawNum>_<year>_<cat>]" then a title block, then "Điều X. ..."
RE_CONTENT_HEADER = re.compile(r"^\s*\[[^\]]+\]\s*", re.IGNORECASE)
RE_LAW_TITLE_LINE = re.compile(
    r"^(LUẬT|BỘ\s+LUẬT|NGHỊ\s+(?:QUYẾT|ĐỊNH)|THÔNG\s+TƯ|PHÁP\s+LỆNH|QUYẾT\s+ĐỊNH)\b.*",
    re.IGNORECASE,
)
RE_ARTICLE_LINE = re.compile(
    r"^\s*Điều\s+(?P<num>\d+[a-z]?)\.?\s*(?P<title>[^\n]*)", re.IGNORECASE
)
# clause specifier line like "Khoản 1:" or "Khoản 1, 2:" or "Khoản 1-3:"
RE_CLAUSE_SPEC = re.compile(
    r"^\s*Khoản\s+(?P<spec>[0-9,\s\-và]+)\s*:?\s*$", re.IGNORECASE
)
# inline clause start like "1." or "1)" at start of line
RE_INLINE_CLAUSE = re.compile(r"^\s*(?P<n>\d{1,2})\s*[\.\-)]\s+", re.IGNORECASE)


def _expand_clause_spec(spec: str) -> List[int]:
    """Parse "1, 2" or "1-3" or "1, 3, 5" or "1 và 2" -> [1,2,3,...]."""
    if not spec:
        return []
    s = re.sub(r"\bvà\b", ",", spec, flags=re.IGNORECASE)
    s = s.replace(";", ",")
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d{1,2})\s*-\s*(\d{1,2})$", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a <= b <= a + 50:
                out.extend(range(a, b + 1))
            continue
        m = re.match(r"^(\d{1,2})$", part)
        if m:
            out.append(int(m.group(1)))
    # dedupe & sort
    return sorted(set(out))


def parse_content(content: str) -> Dict[str, object]:
    """Extract law_short_name, article_num/title, clause_nums, clauses dict from a chunk."""
    if not isinstance(content, str):
        return {}

    # strip leading "[sig]" header
    body = RE_CONTENT_HEADER.sub("", content, count=1)

    law_short_name: Optional[str] = None
    article_num: Optional[str] = None
    article_title: Optional[str] = None
    clause_nums: List[int] = []
    clauses: Dict[int, str] = {}

    lines = [ln for ln in body.split("\n")]
    idx = 0

    # Find law title (first non-empty line that matches)
    while idx < len(lines):
        ln = lines[idx].strip()
        idx += 1
        if not ln:
            continue
        if RE_LAW_TITLE_LINE.match(ln):
            law_short_name = ln
            break
        # If the first non-empty line is already an "Điều ..." we have no separate title
        if RE_ARTICLE_LINE.match(ln):
            idx -= 1
            break

    # Find "Điều N. ..."
    while idx < len(lines):
        ln = lines[idx].strip()
        m = RE_ARTICLE_LINE.match(ln)
        if m:
            article_num = m.group("num")
            article_title = (m.group("title") or "").strip().rstrip(".")
            idx += 1
            break
        idx += 1

    # Find optional "Khoản X:" specifier
    while idx < len(lines):
        ln = lines[idx].strip()
        if not ln:
            idx += 1
            continue
        m = RE_CLAUSE_SPEC.match(ln)
        if m:
            clause_nums = _expand_clause_spec(m.group("spec"))
            idx += 1
            break
        # No clause specifier — body starts here
        break

    # Remaining body = clauses content
    rest = "\n".join(lines[idx:]).strip()

    # Split into individual clauses on lines starting with "<N>." or "<N>)"
    if rest:
        cur_n: Optional[int] = None
        cur_buf: List[str] = []
        for ln in rest.split("\n"):
            m = RE_INLINE_CLAUSE.match(ln)
            if m:
                if cur_n is not None:
                    clauses[cur_n] = "\n".join(cur_buf).strip()
                cur_n = int(m.group("n"))
                cur_buf = [ln]
            else:
                cur_buf.append(ln)
        if cur_n is not None:
            clauses[cur_n] = "\n".join(cur_buf).strip()

    # If clause_nums empty but we extracted clauses inline, infer them
    if not clause_nums and clauses:
        clause_nums = sorted(clauses.keys())

    article_text = rest

    return {
        "law_short_name": law_short_name,
        "article_num": article_num,
        "article_title": article_title,
        "clause_nums": clause_nums,
        "clauses": clauses,
        "article_text": article_text,
    }


def build_corpus(
    raw_path: Path = RAW_PATH, out_parquet: Path = OUT_PARQUET
) -> pd.DataFrame:
    if not raw_path.exists():
        from datasets import load_dataset

        print("[build_corpus] Downloading truro7/vn-law-corpus...")
        ds = load_dataset("truro7/vn-law-corpus", split="train")
        df = ds.to_pandas()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False, encoding="utf-8-sig")
        print(f"[build_corpus] Cached -> {raw_path} ({len(df)} rows)")
    else:
        df = pd.read_csv(raw_path)
        print(f"[build_corpus] Loaded {len(df)} rows from cache")

    rows = []
    for _, r in df.iterrows():
        cid = r["corpus_id"]
        law_num, law_year, law_cat, art_from_id = parse_corpus_id(cid)
        parsed = parse_content(r["content"])

        article_num = parsed.get("article_num") or art_from_id
        rows.append(
            {
                "chunk_id": cid,
                "law_sig": (
                    f"{law_num}_{law_year}_{law_cat}".lower()
                    if (law_num and law_year and law_cat)
                    else None
                ),
                "law_num": law_num,
                "law_year": law_year,
                "law_category": law_cat,
                "law_short_name": parsed.get("law_short_name"),
                "article_num": article_num,
                "article_title": parsed.get("article_title"),
                "clause_nums": parsed.get("clause_nums") or [],
                "content": r["content"],
                "article_text": parsed.get("article_text"),
                "clauses_json": json.dumps(
                    parsed.get("clauses") or {}, ensure_ascii=False
                ),
            }
        )

    out = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)
    out.head(50).to_csv(OUT_CSV_PREVIEW, index=False, encoding="utf-8-sig")

    print(f"\n[build_corpus] Wrote {len(out)} rows -> {out_parquet}")
    print(f"[build_corpus] Preview (first 50) -> {OUT_CSV_PREVIEW}")
    print(
        f"[build_corpus]   law_sig coverage:    {out['law_sig'].notna().sum()}/{len(out)}"
    )
    print(
        f"[build_corpus]   article_num coverage:{out['article_num'].notna().sum()}/{len(out)}"
    )
    print(f"[build_corpus]   unique laws (sig):   {out['law_sig'].nunique()}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the URA-xLaw law corpus.")
    parser.add_argument("--input", default=str(RAW_PATH))
    parser.add_argument("--output", default=str(OUT_PARQUET))
    args = parser.parse_args()
    build_corpus(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
