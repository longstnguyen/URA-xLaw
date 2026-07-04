"""
Map citations in `law_applied` (e.g. "D91:Luật Các tổ chức tín dụng",
"D26:K3:Bộ luật Tố tụng dân sự", "D35:K1:da:BLTTDS", "AL11/2017/AL")
to chunk content from the indexed law corpus.

Output: data/processed/qa_mapped.jsonl
  Each Q&A entry gains a `law_chunks` list (same length as `law_applied`),
  where each item is either:
    {
      "citation":  original citation string,
      "matched":   true|false,
      "law_sig":   matched law_sig (if any),
      "article":   article number,
      "clause":    clause number (or null),
      "point":     point letter (or null),
      "chunk_id":  matched corpus chunk_id,
      "chunk_text":matched chunk content (article + clause text),
      "fail_reason": str or null
    }

Coverage report goes to data/processed/law_chunks_coverage.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from ura_xlaw.config import PATHS


CORPUS_PATH = PATHS.processed / "law_corpus.parquet"
INPUT_JSONL = PATHS.processed / "qa_generated_openai.jsonl"
OUTPUT_JSONL = PATHS.processed / "qa_mapped.jsonl"
COVERAGE_REPORT = PATHS.processed / "law_chunks_coverage.json"


# --- Citation parsing ------------------------------------------------------ #

# D<n>[:K<n>][:d<x>]:LawName  with possibly extra colons in law name
RE_ARTICLE_CIT = re.compile(
    r"^D(?P<art>\d+[A-Za-z]?)"
    r"(?::K(?P<clause>\d+[A-Za-z]?))?"
    r"(?::d(?P<point>[a-zA-Z0-9]+))?"
    r"(?::(?P<law>.+))?$"
)
# AL<n>/<year>/AL[:Name]
RE_AN_LE_CIT = re.compile(r"^AL\s*(?P<num>\d+)\s*/\s*(?P<year>\d{4})\s*/\s*AL(?::(?P<name>.+))?$")


def parse_citation(c: str) -> Optional[Dict]:
    s = c.strip()
    m = RE_AN_LE_CIT.match(s)
    if m:
        return {
            "type": "an_le",
            "num": m.group("num"),
            "year": m.group("year"),
            "name": (m.group("name") or "").strip() or None,
            "article": None, "clause": None, "point": None, "law": None,
        }
    m = RE_ARTICLE_CIT.match(s)
    if m:
        # If clause group looks like a clause-number-and-extra-colons leak:
        # original grammar guarantees that after K<n> next ':' starts d<x> or law name.
        return {
            "type": "article",
            "article": m.group("art"),
            "clause": m.group("clause"),
            "point": m.group("point"),
            "law": (m.group("law") or "").strip() or None,
        }
    return None


# --- Vietnamese normalization --------------------------------------------- #

def norm(s: Optional[str]) -> str:
    if not s:
        return ""
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("đ", "d").replace("ð", "d")
    s = re.sub(r"[^\w\s/\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonical_law_name(raw: str) -> str:
    """Normalize raw law name to a comparable key.
    - lowercase, drop diacritics
    - drop "nam YYYY" -> "YYYY"
    - drop common noise: "(sua doi ...)", trailing year
    """
    n = norm(raw)
    n = re.sub(r"\bnam\s+(\d{4})\b", r"\1", n)
    n = re.sub(r"\(.*?\)", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


# --- Manual aliases (Vietnamese law name -> law_sig) ---------------------- #
#
# Strategy: if user does not specify a year, map to the LATEST in-force version
# present in the corpus. If user specifies an older year, prefer that version.

ALIASES: Dict[str, str] = {
    # ---- Bộ luật ----
    "bo luat dan su": "91_2015_qh13",
    "luat dan su": "91_2015_qh13",
    "bo luat dan su 2015": "91_2015_qh13",
    "luat dan su 2015": "91_2015_qh13",
    "bo luat to tung dan su": "92_2015_qh13",
    "luat to tung dan su": "92_2015_qh13",
    "bo luat to tung dan su 2015": "92_2015_qh13",
    "luat to tung dan su 2015": "92_2015_qh13",
    "bo luat hinh su": "100_2015_qh13",
    "luat hinh su": "100_2015_qh13",
    "bo luat hinh su 2015": "100_2015_qh13",
    "luat hinh su 2015": "100_2015_qh13",
    "bo luat to tung hinh su": "101_2015_qh13",
    "luat to tung hinh su": "101_2015_qh13",
    "bo luat to tung hinh su 2015": "101_2015_qh13",
    "luat to tung hinh su 2015": "101_2015_qh13",
    "bo luat lao dong": "45_2019_qh14",
    "luat lao dong": "45_2019_qh14",
    "bo luat lao dong 2019": "45_2019_qh14",
    "luat lao dong 2019": "45_2019_qh14",

    # ---- Luật ----
    "luat to tung hanh chinh": "93_2015_qh13",
    "bo luat to tung hanh chinh": "93_2015_qh13",
    "luat to tung hanh chinh 2015": "93_2015_qh13",
    "bo luat to tung hanh chinh 2015": "93_2015_qh13",
    "luat dat dai": "45_2013_qh13",
    "bo luat dat dai": "45_2013_qh13",
    "luat dat dai 2013": "45_2013_qh13",
    "bo luat dat dai 2013": "45_2013_qh13",
    "luat hon nhan va gia dinh": "52_2014_qh13",
    "luat hon nhan gia dinh": "52_2014_qh13",
    "luat hon nhan va gia dinh 2014": "52_2014_qh13",
    "luat cac to chuc tin dung": "47_2010_qh12",
    "luat to chuc tin dung": "47_2010_qh12",
    "luat thuong mai": "36_2005_qh11",
    "luat thuong mai 2005": "36_2005_qh11",
    "luat nha o": "65_2014_qh13",
    "luat nha o 2014": "65_2014_qh13",
    "luat thi hanh an dan su": "26_2008_qh12",
    "luat thi hanh an hinh su": "41_2019_qh14",
    "luat doanh nghiep": "59_2020_qh14",
    "luat doanh nghiep 2020": "59_2020_qh14",
    "luat bao hiem xa hoi": "58_2014_qh13",
    "luat bao hiem xa hoi 2014": "58_2014_qh13",
    "luat xay dung": "50_2014_qh13",
    "luat pha san": "51_2014_qh13",
    "luat dau tu": "61_2020_qh14",
    "luat khieu nai": "02_2011_qh13",
    "luat khieu nai 2011": "02_2011_qh13",
    "luat dau gia tai san": "01_2016_qh14",
    "luat dau gia tai san 2016": "01_2016_qh14",
    "luat dau gia": "01_2016_qh14",
    "luat xu ly vi pham hanh chinh": "15_2012_qh13",
    "luat xu ly vi pham hanh chinh 2012": "15_2012_qh13",
    "luat kham benh chua benh": "15_2023_qh15",
    "luat kham chua benh": "15_2023_qh15",
    "luat tre em": "102_2016_qh13",
    "luat giao thong duong bo": "23_2008_qh12",
    "luat bao hiem y te": "25_2008_qh12",
    "luat can bo cong chuc": "22_2008_qh12",
    "luat vien chuc": "58_2010_qh12",
    "luat giao duc": "43_2019_qh14",
    "luat trach nhiem boi thuong cua nha nuoc": "10_2017_qh14",
    "luat ban hanh van ban quy pham phap luat": "80_2015_qh13",
    "luat ngan hang nha nuoc viet nam": "46_2010_qh12",
    "luat ngan hang nha nuoc": "46_2010_qh12",
    "luat to tung hanh chinh nam 2015": "93_2015_qh13",
    "luat dau thau": "22_2023_qh15",
    "luat tham phan": None,  # not applicable
    "luat to chuc toa an nhan dan": "62_2014_qh13",  # may not exist; checked at runtime
    # Newly verified in corpus (small chunk counts but present)
    "luat hoa giai doi thoai tai toa an": "58_2020_qh14",
    "luat hoa giai doi thoai tai toa an 2020": "58_2020_qh14",
    "luat trong tai thuong mai": "54_2010_qh12",
    "luat trong tai thuong mai 2010": "54_2010_qh12",
    "luat quan ly su dung vu khi vat lieu no va cong cu ho tro": "14_2017_qh14",
    "luat ve quan ly su dung vu khi vat lieu no va cong cu ho tro": "14_2017_qh14",
    "luat quan ly su dung vu khi vat lieu no": "14_2017_qh14",

    # ---- More from coverage gap analysis ----
    "luat cong chung": "53_2014_qh13",
    "luat cong chung 2014": "53_2014_qh13",
    "luat ngan sach nha nuoc": "83_2015_qh13",
    "bo luat ngan sach nha nuoc": "83_2015_qh13",
    "luat ngan sach nha nuoc 2015": "83_2015_qh13",
    # 326/UBTVQH14 (without year) -> NQ 326/2016/UBTVQH14
    "nghi quyet 326/ubtvqh14": "326_2016_nq-ubtvqh14",

    # ---- Older versions (in case user explicitly cites them) ----
    "bo luat dan su 2005": "33_2005_qh11",
    "luat dan su 2005": "33_2005_qh11",
    "bo luat dan su 1995": None,
    "bo luat hinh su 1999": "15_1999_qh10",
    "luat hinh su 1999": "15_1999_qh10",
    "bo luat lao dong 2012": "10_2012_qh13",
    "luat lao dong 2012": "10_2012_qh13",
    "bo luat lao dong 1994": "35_l_ctn",
    "luat hon nhan va gia dinh 2000": "22_2000_qh10",
    "luat dat dai 2003": "13_2003_qh11",

    # ---- TVPL-crawled new laws ----
    "luat tu phap nguoi chua thanh nien": "59_2024_qh15",
    "luat tu phap nguoi chua thanh nien 2024": "59_2024_qh15",
    "bo luat tu phap nguoi chua thanh nien": "59_2024_qh15",
    "luat dat dai 2024": "31_2024_qh15",
    "bo luat dat dai 2024": "31_2024_qh15",
    "luat dat dai nam 2024": "31_2024_qh15",
    "luat trat tu an toan giao thong duong bo nam 2024": "36_2024_qh15",
    "luat trat tu an toan giao thong duong bo 2024": "36_2024_qh15",
    "nghi quyet 01/2025/nq-hdtp": "01_2025_nq-hđtp",
    "nghi quyet 01/2025/nq-hđtp": "01_2025_nq-hđtp",
    "nghi dinh 101/2024/nd-cp": "101_2024_nđ-cp",
    "nghi dinh 101/2024/nđ-cp": "101_2024_nđ-cp",
    "nghi dinh 123/2024/nd-cp": "123_2024_nđ-cp",
    "nghi dinh 123/2024/nđ-cp": "123_2024_nđ-cp",
    "luat so 85/2025/qh15": "85_2025_qh15",
    "luat so 86/2025/qh15": "86_2025_qh15",
    "luat sua doi bo luat hinh su 2025": "86_2025_qh15",
    "luat sua doi bo luat to tung hinh su 2025": "85_2025_qh15",
    "thong tu lien tich so 04/2021/ttlt-bca-bqp-tandtc-vksndtc": "04_2021_ttlt-bca-bqp-tandtc-vksndtc",
    "thong tu lien tich 04/2021/ttlt-bca-bqp-tandtc-vksndtc": "04_2021_ttlt-bca-bqp-tandtc-vksndtc",
    "luat giai quyet khieu nai 2011": "02_2011_qh13",
    "nghi quyet 326/2016/ubtvqh": "326_2016_nq-ubtvqh14",
}


# --- Corpus index --------------------------------------------------------- #

class CorpusIndex:
    def __init__(self, df: pd.DataFrame):
        self.df = df

        # Pre-fill missing law_short_name within each law_sig with first non-null
        df_sn = df.dropna(subset=["law_sig", "law_short_name"]).groupby("law_sig")["law_short_name"].first()
        self.sig_to_short: Dict[str, str] = df_sn.to_dict()
        # All sigs that exist (even with null short_name)
        self.all_sigs: set = set(df["law_sig"].dropna().astype(str).unique())

        # Index: (law_sig, article_num) -> list of row indices
        self.by_sig_article: Dict[Tuple[str, str], List[int]] = {}
        for i, r in df.iterrows():
            sig = r["law_sig"]
            art = r["article_num"]
            if sig and art:
                self.by_sig_article.setdefault((str(sig), str(art)), []).append(i)

        # Reverse alias from corpus law_short_name -> law_sig (best effort)
        # Build short_name canonical map
        self.canon_to_sig: Dict[str, str] = {}
        for sig, sn in self.sig_to_short.items():
            if not sn:
                continue
            key = canonical_law_name(sn)
            # prefer not to overwrite if multiple laws share a canonical short name (shouldn't happen often)
            self.canon_to_sig.setdefault(key, sig)

        # Validate & filter ALIASES against corpus
        self.aliases: Dict[str, str] = {}
        for k, sig in ALIASES.items():
            if sig and sig in self.all_sigs:
                self.aliases[k] = sig
        # Add corpus short-name canon as additional aliases
        for canon, sig in self.canon_to_sig.items():
            self.aliases.setdefault(canon, sig)

    def resolve_law_sig(self, raw_law_name: str) -> Optional[str]:
        if not raw_law_name:
            return None
        key = canonical_law_name(raw_law_name)
        if key in self.aliases:
            return self.aliases[key]

        # Strip trailing 4-digit year and try again (e.g. "Bộ luật Dân sự 2015")
        no_year = re.sub(r"\s*\b(?:19|20)\d{2}\b\s*$", "", key).strip()
        if no_year and no_year != key and no_year in self.aliases:
            return self.aliases[no_year]

        # Doc-id pattern: "Nghị định 43/2014/NĐ-CP", "Nghị quyết 326/2016/UBTVQH14", etc.
        # Try to derive corpus law_sig directly: <num>_<year>_<cat>
        m = re.search(r"(\d+)\s*/\s*(\d{4})\s*/\s*([\wđĐ\-]+)", raw_law_name, re.UNICODE)
        if m:
            num = m.group(1).lstrip("0") or "0"
            year = m.group(2)
            cat_raw = m.group(3).lower()
            # Determine document type from prefix (NQ vs raw cat)
            prefix = ""
            head = raw_law_name.lower()
            if "nghị quyết" in head or "nghi quyet" in head:
                if not cat_raw.startswith("nq-"):
                    prefix = "nq-"
            # Try multiple variants
            cat_variants = {cat_raw, cat_raw.replace("đ", "d"), cat_raw.replace("d", "đ", 1)}
            for cv in cat_variants:
                for c in (cv, prefix + cv if prefix else cv):
                    for n in (num, num.zfill(2), num.zfill(3)):
                        sig = f"{n}_{year}_{c}"
                        if sig in self.all_sigs:
                            return sig

        # Substring fallback over canonical short names (slow but small)
        # Prefer longest canon key contained in raw key
        best = None
        best_len = 0
        for canon, sig in self.canon_to_sig.items():
            if len(canon) >= 8 and canon in key and len(canon) > best_len:
                best = sig
                best_len = len(canon)
        return best

    def lookup_chunks(self, sig: str, article: str) -> List[int]:
        """All corpus row indices for (sig, article)."""
        return self.by_sig_article.get((str(sig), str(article)), [])


# --- Chunk text rendering ------------------------------------------------- #

def render_chunk(corpus: pd.DataFrame, idx: int, clause: Optional[str] = None) -> str:
    """Return a clean text for a corpus row, optionally narrowed to a specific clause."""
    row = corpus.iloc[idx]
    short = row.get("law_short_name") or ""
    art = row.get("article_num") or ""
    title = row.get("article_title") or ""
    body = row.get("article_text") or ""

    header = f"[{short}] Điều {art}"
    if title:
        header += f". {title}"

    if clause:
        try:
            clauses = json.loads(row.get("clauses_json") or "{}")
        except Exception:
            clauses = {}
        ctext = clauses.get(str(clause)) or clauses.get(clause)
        if ctext:
            return f"{header}\n{ctext}".strip()

    return f"{header}\n{body}".strip()


# --- Main mapping --------------------------------------------------------- #

def map_citation(
    cit: str, parsed: Dict, idx: CorpusIndex, corpus: pd.DataFrame
) -> Dict:
    out = {
        "citation": cit,
        "matched": False,
        "law_sig": None,
        "article": None,
        "clause": None,
        "point": None,
        "chunk_id": None,
        "chunk_text": None,
        "fail_reason": None,
    }

    if parsed is None:
        out["fail_reason"] = "unparseable"
        return out

    if parsed["type"] == "an_le":
        n_str = parsed["num"]
        n_int = int(n_str)
        year = parsed["year"]
        candidates = [f"{n_int}_{year}_al", f"{n_str}_{year}_al",
                      f"{n_int:02d}_{year}_al"]
        rows: List[int] = []
        sig = None
        for c in candidates:
            r2 = idx.lookup_chunks(c, "1")
            if r2:
                sig = c
                rows = r2
                break
        if not rows:
            out["fail_reason"] = "an_le_not_in_corpus"
            return out
        out["law_sig"] = sig
        chosen = rows[0]
        out["matched"] = True
        out["chunk_id"] = corpus.iloc[chosen]["chunk_id"]
        out["chunk_text"] = corpus.iloc[chosen]["content"]
        return out

    out["article"] = parsed.get("article")
    out["clause"] = parsed.get("clause")
    out["point"] = parsed.get("point")

    sig = idx.resolve_law_sig(parsed.get("law") or "")
    if not sig:
        out["fail_reason"] = "law_name_unmapped"
        return out
    out["law_sig"] = sig

    rows = idx.lookup_chunks(sig, parsed["article"])
    if not rows:
        out["fail_reason"] = "article_not_found"
        return out

    # Prefer a row that explicitly covers the requested clause (clause_nums column)
    chosen = rows[0]
    if parsed.get("clause"):
        try:
            cn_int = int(re.match(r"\d+", parsed["clause"]).group(0))
            for ridx in rows:
                cnums = corpus.iloc[ridx]["clause_nums"]
                if cnums is not None and len(cnums) and cn_int in list(cnums):
                    chosen = ridx
                    break
        except Exception:
            pass

    out["matched"] = True
    out["chunk_id"] = corpus.iloc[chosen]["chunk_id"]
    out["chunk_text"] = render_chunk(corpus, chosen, clause=parsed.get("clause"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(CORPUS_PATH))
    p.add_argument("--supp", default=str(PATHS.processed / "law_corpus_supplemental.parquet"),
                   help="Optional supplemental corpus parquet (merged in if present)")
    p.add_argument("--input", default=str(INPUT_JSONL))
    p.add_argument("--output", default=str(OUTPUT_JSONL))
    p.add_argument("--report", default=str(COVERAGE_REPORT))
    args = p.parse_args()

    print(f"[map] Loading corpus from {args.corpus}")
    corpus = pd.read_parquet(args.corpus)
    print(f"[map]   primary: {len(corpus):,} chunks, {corpus['law_sig'].nunique():,} laws")
    supp_path = Path(args.supp)
    if supp_path.exists():
        supp = pd.read_parquet(supp_path)
        print(f"[map]   supplemental: {len(supp):,} chunks, {supp['law_sig'].nunique():,} laws")
        corpus = pd.concat([corpus, supp], ignore_index=True)
        print(f"[map]   merged: {len(corpus):,} chunks, {corpus['law_sig'].nunique():,} laws")

    idx = CorpusIndex(corpus)
    print(f"[map]   {len(idx.aliases):,} aliases active")

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_match = 0
    fail_counter: Counter = Counter()
    unmapped_law_names: Counter = Counter()
    sig_hit: Counter = Counter()

    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in tqdm(fin, desc="map"):
            doc = json.loads(line)
            for entry in doc.get("entries", []):
                chunks: List[Dict] = []
                for cit in entry.get("law_applied", []):
                    parsed = parse_citation(cit)
                    res = map_citation(cit, parsed, idx, corpus)
                    chunks.append(res)
                    n_total += 1
                    if res["matched"]:
                        n_match += 1
                        sig_hit[res["law_sig"]] += 1
                    else:
                        fail_counter[res["fail_reason"]] += 1
                        if res["fail_reason"] == "law_name_unmapped" and parsed and parsed.get("law"):
                            unmapped_law_names[parsed["law"]] += 1
                entry["law_chunks"] = chunks
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

    coverage = (n_match / n_total) if n_total else 0.0
    report = {
        "total_citations": n_total,
        "matched": n_match,
        "coverage_pct": round(coverage * 100, 2),
        "failure_breakdown": dict(fail_counter.most_common()),
        "top_unmapped_law_names": dict(unmapped_law_names.most_common(40)),
        "top_matched_laws": [
            {"law_sig": s, "count": n, "name": idx.sig_to_short.get(s)}
            for s, n in sig_hit.most_common(30)
        ],
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print()
    print(f"[map] Total citations: {n_total:,}")
    print(f"[map] Matched:         {n_match:,}  ({coverage*100:.2f}%)")
    print(f"[map] Failure breakdown:")
    for k, v in fail_counter.most_common():
        print(f"        {v:5d}  {k}")
    print()
    print(f"[map] Output:  {out_path}")
    print(f"[map] Report:  {args.report}")


if __name__ == "__main__":
    main()
