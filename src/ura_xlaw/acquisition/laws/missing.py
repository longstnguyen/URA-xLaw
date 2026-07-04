"""
Crawl missing laws from thuvienphapluat.vn (TVPL) via DuckDuckGo search.

Workflow:
  1. Read coverage report -> extract unmapped doc IDs (e.g. "326/2016/UBTVQH14")
     and name-only citations.
  2. For each doc, query DDG: site:thuvienphapluat.vn "<doc_id>"
  3. Fetch first matching /van-ban/ page from TVPL.
  4. Strip HTML -> clean text.
  5. Split by "Điều N. Title" -> per-article rows matching law_corpus.parquet schema.
  6. Cache raw HTML to data/raw/crawled/ ; output parquet to data/processed/crawled_corpus.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
import urllib.parse
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

from ura_xlaw.config import PATHS

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
HEADERS = {"User-Agent": UA, "Accept-Language": "vi,en;q=0.9"}

RAW_DIR = PATHS.raw / "crawled"
RAW_DIR.mkdir(parents=True, exist_ok=True)
COVERAGE_REPORT = PATHS.processed / "law_chunks_coverage.json"
INPUT_JSONL = PATHS.processed / "qa_mapped.jsonl"
OUT_PARQUET = PATHS.processed / "crawled_corpus.parquet"

# --------------------------------------------------------------------------- #
# Step 1: collect missing doc_ids from current mapping result
# --------------------------------------------------------------------------- #

RE_DOC_ID = re.compile(r"(\d+)\s*/\s*(\d{4})\s*/\s*([A-Za-zĐđ0-9\-]+)")


def collect_missing(jsonl: Path, min_count: int = 1) -> Tuple[Counter, Counter]:
    """Return (doc_id_counter, name_only_counter)."""
    doc_ids = Counter()
    names = Counter()
    for line in jsonl.open():
        rec = json.loads(line)
        for e in rec.get("entries", []):
            for ch in e.get("law_chunks", []):
                if ch.get("matched"):
                    continue
                if ch.get("fail_reason") != "law_name_unmapped":
                    continue
                cit = ch["citation"]
                law = cit.split(":")[-1]
                m = RE_DOC_ID.search(law)
                if m:
                    key = f"{m.group(1)}/{m.group(2)}/{m.group(3).upper()}"
                    doc_ids[key] += 1
                else:
                    names[law.strip()] += 1
    doc_ids = Counter({k: v for k, v in doc_ids.items() if v >= min_count})
    names = Counter({k: v for k, v in names.items() if v >= min_count})
    return doc_ids, names


# --------------------------------------------------------------------------- #
# Step 2: DDG search
# --------------------------------------------------------------------------- #

RE_TVPL_URL = re.compile(r"https?://(?:www\.)?thuvienphapluat\.vn/van-ban/[^\"'<>\s&]+\.aspx", re.I)


def ddg_search_tvpl(query: str, session: requests.Session, retries: int = 3) -> Optional[str]:
    url = "https://duckduckgo.com/html/"
    params = {"q": f'site:thuvienphapluat.vn "{query}"'}
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, headers=HEADERS, timeout=20)
            if r.status_code != 200:
                time.sleep(2 + attempt)
                continue
            urls = RE_TVPL_URL.findall(r.text)
            # Skip /EN/ (English) and /tieng-anh/ versions
            urls = [u for u in urls if "/van-ban/EN/" not in u and "/tieng-anh" not in u]
            # Decode any DDG redirect wrapper
            urls = [urllib.parse.unquote(u) for u in urls]
            if urls:
                return urls[0]
        except requests.RequestException:
            time.sleep(2 + attempt)
    return None


# --------------------------------------------------------------------------- #
# Step 3: fetch + strip HTML
# --------------------------------------------------------------------------- #

class TextExtractor(HTMLParser):
    """Extract visible text, preserving paragraph breaks."""

    BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5"}
    SKIP = {"script", "style", "noscript"}

    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        s = "".join(self.parts)
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n[ \t]+", "\n", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()


def fetch_tvpl(url: str, session: requests.Session, retries: int = 3) -> Optional[str]:
    for attempt in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200 and len(r.text) > 5000:
                return r.text
        except requests.RequestException:
            pass
        time.sleep(2 + attempt)
    return None


def extract_main_text(html: str) -> str:
    """TVPL document body is in #tab1 div. Fall back to full strip if not found."""
    m = re.search(r'<div[^>]*id=["\']?tab1["\']?[^>]*>(.*?)</div>\s*<div[^>]*id=["\']?tab2',
                  html, re.S | re.I)
    body = m.group(1) if m else html
    p = TextExtractor()
    p.feed(body)
    return p.text()


# --------------------------------------------------------------------------- #
# Step 4: parse Điều/Khoản
# --------------------------------------------------------------------------- #

RE_ARTICLE = re.compile(r"^\s*Điều\s+(?P<num>\d+[a-z]?)\.?\s*(?P<title>.*)$", re.M)
RE_CLAUSE_INLINE = re.compile(r"^\s*(?P<n>\d{1,2})\s*[\.)\-]\s+", re.M)


def parse_articles(text: str) -> List[Dict]:
    """Return list of {article_num, article_title, article_text, clauses_json, clause_nums}."""
    matches = list(RE_ARTICLE.finditer(text))
    if not matches:
        return []
    out = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        title = m.group("title").strip()
        # Split by inline clauses to populate clauses dict
        clauses = {}
        clause_nums = []
        # Find clause splits "1. ", "2. " at line start
        cm = list(RE_CLAUSE_INLINE.finditer(body))
        for j, cmm in enumerate(cm):
            cn = cmm.group("n")
            cstart = cmm.end()
            cend = cm[j + 1].start() if j + 1 < len(cm) else len(body)
            ctext = body[cstart:cend].strip()
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


# --------------------------------------------------------------------------- #
# Step 5: extract law metadata (short_name, signature)
# --------------------------------------------------------------------------- #

RE_LAW_HEADER = re.compile(
    r"(NGHỊ\s+QUYẾT|NGHỊ\s+ĐỊNH|THÔNG\s+TƯ|LUẬT|BỘ\s+LUẬT|QUYẾT\s+ĐỊNH|"
    r"PHÁP\s+LỆNH|THÔNG\s+TƯ\s+LIÊN\s+TỊCH)\s*([\n\r][^\n\r]+)?",
    re.I,
)


def extract_short_name(text: str, fallback: str) -> str:
    """Heuristic: the 'LUẬT/NGHỊ QUYẾT/...' header line near the top."""
    head = text[:3000]
    m = RE_LAW_HEADER.search(head)
    if m:
        # Take up to 200 chars from the match
        chunk = head[m.start(): m.start() + 200].strip()
        # First line(s) until blank or "Căn cứ"
        chunk = re.split(r"\n\n|Căn cứ", chunk)[0].strip()
        chunk = re.sub(r"\s+", " ", chunk)
        return chunk[:200]
    return fallback


# --------------------------------------------------------------------------- #
# Main crawl
# --------------------------------------------------------------------------- #

def doc_id_to_filename(doc_id: str) -> str:
    return doc_id.replace("/", "_").replace(" ", "")


def doc_id_to_law_sig(doc_id: str) -> str:
    """326/2016/UBTVQH14 -> 326_2016_nq-ubtvqh14 (or just 326_2016_ubtvqh14
       depending on convention). For NQ docs we add nq- prefix.
    """
    parts = doc_id.split("/")
    num, year, cat = parts[0], parts[1], parts[2]
    cat_low = cat.lower()
    # If cat looks like UBTVQHxx or HĐTP and not already prefixed, add nq-
    if not re.match(r"(nq-|nđ-|nd-|tt-|qđ-|pl-|ql-|qh|hđnn|lct)", cat_low):
        cat_low = "nq-" + cat_low
    elif cat_low.startswith("nq-"):
        pass
    return f"{num}_{year}_{cat_low}"


def crawl_doc(doc_id: str, session: requests.Session) -> Optional[Dict]:
    """Crawl one document by doc_id. Returns dict with rows or None."""
    cache_html = RAW_DIR / f"{doc_id_to_filename(doc_id)}.html"
    cache_meta = RAW_DIR / f"{doc_id_to_filename(doc_id)}.json"

    if cache_meta.exists():
        try:
            return json.loads(cache_meta.read_text())
        except Exception:
            pass

    if cache_html.exists():
        html = cache_html.read_text()
        url = "(cached)"
    else:
        url = ddg_search_tvpl(doc_id, session)
        if not url:
            meta = {"doc_id": doc_id, "url": None, "rows": [], "error": "no_search_result"}
            cache_meta.write_text(json.dumps(meta, ensure_ascii=False))
            return meta
        time.sleep(1.0)  # polite
        html = fetch_tvpl(url, session)
        if not html:
            meta = {"doc_id": doc_id, "url": url, "rows": [], "error": "fetch_failed"}
            cache_meta.write_text(json.dumps(meta, ensure_ascii=False))
            return meta
        cache_html.write_text(html)
        time.sleep(1.0)

    text = extract_main_text(html)
    short_name = extract_short_name(text, fallback=f"VĂN BẢN {doc_id}")
    arts = parse_articles(text)

    law_sig = doc_id_to_law_sig(doc_id)
    parts = doc_id.split("/")
    rows = []
    for art in arts:
        rows.append({
            "chunk_id": f"{law_sig}_{art['article_num']}",
            "law_sig": law_sig,
            "law_num": parts[0],
            "law_year": parts[1],
            "law_category": law_sig.split("_", 2)[2],
            "law_short_name": short_name,
            "article_num": art["article_num"],
            "article_title": art["article_title"],
            "clause_nums": art["clause_nums"],
            "content": f"[{law_sig.upper()}]\n\n{short_name}\nĐiều {art['article_num']}. {art['article_title']}\n\n{art['article_text']}",
            "article_text": art["article_text"],
            "clauses_json": art["clauses_json"],
        })
    meta = {"doc_id": doc_id, "url": url, "law_sig": law_sig, "short_name": short_name,
            "n_articles": len(rows), "rows": rows, "error": None if rows else "no_articles_parsed"}
    cache_meta.write_text(json.dumps(meta, ensure_ascii=False))
    return meta


def crawl_name_only(name: str, session: requests.Session) -> Optional[Dict]:
    """Search by law name (no doc_id). Use first /van-ban/ result."""
    safe = re.sub(r'[^\w\s]', ' ', name)[:80]
    cache_meta = RAW_DIR / f"NAME_{re.sub(r'[^a-zA-Z0-9]+','_',unicodedata.normalize('NFKD',safe).encode('ascii','ignore').decode())[:60]}.json"
    cache_html = cache_meta.with_suffix(".html")
    if cache_meta.exists():
        try:
            return json.loads(cache_meta.read_text())
        except Exception:
            pass

    if cache_html.exists():
        html = cache_html.read_text(); url = "(cached)"
    else:
        url = ddg_search_tvpl(name, session)
        if not url:
            meta = {"name": name, "url": None, "rows": [], "error": "no_search_result"}
            cache_meta.write_text(json.dumps(meta, ensure_ascii=False))
            return meta
        time.sleep(1.0)
        html = fetch_tvpl(url, session)
        if not html:
            meta = {"name": name, "url": url, "rows": [], "error": "fetch_failed"}
            cache_meta.write_text(json.dumps(meta, ensure_ascii=False))
            return meta
        cache_html.write_text(html); time.sleep(1.0)

    text = extract_main_text(html)
    # Try to extract doc_id from URL
    m = re.search(r"-(\d+)-(\d{4})-([A-Za-z0-9\-]+?)-(?:cua|nam|so|muc|huong|quy|quan|ve|cho)?", url, re.I)
    # Better: extract from text header
    m_text = re.search(r"\b(\d+)\s*/\s*(\d{4})\s*/\s*([A-Za-zĐđ0-9\-]+)", text[:2000])
    if m_text:
        doc_id = f"{m_text.group(1)}/{m_text.group(2)}/{m_text.group(3).upper()}"
        law_sig = doc_id_to_law_sig(doc_id)
        parts = doc_id.split("/")
    else:
        # Synthesize a sig from URL slug
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", url.split("/")[-1].split(".")[0])[:40]
        law_sig = f"name_{slug}"
        parts = [None, None, slug]

    short_name = extract_short_name(text, fallback=name)
    arts = parse_articles(text)
    rows = []
    for art in arts:
        rows.append({
            "chunk_id": f"{law_sig}_{art['article_num']}",
            "law_sig": law_sig,
            "law_num": parts[0],
            "law_year": parts[1],
            "law_category": law_sig.split("_", 2)[-1],
            "law_short_name": short_name,
            "article_num": art["article_num"],
            "article_title": art["article_title"],
            "clause_nums": art["clause_nums"],
            "content": f"[{law_sig.upper()}]\n\n{short_name}\nĐiều {art['article_num']}. {art['article_title']}\n\n{art['article_text']}",
            "article_text": art["article_text"],
            "clauses_json": art["clauses_json"],
        })
    meta = {"name": name, "url": url, "law_sig": law_sig, "short_name": short_name,
            "n_articles": len(rows), "rows": rows, "error": None if rows else "no_articles_parsed"}
    cache_meta.write_text(json.dumps(meta, ensure_ascii=False))
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(INPUT_JSONL))
    p.add_argument("--output", default=str(OUT_PARQUET))
    p.add_argument("--min-count", type=int, default=2,
                   help="Only crawl docs with at least this many citations")
    p.add_argument("--max-docs", type=int, default=200)
    p.add_argument("--include-names", action="store_true",
                   help="Also crawl name-only citations (no doc_id)")
    args = p.parse_args()

    print(f"[crawl] Collecting missing docs from {args.input}")
    doc_ids, names = collect_missing(Path(args.input), min_count=args.min_count)
    print(f"[crawl]   {len(doc_ids)} unique doc_ids ({sum(doc_ids.values())} cites)")
    print(f"[crawl]   {len(names)} unique name-only ({sum(names.values())} cites)")

    targets_doc = list(doc_ids.most_common(args.max_docs))
    targets_name = list(names.most_common(args.max_docs)) if args.include_names else []

    session = requests.Session()
    all_rows: List[Dict] = []
    n_ok = 0
    n_fail = 0

    for doc_id, cnt in tqdm(targets_doc, desc="docs"):
        meta = crawl_doc(doc_id, session)
        if meta and meta.get("rows"):
            all_rows.extend(meta["rows"])
            n_ok += 1
        else:
            n_fail += 1

    for name, cnt in tqdm(targets_name, desc="names"):
        meta = crawl_name_only(name, session)
        if meta and meta.get("rows"):
            all_rows.extend(meta["rows"])
            n_ok += 1
        else:
            n_fail += 1

    print(f"\n[crawl] Success: {n_ok}, Failed: {n_fail}")
    print(f"[crawl] Total rows produced: {len(all_rows)}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_parquet(args.output, index=False)
        print(f"[crawl] Wrote {len(df)} rows -> {args.output}")
    else:
        print("[crawl] No rows produced")


if __name__ == "__main__":
    main()
