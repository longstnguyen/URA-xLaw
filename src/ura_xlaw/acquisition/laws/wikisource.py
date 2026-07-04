"""
Crawl missing law documents from vi.wikisource.org.

Wikisource hosts full, unpaywalled text of many Vietnamese laws, organised
as a top-level page + one sub-page per chapter. We use the MediaWiki API
to list sub-pages and fetch parsed HTML, then extract Điều N. blocks the
same way as crawl_tvpl_playwright.

Output: appends to data/processed/law_corpus_supplemental.parquet
"""
from __future__ import annotations
import json
import re
import time
import html
import urllib.request
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List
import pandas as pd

from ura_xlaw.config import PATHS

OUT = PATHS.processed / "law_corpus_supplemental.parquet"
CACHE_DIR = PATHS.raw / "wikisource_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 lawcrawl/1.0"
API = "https://vi.wikisource.org/w/api.php"

# (doc_id, short_name, law_sig, wikisource_root_title)
WIKI_DOCS = [
    ("31/2024/QH15", "Luật Đất đai 2024", "31_2024_qh15",
     "Luật Đất đai nước Cộng hòa xã hội chủ nghĩa Việt Nam 2024"),
    ("36/2024/QH15", "Luật Trật tự, an toàn giao thông đường bộ 2024", "36_2024_qh15",
     "Luật Trật tự, an toàn giao thông đường bộ nước Cộng hòa xã hội chủ nghĩa Việt Nam 2024"),
    ("59/2024/QH15", "Luật Tư pháp người chưa thành niên 2024", "59_2024_qh15",
     "Luật Tư pháp người chưa thành niên nước Cộng hòa xã hội chủ nghĩa Việt Nam 2024"),
    ("100/2015/QH13", "Bộ luật Hình sự 2015 (sửa đổi 2017)", "100_2015_qh13",
     "Bộ luật Hình sự nước Cộng hòa xã hội chủ nghĩa Việt Nam 2015 (sửa đổi, bổ sung 2017)"),
]


def http_get(url: str) -> bytes:
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            wait = 5.0 * (attempt + 1) if e.code == 429 else 2.0 * (attempt + 1)
            print(f"      http {e.code} (try {attempt+1}), sleeping {wait:.0f}s")
            time.sleep(wait)
        except Exception as e:
            print(f"      http err (try {attempt+1}): {e}")
            time.sleep(2.0 * (attempt + 1))
    return b""


def list_subpages(root_title: str) -> List[str]:
    """Use action=query&list=allpages with apprefix to enumerate sub-pages."""
    out: List[str] = []
    cont = ""
    while True:
        url = (f"{API}?action=query&list=allpages"
               f"&apprefix={urllib.parse.quote(root_title + '/')}"
               f"&aplimit=500&format=json{cont}")
        data = json.loads(http_get(url))
        for p in data.get("query", {}).get("allpages", []):
            out.append(p["title"])
        c = data.get("continue", {}).get("apcontinue")
        if not c:
            break
        cont = "&apcontinue=" + urllib.parse.quote(c)
        time.sleep(0.5)
    return out


def fetch_page_html(title: str) -> str:
    cache = CACHE_DIR / (re.sub(r"[^a-zA-Z0-9]+", "_", title)[-180:] + ".html")
    if cache.exists() and cache.stat().st_size > 1000:
        return cache.read_text("utf-8", errors="replace")
    url = (f"{API}?action=parse&page={urllib.parse.quote(title)}"
           f"&prop=text&formatversion=2&format=json")
    data = json.loads(http_get(url))
    text = data.get("parse", {}).get("text", "")
    if isinstance(text, dict):
        text = text.get("*", "")
    if text:
        cache.write_text(text, "utf-8")
    return text


# --- HTML extractor (full body text) ---
class TextExtractor(HTMLParser):
    """Extract text, skipping subtrees whose root tag is noisy.

    Tracks a stack of skip-causing tags so closing them decrements correctly.
    """
    SKIP_TAGS = {"script", "style", "table"}

    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self.skip_stack: List[str] = []  # tag names that opened a skip region

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        cls = d.get("class", "") or ""
        skip_cls = ("noprint" in cls) or ("toc" in cls.split()) or ("dynlayout-exempt" in cls)
        if tag in self.SKIP_TAGS or skip_cls:
            self.skip_stack.append(tag)

    def handle_endtag(self, tag):
        if self.skip_stack and self.skip_stack[-1] == tag:
            self.skip_stack.pop()
        if tag in ("p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip_stack:
            self.parts.append(data)


def extract_text(html_str: str) -> str:
    p = TextExtractor()
    try:
        p.feed(html_str)
    except Exception:
        pass
    text = html.unescape("".join(p.parts))
    text = re.sub(r"\[\d+\]", " ", text)  # footnote markers
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


RE_ARTICLE = re.compile(
    r"(?<![\w\.])Điều\s+(?P<num>\d+[a-z]?)\.\s*(?P<rest>[^\n]{0,200})",
    re.UNICODE,
)
RE_CLAUSE_INLINE = re.compile(r"(?:^|\n|\.\s+)(?P<n>\d{1,2})\s*\.\s+", re.UNICODE)


def parse_articles(text: str) -> List[Dict]:
    matches = list(RE_ARTICLE.finditer(text))
    if not matches:
        return []
    out_by_num: Dict[str, Dict] = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body) < 30:
            continue
        rest = m.group("rest").strip()
        title = re.split(r"\s+\d+\.\s", rest, 1)[0][:200].strip()
        clauses: Dict[str, str] = {}
        clause_nums: List[int] = []
        full_body = body
        clause_iter = list(RE_CLAUSE_INLINE.finditer(full_body))
        for j, cm in enumerate(clause_iter):
            cn = cm.group("n")
            cs = cm.end()
            ce = clause_iter[j + 1].start() if j + 1 < len(clause_iter) else len(full_body)
            ctext = full_body[cs:ce].strip()
            if len(ctext) >= 5:
                clauses[cn] = ctext
                try:
                    clause_nums.append(int(cn))
                except ValueError:
                    pass
        existing = out_by_num.get(m.group("num"))
        if existing is None or len(body) > len(existing["article_text"]):
            out_by_num[m.group("num")] = {
                "article_num": m.group("num"),
                "article_title": title,
                "article_text": body,
                "clauses_json": json.dumps(clauses, ensure_ascii=False) if clauses else "{}",
                "clause_nums": clause_nums,
            }
    return list(out_by_num.values())


def main():
    rows: List[Dict] = []
    for doc_id, short_name, law_sig, root in WIKI_DOCS:
        print(f"\n[wiki] {doc_id} -> {law_sig}")
        subs = list_subpages(root)
        # Take ALL sub-pages (chapters, sections, parts). parse_articles
        # de-duplicates by article number, keeping the longest body.
        pages = subs or [root]
        print(f"    {len(pages)} sub-pages to fetch")

        full_text_parts: List[str] = []
        for title in pages:
            try:
                page_html = fetch_page_html(title)
            except Exception as e:
                print(f"    ERR fetch {title}: {e}")
                continue
            time.sleep(1.5)
            txt = extract_text(page_html)
            full_text_parts.append(txt)
        full_text = "\n\n".join(full_text_parts)
        arts = parse_articles(full_text)
        print(f"    extracted {len(arts)} articles ({len(full_text)} chars)")
        if not arts:
            continue
        parts = doc_id.split("/")
        for art in arts:
            rows.append({
                "chunk_id": f"supp_wiki_{law_sig}_{art['article_num']}",
                "law_sig": law_sig,
                "law_num": parts[0],
                "law_year": parts[1] if len(parts) > 1 else "",
                "law_category": law_sig.split("_", 2)[2] if law_sig.count("_") >= 2 else "",
                "law_short_name": short_name,
                "article_num": art["article_num"],
                "article_title": art["article_title"],
                "clause_nums": art["clause_nums"],
                "content": (
                    f"[{short_name.upper()}]\n"
                    f"Điều {art['article_num']}. {art['article_title']}\n\n"
                    f"{art['article_text']}"
                ),
                "article_text": art["article_text"],
                "clauses_json": art["clauses_json"],
            })

    if not rows:
        print("[wiki] Nothing extracted")
        return
    new_df = pd.DataFrame(rows)
    print(f"\n[wiki] Produced {len(new_df)} rows from {new_df['law_sig'].nunique()} laws")

    if OUT.exists():
        existing = pd.read_parquet(OUT)
        existing["_key"] = existing["law_sig"].astype(str) + "||" + existing["article_num"].astype(str)
        existing["_len"] = existing["article_text"].astype(str).str.len()
        new_df["_key"] = new_df["law_sig"].astype(str) + "||" + new_df["article_num"].astype(str)
        new_df["_len"] = new_df["article_text"].astype(str).str.len()
        existing_keys = set(existing["_key"])
        existing_len_by_key = dict(zip(existing["_key"], existing["_len"]))

        def keep_new(row):
            k = row["_key"]
            if k not in existing_keys:
                return True
            return row["_len"] >= 1.5 * existing_len_by_key.get(k, 0)

        mask = new_df.apply(keep_new, axis=1)
        kept = new_df[mask]
        existing = existing[~existing["_key"].isin(set(kept["_key"]))]
        merged = pd.concat([existing, kept], ignore_index=True)
        merged = merged.drop(columns=["_key", "_len"]).reset_index(drop=True)
        print(f"[wiki] {len(kept)}/{len(new_df)} new rows kept after merge")
    else:
        merged = new_df
    merged.to_parquet(OUT, index=False)
    print(f"[wiki] Wrote {len(merged)} total rows -> {OUT}")


if __name__ == "__main__":
    main()
