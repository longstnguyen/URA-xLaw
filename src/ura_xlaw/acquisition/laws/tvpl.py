"""
Crawl missing law documents from thuvienphapluat.vn (TVPL).

For each entry in MISSING_DOCS, fetch the HTML page, extract the law text from
the #tab1 container, parse Điều/Khoản, and append to the supplemental corpus.

Output: appends to data/processed/law_corpus_supplemental.parquet
"""
from __future__ import annotations
import json
import re
import time
import html
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from ura_xlaw.config import PATHS

OUT = PATHS.processed / "law_corpus_supplemental.parquet"
CACHE_DIR = PATHS.raw / "tvpl_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# (doc_id, short_name, law_sig, url)
MISSING_DOCS = [
    # ---- Luật Tư pháp người chưa thành niên (59/2024/QH15) ----
    ("59/2024/QH15", "Luật Tư pháp người chưa thành niên 2024", "59_2024_qh15",
     "https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/Luat-Tu-phap-nguoi-chua-thanh-nien-2024-so-59-2024-QH15-499650.aspx"),
    # ---- Luật Đất đai 2024 ----
    ("31/2024/QH15", "Luật Đất đai 2024", "31_2024_qh15",
     "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Luat-Dat-dai-2024-31-2024-QH15-501251.aspx"),
    # ---- Nghị định 101/2024/NĐ-CP (HD Đất đai) ----
    ("101/2024/NĐ-CP", "Nghị định 101/2024/NĐ-CP", "101_2024_nđ-cp",
     "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Nghi-dinh-101-2024-ND-CP-dieu-tra-co-ban-dat-dai-dang-ky-cap-Giay-chung-nhan-quyen-su-dung-dat-616797.aspx"),
    # ---- Nghị định 123/2024/NĐ-CP ----
    ("123/2024/NĐ-CP", "Nghị định 123/2024/NĐ-CP", "123_2024_nđ-cp",
     "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Nghi-dinh-123-2024-ND-CP-xu-phat-vi-pham-hanh-chinh-linh-vuc-dat-dai-625443.aspx"),
    # ---- Bộ luật Hình sự (sửa đổi) 2025 - 86/2025/QH15 ----
    ("86/2025/QH15", "Luật sửa đổi Bộ luật Hình sự 2025", "86_2025_qh15",
     "https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/Luat-sua-doi-Bo-luat-Hinh-su-2025-86-2025-QH15-657767.aspx"),
    # ---- 85/2025/QH15 ----
    ("85/2025/QH15", "Luật sửa đổi Bộ luật Tố tụng Hình sự 2025", "85_2025_qh15",
     "https://thuvienphapluat.vn/van-ban/Thu-tuc-To-tung/Luat-sua-doi-Bo-luat-To-tung-Hinh-su-2025-85-2025-QH15-657766.aspx"),
    # ---- 76/2025/QH15 ----
    ("76/2025/QH15", "Luật sửa đổi Bộ luật Hình sự 2025 (76/2025)", "76_2025_qh15",
     "https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/Luat-sua-doi-Bo-luat-Hinh-su-76-2025-QH15-643040.aspx"),
    # ---- Luật Trật tự an toàn giao thông 2024 ----
    ("36/2024/QH15", "Luật Trật tự, an toàn giao thông đường bộ 2024", "36_2024_qh15",
     "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Luat-Trat-tu-an-toan-giao-thong-duong-bo-2024-36-2024-QH15-481106.aspx"),
    # ---- Nghị quyết 01/2025/NQ-HĐTP ----
    ("01/2025/NQ-HĐTP", "Nghị quyết 01/2025/NQ-HĐTP", "01_2025_nq-hđtp",
     "https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Nghi-quyet-01-2025-NQ-HDTP-huong-dan-ap-dung-quy-dinh-cua-phap-luat-trong-xet-xu-vu-an-hinh-su-650044.aspx"),
    # ---- TTLT 04/2021/TTLT-BCA-BQP-TANDTC-VKSNDTC ----
    ("04/2021/TTLT-BCA-BQP-TANDTC-VKSNDTC", "Thông tư liên tịch 04/2021", "04_2021_ttlt-bca-bqp-tandtc-vksndtc",
     "https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/Thong-tu-lien-tich-04-2021-TTLT-BCA-BQP-TANDTC-VKSNDTC-phoi-hop-trong-thuc-hien-Bo-luat-To-tung-hinh-su-471895.aspx"),
    # ---- Luật Trọng tài thương mại 2010 (full version) ----
    ("54/2010/QH12", "Luật Trọng tài thương mại 2010", "54_2010_qh12",
     "https://thuvienphapluat.vn/van-ban/Thuong-mai/Luat-trong-tai-thuong-mai-2010-108083.aspx"),
    # ---- Luật Người Việt Nam đi làm việc ở nước ngoài 132/2020/QH14 ----
    ("69/2020/QH14", "Luật Người lao động Việt Nam đi làm việc ở nước ngoài 2020", "69_2020_qh14",
     "https://thuvienphapluat.vn/van-ban/Lao-dong-Tien-luong/Luat-Nguoi-lao-dong-Viet-Nam-di-lam-viec-o-nuoc-ngoai-theo-hop-dong-2020-432127.aspx"),
]


# --- HTML extractor: get text from #tab1 div --- #
class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self.skip = 0
        self.in_target = False
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag in ('script', 'style', 'nav', 'header', 'footer'):
            self.skip += 1
        if tag == 'div' and d.get('id') == 'tab1':
            self.in_target = True
            self.depth = 0
            return
        if self.in_target and tag == 'div':
            self.depth += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'nav', 'header', 'footer'):
            self.skip = max(0, self.skip - 1)
        if self.in_target and tag == 'div':
            self.depth -= 1
            if self.depth < 0:
                self.in_target = False
        if tag in ('p', 'div', 'br', 'tr', 'li', 'h1', 'h2', 'h3'):
            self.parts.append('\n')

    def handle_data(self, data):
        if self.skip == 0 and self.in_target:
            self.parts.append(data)


def extract_text(html_str: str) -> str:
    p = TextExtractor()
    try:
        p.feed(html_str)
    except Exception:
        pass
    text = ''.join(p.parts)
    text = html.unescape(text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def fetch(url: str) -> str:
    """Fetch URL with caching, return HTML text."""
    cache_key = re.sub(r'[^a-zA-Z0-9]+', '_', url)[-150:]
    cache_file = CACHE_DIR / f"{cache_key}.html"
    if cache_file.exists():
        return cache_file.read_text(encoding='utf-8', errors='replace')
    print(f"    GET {url}")
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "vi,en;q=0.9",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode('utf-8', errors='replace')
    cache_file.write_text(body, encoding='utf-8')
    return body


# --- Article parser --- #
RE_ARTICLE = re.compile(
    r"(?<![\w\.])Điều\s+(?P<num>\d+[a-z]?)\.\s*(?P<rest>[^\n]{0,200})",
    re.UNICODE,
)
RE_CLAUSE_INLINE = re.compile(r"(?:^|\n|\.\s+)(?P<n>\d{1,2})\s*\.\s+", re.UNICODE)


def parse_articles(text: str) -> List[Dict]:
    matches = list(RE_ARTICLE.finditer(text))
    if not matches:
        return []

    # Many TVPL pages include a "related documents" section that has duplicate/extra
    # Điều listings. We take the largest contiguous run starting from Điều 1 with
    # increasing numbers. To keep it simple, just keep all occurrences but de-dup
    # by article_num (keep the longest body per number).
    out_by_num: Dict[str, Dict] = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body) < 30:
            continue
        rest = m.group("rest").strip()
        title = re.split(r"\s+\d+\.\s", rest, 1)[0][:200].strip()

        # Clause extraction
        clauses: Dict[str, str] = {}
        clause_nums: List[int] = []
        full_body = rest + "\n" + body[len(rest):] if body.startswith(rest) else body
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
    for doc_id, short_name, law_sig, url in MISSING_DOCS:
        print(f"\n[tvpl] {doc_id} -> {law_sig}")
        try:
            html_text = fetch(url)
        except Exception as e:
            print(f"    ERR fetch: {e}")
            continue
        time.sleep(1.0)  # be polite
        text = extract_text(html_text)
        if len(text) < 500:
            print(f"    SKIP: extracted text too short ({len(text)} chars)")
            continue
        arts = parse_articles(text)
        print(f"    extracted {len(arts)} articles ({len(text)} chars)")
        if not arts:
            continue
        parts = doc_id.split("/")
        for art in arts:
            rows.append({
                "chunk_id": f"supp_tvpl_{law_sig}_{art['article_num']}",
                "law_sig": law_sig,
                "law_num": parts[0],
                "law_year": parts[1],
                "law_category": law_sig.split("_", 2)[2],
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
        print("[tvpl] Nothing extracted")
        return

    new_df = pd.DataFrame(rows)
    print(f"\n[tvpl] Produced {len(new_df)} rows from {new_df['law_sig'].nunique()} laws")

    # Merge with existing supp parquet (replace any sigs we just refreshed)
    if OUT.exists():
        existing = pd.read_parquet(OUT)
        sigs_to_replace = set(new_df["law_sig"].unique())
        existing = existing[~existing["law_sig"].isin(sigs_to_replace)]
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        merged = new_df
    merged.to_parquet(OUT, index=False)
    print(f"[tvpl] Wrote {len(merged)} total rows -> {OUT}")


if __name__ == "__main__":
    main()
