"""
Crawl missing law documents from thuvienphapluat.vn (TVPL) using Playwright.

TVPL hides full article body for non-logged-in HTTP clients (only first 5-7
articles visible in raw HTML). A real browser session sees the full text in
`#tab1` because TVPL injects content via JS / cookies. Playwright handles both.

Output: appends to data/processed/law_corpus_supplemental.parquet
"""

from __future__ import annotations
import json
import re
import time
import html
from html.parser import HTMLParser
from typing import Dict, List
import pandas as pd
from playwright.sync_api import sync_playwright

from ura_xlaw.config import PATHS

OUT = PATHS.processed / "law_corpus_supplemental.parquet"
CACHE_DIR = PATHS.raw / "tvpl_pw_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# (doc_id, short_name, law_sig, url)
MISSING_DOCS = [
    (
        "59/2024/QH15",
        "Luật Tư pháp người chưa thành niên 2024",
        "59_2024_qh15",
        "https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/Luat-Tu-phap-nguoi-chua-thanh-nien-2024-so-59-2024-QH15-499650.aspx",
    ),
    (
        "31/2024/QH15",
        "Luật Đất đai 2024",
        "31_2024_qh15",
        "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Luat-Dat-dai-2024-31-2024-QH15-501251.aspx",
    ),
    (
        "101/2024/NĐ-CP",
        "Nghị định 101/2024/NĐ-CP",
        "101_2024_nđ-cp",
        "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Nghi-dinh-101-2024-ND-CP-dieu-tra-co-ban-dat-dai-dang-ky-cap-Giay-chung-nhan-quyen-su-dung-dat-616797.aspx",
    ),
    (
        "123/2024/NĐ-CP",
        "Nghị định 123/2024/NĐ-CP",
        "123_2024_nđ-cp",
        "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Nghi-dinh-123-2024-ND-CP-xu-phat-vi-pham-hanh-chinh-linh-vuc-dat-dai-625443.aspx",
    ),
    (
        "86/2025/QH15",
        "Bộ luật Hình sự 2025 (86/2025/QH15)",
        "86_2025_qh15",
        "https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/Luat-sua-doi-Bo-luat-Hinh-su-2025-86-2025-QH15-657767.aspx",
    ),
    (
        "85/2025/QH15",
        "Luật sửa đổi Bộ luật Tố tụng Hình sự 2025",
        "85_2025_qh15",
        "https://thuvienphapluat.vn/van-ban/Thu-tuc-To-tung/Luat-sua-doi-Bo-luat-To-tung-Hinh-su-2025-85-2025-QH15-657766.aspx",
    ),
    (
        "76/2025/QH15",
        "Luật sửa đổi Bộ luật Hình sự 76/2025/QH15",
        "76_2025_qh15",
        "https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/Luat-sua-doi-Bo-luat-Hinh-su-76-2025-QH15-643040.aspx",
    ),
    (
        "36/2024/QH15",
        "Luật Trật tự an toàn giao thông đường bộ 2024",
        "36_2024_qh15",
        "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Luat-Trat-tu-an-toan-giao-thong-duong-bo-2024-36-2024-QH15-481106.aspx",
    ),
    (
        "01/2025/NQ-HĐTP",
        "Nghị quyết 01/2025/NQ-HĐTP",
        "01_2025_nq-hđtp",
        "https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Nghi-quyet-01-2025-NQ-HDTP-huong-dan-ap-dung-quy-dinh-cua-phap-luat-trong-xet-xu-vu-an-hinh-su-650044.aspx",
    ),
    (
        "04/2021/TTLT-BCA-BQP-TANDTC-VKSNDTC",
        "Thông tư liên tịch 04/2021/TTLT",
        "04_2021_ttlt-bca-bqp-tandtc-vksndtc",
        "https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/Thong-tu-lien-tich-04-2021-TTLT-BCA-BQP-TANDTC-VKSNDTC-phoi-hop-trong-thuc-hien-Bo-luat-To-tung-hinh-su-471895.aspx",
    ),
    (
        "54/2010/QH12",
        "Luật Trọng tài thương mại 2010",
        "54_2010_qh12",
        "https://thuvienphapluat.vn/van-ban/Thuong-mai/Luat-trong-tai-thuong-mai-2010-108083.aspx",
    ),
    (
        "69/2020/QH14",
        "Luật Người lao động VN đi làm việc ở nước ngoài 2020",
        "69_2020_qh14",
        "https://thuvienphapluat.vn/van-ban/Lao-dong-Tien-luong/Luat-Nguoi-lao-dong-Viet-Nam-di-lam-viec-o-nuoc-ngoai-theo-hop-dong-2020-432127.aspx",
    ),
    (
        "326/2016/NQ-UBTVQH14",
        "Nghị quyết 326/2016/UBTVQH14 án phí lệ phí",
        "326_2016_nq-ubtvqh14",
        "https://thuvienphapluat.vn/van-ban/Thue-Phi-Le-Phi/Nghi-quyet-326-2016-UBTVQH14-quy-dinh-ve-muc-thu-mien-giam-thu-nop-quan-ly-su-dung-an-phi-le-phi-Toa-an-336186.aspx",
    ),
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
        if tag in ("script", "style", "nav", "header", "footer"):
            self.skip += 1
        if tag == "div" and d.get("id") == "tab1":
            self.in_target = True
            self.depth = 0
            return
        if self.in_target and tag == "div":
            self.depth += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "header", "footer"):
            self.skip = max(0, self.skip - 1)
        if self.in_target and tag == "div":
            self.depth -= 1
            if self.depth < 0:
                self.in_target = False
        if tag in ("p", "div", "br", "tr", "li", "h1", "h2", "h3"):
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip == 0 and self.in_target:
            self.parts.append(data)


def extract_text(html_str: str) -> str:
    p = TextExtractor()
    try:
        p.feed(html_str)
    except Exception:
        pass
    text = "".join(p.parts)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_challenge_page(html_str: str) -> bool:
    return ("Chờ một chút" in html_str[:5000]) or len(html_str) < 60_000


def fetch_with_browser(page, url: str, attempts: int = 3) -> str:
    """Use Playwright page to fetch full HTML, with caching.

    Retries the navigation if TVPL serves the JS anti-bot challenge
    page ("Chờ một chút..."), waiting for it to resolve to the real page.
    """
    cache_key = re.sub(r"[^a-zA-Z0-9]+", "_", url)[-150:]
    cache_file = CACHE_DIR / f"{cache_key}.html"
    if cache_file.exists() and cache_file.stat().st_size > 200_000:
        body = cache_file.read_text(encoding="utf-8", errors="replace")
        if not _is_challenge_page(body):
            return body

    last_body = ""
    for attempt in range(attempts):
        print(f"    GOTO (attempt {attempt + 1}) {url}")
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        except Exception as e:
            print(f"      goto err: {e}")
            page.wait_for_timeout(2000)
            continue

        # Wait for the JS challenge ("Chờ một chút...") to resolve.
        # The challenge auto-redirects via JS to the real page.
        for _ in range(20):  # up to ~20 sec
            page.wait_for_timeout(1000)
            title = (page.title() or "").strip()
            if "Chờ một chút" not in title:
                break
        # Now wait for actual content selector
        try:
            page.wait_for_selector("#tab1", timeout=20_000)
        except Exception:
            pass
        # Trigger lazy load
        try:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight/2)")
            page.wait_for_timeout(1200)
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(1200)
        except Exception:
            pass
        # Dismiss popups
        for sel in [
            "button.close",
            ".modal.show button.close",
            "#popupLogin .close",
            "div.popup-overlay .close",
            "button[aria-label='Close']",
        ]:
            try:
                el = page.query_selector(sel)
                if el and el.is_visible():
                    el.click(timeout=1500)
                    page.wait_for_timeout(300)
            except Exception:
                pass

        body = page.content()
        last_body = body
        if not _is_challenge_page(body):
            cache_file.write_text(body, encoding="utf-8")
            return body
        print("      still on challenge page, retrying...")
        page.wait_for_timeout(3000)

    # Save whatever we got for debugging
    if last_body:
        cache_file.write_text(last_body, encoding="utf-8")
    return last_body


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
        full_body = rest + "\n" + body[len(rest) :] if body.startswith(rest) else body
        clause_iter = list(RE_CLAUSE_INLINE.finditer(full_body))
        for j, cm in enumerate(clause_iter):
            cn = cm.group("n")
            cs = cm.end()
            ce = (
                clause_iter[j + 1].start()
                if j + 1 < len(clause_iter)
                else len(full_body)
            )
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
                "clauses_json": json.dumps(clauses, ensure_ascii=False)
                if clauses
                else "{}",
                "clause_nums": clause_nums,
            }
    return list(out_by_num.values())


def main():
    # Determine which sigs are already well-covered. Skip those (we'd just be
    # overwriting hirine data with TVPL paywall previews).
    well_covered = set()
    for path in [
        PATHS.processed / "law_corpus.parquet",
        PATHS.processed / "law_corpus_supplemental.parquet",
    ]:
        if path.exists():
            df = pd.read_parquet(path, columns=["law_sig"])
            counts = df["law_sig"].value_counts()
            well_covered.update(counts[counts >= 10].index.astype(str).tolist())

    rows: List[Dict] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=[
                "--ignore-certificate-errors",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = browser.new_context(
            ignore_https_errors=True,
            user_agent=UA,
            viewport={"width": 1366, "height": 900},
            locale="vi-VN",
        )
        # Block heavy assets
        context.route(
            "**/*.{png,jpg,jpeg,gif,svg,woff,woff2,ttf,ico}",
            lambda route: route.abort(),
        )
        page = context.new_page()

        for doc_id, short_name, law_sig, url in MISSING_DOCS:
            print(f"\n[tvpl-pw] {doc_id} -> {law_sig}")
            if law_sig in well_covered:
                print(f"    SKIP: {law_sig} already well-covered in primary corpus")
                continue
            try:
                html_text = fetch_with_browser(page, url)
            except Exception as e:
                print(f"    ERR fetch: {e}")
                continue
            time.sleep(1.0)
            text = extract_text(html_text)
            if len(text) < 500:
                print(f"    SKIP: extracted text too short ({len(text)} chars)")
                continue
            arts = parse_articles(text)
            print(f"    extracted {len(arts)} articles ({len(text)} chars)")
            if not arts:
                continue
            # TVPL paywalls many laws and exposes only the first ~3 articles.
            # If we get fewer than 10 articles AND the doc is a substantive law
            # (Luật/Bộ luật), this is almost certainly a preview — skip rather
            # than pollute the alias map with a partial law_sig.
            if len(arts) < 10 and short_name.lower().startswith(
                ("luật", "bộ luật", "luat", "bo luat")
            ):
                print(f"    SKIP: only {len(arts)} articles — likely paywall preview")
                continue
            parts = doc_id.split("/")
            for art in arts:
                rows.append(
                    {
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
                    }
                )

        browser.close()

    if not rows:
        print("[tvpl-pw] Nothing extracted")
        return

    new_df = pd.DataFrame(rows)
    print(
        f"\n[tvpl-pw] Produced {len(new_df)} rows from {new_df['law_sig'].nunique()} laws"
    )

    if OUT.exists():
        existing = pd.read_parquet(OUT)
        # Determine which (sig, art) keys already exist in supplemental.
        existing_keys = set(
            existing["law_sig"].astype(str) + "||" + existing["article_num"].astype(str)
        )
        # Only insert TVPL rows for keys not already present, OR for which TVPL is
        # substantially longer (>=1.5x) than the existing chunk.
        existing["_key"] = (
            existing["law_sig"].astype(str) + "||" + existing["article_num"].astype(str)
        )
        existing["_len"] = existing["article_text"].astype(str).str.len()
        new_df["_key"] = (
            new_df["law_sig"].astype(str) + "||" + new_df["article_num"].astype(str)
        )
        new_df["_len"] = new_df["article_text"].astype(str).str.len()
        existing_len_by_key = dict(zip(existing["_key"], existing["_len"]))

        def keep_new(row):
            k = row["_key"]
            if k not in existing_keys:
                return True
            return row["_len"] >= 1.5 * existing_len_by_key.get(k, 0)

        keep_mask = new_df.apply(keep_new, axis=1)
        kept_new = new_df[keep_mask]
        # Drop those keys from existing (will be replaced by kept_new)
        replaced = set(kept_new["_key"])
        existing = existing[~existing["_key"].isin(replaced)]
        merged = pd.concat([existing, kept_new], ignore_index=True)
        merged = merged.drop(columns=["_key", "_len"]).reset_index(drop=True)
        print(f"[tvpl-pw] {len(kept_new)}/{len(new_df)} new rows kept after merge")
    else:
        merged = new_df
    merged.to_parquet(OUT, index=False)
    print(f"[tvpl-pw] Wrote {len(merged)} total rows -> {OUT}")


if __name__ == "__main__":
    main()
