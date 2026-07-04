"""
Crawler for congbobanan.toaan.gov.vn (Cổng công bố bản án Tòa án nhân dân Việt Nam)

Uses Playwright (headless browser) to handle WAF protection and ASP.NET forms,
plus PyMuPDF for PDF text extraction.
"""

from __future__ import annotations

import json
import time
import os
import re
import argparse
import logging
from typing import Optional
from io import BytesIO

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    print("WARNING: PyMuPDF not installed. Run: pip install PyMuPDF")

try:
    from playwright.sync_api import sync_playwright, Page, Browser
except ImportError:  # Loaded lazily so the rest of URA-xLaw works without crawl extras.
    sync_playwright = None
    Page = Browser = object
from tqdm import tqdm

from ura_xlaw.config import PATHS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class CongBoBanAnCrawler:
    """Crawler using Playwright headless browser."""

    BASE_URL = "https://congbobanan.toaan.gov.vn"
    SEARCH_URL = f"{BASE_URL}/0tat1cvn/ban-an-quyet-dinh"
    DETAIL_URL_TPL = f"{BASE_URL}/2ta{{doc_id}}t1cvn/chi-tiet-ban-an"
    PDF_URL_TPL = f"{BASE_URL}/3ta{{doc_id}}t1cvn/"

    CASE_TYPES = {
        "all": "",
        "hinh_su": "50",
        "dan_su": "51",
        "hon_nhan": "52",
        "hanh_chinh": "53",
        "kinh_doanh": "54",
        "lao_dong": "55",
    }

    CASE_LEVELS = {
        "all": "",
        "so_tham": "1",
        "phuc_tham": "2",
        "giam_doc_tham": "3",
        "tai_tham": "4",
    }

    DOC_TYPES = {
        "all": "",
        "ban_an": "1",
        "quyet_dinh": "2",
    }

    # Known recent document ID for fallback probing
    FALLBACK_RECENT_ID = 2086589

    def __init__(self, data_dir: str = str(PATHS.raw_judgments), delay: float = 2.0):
        self.data_dir = data_dir
        self.delay = delay
        os.makedirs(data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Auto-discover the most recent document ID from homepage links.
    # ------------------------------------------------------------------
    def discover_latest_id(self) -> int:
        """Scrape homepage / search page to find the largest visible doc_id.

        Returns the largest matched ID, or FALLBACK_RECENT_ID if nothing found.
        """
        log.info("Discovering latest document ID …")
        candidates: set[int] = set()
        for url in (self.BASE_URL, self.SEARCH_URL):
            if not self._navigate(url):
                continue
            time.sleep(2)
            self._dismiss_modal()
            self._dismiss_notifications()
            try:
                html = self._page.content()
                for m in re.finditer(r"/2ta(\d+)t1cvn/", html):
                    candidates.add(int(m.group(1)))
            except Exception as exc:
                log.warning("Could not read content from %s: %s", url, exc)
            if candidates:
                break

        if candidates:
            latest = max(candidates)
            log.info(
                "Discovered latest doc_id: %d (from %d candidates)",
                latest,
                len(candidates),
            )
            return latest

        log.warning(
            "Could not discover latest ID, using fallback %d", self.FALLBACK_RECENT_ID
        )
        return self.FALLBACK_RECENT_ID

        self._pw = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None

    # ------------------------------------------------------------------
    # Browser lifecycle
    # ------------------------------------------------------------------

    def _start_browser(self):
        if sync_playwright is None:
            raise ImportError(
                "Playwright is required for crawling. Install URA-xLaw with "
                "the 'crawling' extra and run: playwright install chromium"
            )
        """Launch headless Chromium."""
        log.info("Starting headless browser …")
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=True,
            args=["--ignore-certificate-errors"],
        )
        context = self._browser.new_context(
            ignore_https_errors=True,
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        self._page = context.new_page()
        # Block images/fonts to speed things up
        self._page.route(
            "**/*.{png,jpg,jpeg,gif,svg,woff,woff2,ttf}", lambda route: route.abort()
        )

    def _stop_browser(self):
        """Close browser."""
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()
        log.info("Browser closed.")

    def _screenshot(self, name: str):
        """Take a debug screenshot."""
        path = os.path.join(self.data_dir, f"debug_{name}.png")
        try:
            self._page.screenshot(path=path)
            log.info("Screenshot saved: %s", path)
        except Exception as exc:
            log.warning("Screenshot failed: %s", exc)

    def _navigate(self, url: str, wait_until: str = "load", retries: int = 3) -> bool:
        """Navigate with retry logic. Returns True if page loaded without server error."""
        for attempt in range(retries):
            try:
                response = self._page.goto(url, wait_until=wait_until, timeout=60000)
                if response and response.status >= 500:
                    log.warning(
                        "%s returned status %d (attempt %d)",
                        url,
                        response.status,
                        attempt + 1,
                    )
                    time.sleep(5 * (attempt + 1))
                    continue
                return True
            except Exception as exc:
                log.warning(
                    "Navigate to %s attempt %d failed: %s", url, attempt + 1, exc
                )
                time.sleep(5 * (attempt + 1))
        return False

    def _dismiss_modal(self):
        """Dismiss the feedback modal if it appears."""
        try:
            # Wait briefly for modal to potentially appear
            self._page.wait_for_timeout(1500)

            # Try multiple known modal selectors
            modal_selectors = [
                "#ctl00_Feedback_Home_pnl_Model_Popup",
                ".modal.show",
                "[id*='Feedback'][id*='Popup']",
                ".modal-dialog",
            ]

            for selector in modal_selectors:
                modal = self._page.query_selector(selector)
                if modal and modal.is_visible():
                    log.debug("Found modal: %s — dismissing …", selector)

                    # Try clicking a radio button ("Khác" = value 6)
                    radio_selectors = [
                        "input[type='radio'][value='6']",
                        "input[type='radio'][value='5']",
                        "input[type='radio']:last-of-type",
                    ]
                    for rs in radio_selectors:
                        try:
                            self._page.click(rs, timeout=2000)
                            break
                        except Exception:
                            continue

                    time.sleep(0.5)

                    # Try clicking confirm button
                    confirm_selectors = [
                        "#ctl00_Feedback_Home_cmdSave_Regis",
                        "input[value='Xác nhận']",
                        "button:has-text('Xác nhận')",
                        ".btn-primary",
                    ]
                    for cs in confirm_selectors:
                        try:
                            self._page.click(cs, timeout=2000)
                            log.debug("Modal dismissed via %s", cs)
                            time.sleep(1)
                            return
                        except Exception:
                            continue

                    # As fallback, try pressing Escape
                    self._page.keyboard.press("Escape")
                    time.sleep(0.5)
                    return

        except Exception as exc:
            log.debug("Modal check: %s", exc)

    def _dismiss_notifications(self):
        """Dismiss browser notification popups (SendPulse etc)."""
        try:
            # Wait a bit for potential popups
            self._page.wait_for_timeout(1000)

            # 1. Try clicking the "Don't allow" or "X" button on SendPulse notification
            # These are often in a shadow DOM or very specific classes
            for sel in [
                "text=Don't allow",
                "text=Từ chối",
                ".sp-push-prompt-close",
                ".sp-push-notification-deny",
                "[class*='close']",
                "#sp-push-deny",
            ]:
                try:
                    btn = self._page.query_selector(sel)
                    if btn and btn.is_visible():
                        btn.click(timeout=1000)
                        log.debug("Dismissed notification/popup via selector: %s", sel)
                        time.sleep(0.5)
                except Exception:
                    continue

            # 2. Try closing any generic modal by clicking outside or pressing Escape
            self._page.keyboard.press("Escape")
            time.sleep(0.3)
        except Exception as exc:
            log.debug("Notification dismissal warning: %s", exc)

    # ------------------------------------------------------------------
    # Phase 1: Search & get listing
    # ------------------------------------------------------------------

    def get_listing(
        self,
        case_type: str = "all",
        case_level: str = "all",
        doc_type: str = "all",
        max_pages: int = 5,
    ) -> list[dict]:
        """Use browser to search and get document listing with filters."""
        # Step 1: Visit homepage first to establish session cookies
        log.info("Visiting homepage to establish session …")
        if not self._navigate(self.BASE_URL):
            # Try without checking status
            log.warning("Homepage returned error, trying search page directly …")

        time.sleep(3)
        self._dismiss_modal()
        time.sleep(1)

        # Step 2: Navigate to search page
        log.info("Opening search page …")
        if not self._navigate(self.SEARCH_URL):
            # Page returned 500 — try one more time after a longer delay
            log.warning("Search page returned error, retrying after 10s …")
            time.sleep(10)
            if not self._navigate(self.SEARCH_URL):
                log.error("Cannot reach search page after retries")
                self._screenshot("search_page_error")
                return []

        time.sleep(3)
        self._dismiss_modal()
        self._dismiss_notifications()
        time.sleep(1)

        self._screenshot("after_modal")

        # 1. Select case type
        case_val = self.CASE_TYPES.get(case_type, "")
        if case_val:
            dropdown_selectors = [
                "#ctl00_Content_home_Public_ctl00_Drop_CASES_STYLES_SEARCH_top",
                "#ctl00_Content_home_Public_ctl00_Drop_CASES_STYLES_SEARCH",
            ]
            for sel in dropdown_selectors:
                try:
                    self._page.select_option(sel, case_val, timeout=2000)
                    log.info("Selected case type: %s", case_type)
                    break
                except Exception:
                    continue

        # 1.1 Select case level (Appellate/Cassation etc)
        level_val = self.CASE_LEVELS.get(case_level, "")
        if level_val:
            level_selectors = [
                "#ctl00_Content_home_Public_ctl00_Drop_LEVEL_JUDGMENT_SEARCH_top",
                "#ctl00_Content_home_Public_ctl00_Drop_LEVEL_JUDGMENT_SEARCH",
            ]
            for sel in level_selectors:
                try:
                    self._page.select_option(sel, level_val, timeout=2000)
                    log.info("Selected case level: %s", case_level)
                    break
                except Exception:
                    continue

        # 1.2 Select document type (Judgment/Decision)
        type_val = self.DOC_TYPES.get(doc_type, "")
        if type_val:
            type_selectors = [
                "#ctl00_Content_home_Public_ctl00_Drop_STATUS_JUDGMENT_SEARCH_top",
                "#ctl00_Content_home_Public_ctl00_Drop_STATUS_JUDGMENT_SEARCH",
            ]
            for sel in type_selectors:
                try:
                    self._page.select_option(sel, type_val, timeout=2000)
                    log.info("Selected doc type: %s", doc_type)
                    break
                except Exception:
                    continue

        # 2. Try to fill a dummy keyword to wake up the form
        try:
            kw_box = "#ctl00_Content_home_Public_ctl00_txtKeyword_top"
            self._page.fill(kw_box, "")
            self._page.focus(kw_box)
        except Exception:
            pass

        # 3. Dismiss notifications
        self._dismiss_notifications()
        time.sleep(1)

        # 4. Try to submit via Enter key first (often more reliable in ASP.NET)
        log.info("Attempting search via Enter key …")
        try:
            self._page.keyboard.press("Enter")
            time.sleep(5)
        except Exception:
            pass

        # 5. Fallback to robust click if no results yet
        if not self._page.query_selector("a.echo_id_pub"):
            log.info("No results from Enter, trying Click …")
            search_btn_selectors = [
                "input[value='Tìm kiếm']",
                "#ctl00_Content_home_Public_ctl00_cmd_search_banner",
            ]
            for sel in search_btn_selectors:
                try:
                    el = self._page.query_selector(sel)
                    if el and el.is_visible():
                        el.click(force=True, timeout=5000)
                        log.info("Clicked search button: %s", sel)
                        time.sleep(5)
                        break
                except Exception:
                    continue

        # Wait for results area
        try:
            self._page.wait_for_selector(
                "a.echo_id_pub, text=Kết quả tìm kiếm", timeout=10000
            )
        except Exception:
            pass

        time.sleep(2)
        self._screenshot("after_search_final")

        # Scroll down to ensure all lazy elements load
        self._page.evaluate("window.scrollBy(0, 800)")
        time.sleep(1)

        all_docs: list[dict] = []

        for page_num in range(1, max_pages + 1):
            docs = self._parse_listing_page()
            if not docs:
                log.info("No results on page %d, stopping", page_num)
                break

            all_docs.extend(docs)
            log.info(
                "Page %d: found %d documents (total: %d)",
                page_num,
                len(docs),
                len(all_docs),
            )

            # Go to next page
            if page_num < max_pages:
                if not self._goto_next_page():
                    break
                time.sleep(2)

        return all_docs

    def _parse_listing_page(self) -> list[dict]:
        """Extract document stubs from current listing page."""
        results = []
        try:
            links = self._page.query_selector_all("a.echo_id_pub")
            for link in links:
                href = link.get_attribute("href") or ""
                match = re.search(r"/2ta(\d+)t1cvn/", href)
                if not match:
                    continue

                doc_id = match.group(1)
                # Get title from h4 tag inside the link
                h4 = link.query_selector("h4")
                title = h4.inner_text().strip() if h4 else ""
                title = re.sub(r"\s+", " ", title)

                results.append(
                    {
                        "doc_id": doc_id,
                        "title": title,
                    }
                )
        except Exception as exc:
            log.error("Error parsing listing: %s", exc)

        return results

    def _goto_next_page(self) -> bool:
        """Navigate to next page of search results."""
        try:
            # Use the exact Next button ID from DOM analysis
            next_selectors = [
                "#ctl00_Content_home_Public_ctl00_LinkButton_Next",
                "a[id*='LinkButton_Next']",
                "a[title='Next']",
            ]
            for sel in next_selectors:
                btn = self._page.query_selector(sel)
                if btn:
                    btn.click()
                    time.sleep(3)
                    self._page.wait_for_timeout(2000)
                    return True
        except Exception as exc:
            log.warning("Could not navigate to next page: %s", exc)

        return False

    # ------------------------------------------------------------------
    # Phase 2: Detail page metadata
    # ------------------------------------------------------------------

    def get_detail_metadata(self, doc_id: str, fast_mode: bool = False) -> dict:
        """Navigate to detail page and extract all metadata using hybrid logic.

        fast_mode: skip the 1.5s wait + modal/notification dismissal. Use for
        scan-only workflows where we never click anything (just read text).
        """
        url = (
            f"{self.BASE_URL}/2ta{doc_id}t1cvn/chi-tiet-ban-an"
            if "toaan.gov.vn" not in doc_id
            else doc_id
        )
        if not self._navigate(url):
            return {}

        # 1. Quick wait and cleanup
        if fast_mode:
            time.sleep(0.3)
        else:
            time.sleep(1.5)
            self._dismiss_modal()
            self._dismiss_notifications()

        try:
            # 2. Extract full page text for regex-based parsing (most robust)
            full_text = self._page.inner_text("body")

            meta: dict = {"doc_id": doc_id, "url": url}

            # Regex patterns for Vietnamese fields
            patterns = {
                "case_number": r"(?:Số bản án|Số quyết định|Số|Quyết định số|Bản án số)\s*:\s*([^\n\r]+)",
                "date": r"(?:Ngày ban hành|Ngày quyết định|Ngày)\s*:\s*([^\n\r]+)",
                "title": r"(?:Tên bản án|Tên quyết định|Tên|Nguyên đơn|Bị đơn|Đối tượng)\s*:\s*([^\n\r]+)",
                "court": r"(?:Tòa án xét xử|Tòa án)\s*:\s*([^\n\r]+)",
                "trial_level": r"(?:Cấp xét xử)\s*:\s*([^\n\r]+)",
                "case_type": r"(?:Loại án|Loại vụ/việc)\s*:\s*([^\n\r]+)",
                "judge": r"(?:Thẩm phán|Chủ tọa)\s*:\s*([^\n\r]+)",
                "legal_relation": r"(?:Quan hệ pháp luật|Vụ án về)\s*:\s*([^\n\r]+)",
                "precedent_applied": r"(?:Áp dụng án lệ|Án lệ áp dụng|Án lệ)\s*:\s*([^\n\r]+)",
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, full_text, re.IGNORECASE)
                if match:
                    meta[key] = match.group(1).strip()

            # 3. Strategy 2: If title or essential fields missing, try specific selectors
            if "title" not in meta or len(meta.get("title", "")) < 3:
                try:
                    # Look for h4 or elements with title-like text
                    title_el = self._page.query_selector(
                        "h4, .title-detail, .title-label"
                    )
                    if title_el:
                        meta["title"] = title_el.inner_text().strip()
                except Exception:
                    pass

            # 4. Strategy 3: Identify PDF URL
            try:
                # Look for links containing 3ta or 5ta or download attributes
                links = self._page.query_selector_all("a[href*='ta'], a[href$='.pdf']")
                for link in links:
                    href = link.get_attribute("href")
                    if href and ("3ta" in href or "5ta" in href):
                        meta["found_pdf_url"] = (
                            href
                            if href.startswith("http")
                            else f"{self.BASE_URL}{href}"
                        )
                        break
            except Exception:
                pass

            # Minimum check: if we have no title and no case number, it's a fail
            if not meta.get("title") and not meta.get("case_number"):
                # One last try check if page content is just "không tìm thấy"
                if "không tìm thấy" in full_text.lower():
                    log.debug("Doc %s: Not found on portal", doc_id)
                return {}

            return meta
        except Exception as exc:
            log.warning("Final metadata extraction failed for doc %s: %s", doc_id, exc)
            return {"doc_id": doc_id}

            # Look for PDF link (5ta or 3ta)
            try:
                pdf_link = self._page.query_selector("a[href*='/5ta'], a[href*='/3ta']")
                if pdf_link:
                    meta["found_pdf_url"] = (
                        f"{self.BASE_URL}{pdf_link.get_attribute('href')}"
                    )
            except Exception:
                pass

            return meta
        except Exception as exc:
            log.warning("Metadata extraction failed for doc %s: %s", doc_id, exc)
            return {"doc_id": doc_id}

    # ------------------------------------------------------------------
    # PDF-based metadata extraction (more reliable than HTML page)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_metadata_from_pdf_text(text: str) -> dict:
        """Extract case_number, date, court, case_type from PDF text.

        Vietnamese court documents have a fairly stable header layout:
            TÒA ÁN NHÂN DÂN <COURT_NAME>
            ...
            Số: <case_number>
            <Place>, ngày <DD> tháng <MM> năm <YYYY>
            QUYẾT ĐỊNH / BẢN ÁN ...
        """
        meta: dict = {}
        if not text:
            return meta

        # Court name: lines starting with "TÒA ÁN" until blank line.
        court_match = re.search(
            r"(T[ÒO]A\s*[ÁA]N[^\n]*(?:\n[^\n]*){0,3})",
            text[:1500],
        )
        if court_match:
            court = re.sub(r"\s+", " ", court_match.group(1)).strip()
            # Stop at common header words
            court = re.split(r"C[ỘO]NG\s*H[ÒO]A|\bĐộc lập\b", court)[0].strip()
            if court:
                meta["court"] = court

        # Case number: handles "Số:", "Bản án số:", "Quyết định số:" prefixes
        # and full form like "117/2026/QĐST-HNGĐ" or "66 /2026/HNGĐ-ST"
        # (court documents sometimes have spaces around slashes).
        num_match = re.search(
            r"(?:B[ảa]n\s*[áa]n\s+s[ốo]|Quy[ếe]t\s*đ[ịi]nh\s+s[ốo]|S[ốo])"
            r"\s*:\s*(\d+\s*/\s*\d{4}\s*/\s*[\w\-Đđ]+)",
            text[:2500],
            re.UNICODE | re.IGNORECASE,
        )
        if num_match:
            # Normalize: strip spaces around slashes
            raw = num_match.group(1).strip()
            meta["case_number"] = re.sub(r"\s*/\s*", "/", raw)

        # Date: "ngày 27 tháng 3 năm 2026"
        date_match = re.search(
            r"ng[àa]y\s+(\d{1,2})\s+th[áa]ng\s+(\d{1,2})\s+n[ăa]m\s+(\d{4})",
            text[:2000],
            re.IGNORECASE,
        )
        if date_match:
            d, m, y = date_match.groups()
            meta["date"] = f"{int(d):02d}/{int(m):02d}/{y}"

        # Document type: BẢN ÁN / QUYẾT ĐỊNH
        if re.search(r"\bQUY[ẾE]T\s+Đ[ỊI]NH\b", text[:2500], re.IGNORECASE):
            meta["doc_type"] = "Quyết định"
        elif re.search(r"\bB[ẢA]N\s+[ÁA]N\b", text[:2500], re.IGNORECASE):
            meta["doc_type"] = "Bản án"

        # Case type heuristic from case_number: search for legal-domain code
        # anywhere in the suffix (e.g. "QĐST-HNGĐ" or "HNGĐ-ST" both -> HN&GĐ)
        if "case_number" in meta:
            up = meta["case_number"].upper()
            mapping = {
                "HNGĐ": "Hôn nhân và gia đình",
                "HS": "Hình sự",
                "DS": "Dân sự",
                "KDTM": "Kinh doanh thương mại",
                "LĐ": "Lao động",
                "HC": "Hành chính",
            }
            # Order matters: check longest codes first to avoid partial matches.
            for code in sorted(mapping, key=len, reverse=True):
                if code in up:
                    meta["case_type"] = mapping[code]
                    break

        # Trial level from suffix: ST = Sơ thẩm, PT = Phúc thẩm, GĐT, TT
        if "case_number" in meta:
            up = meta["case_number"].upper()
            if "ST" in up:
                meta["trial_level"] = "Sơ thẩm"
            elif "PT" in up:
                meta["trial_level"] = "Phúc thẩm"
            elif "GĐT" in up:
                meta["trial_level"] = "Giám đốc thẩm"
            elif "TT" in up:
                meta["trial_level"] = "Tái thẩm"

        return meta

    # ------------------------------------------------------------------
    # Phase 3: PDF text extraction
    # ------------------------------------------------------------------

    # Heuristic: PDFs > this size are almost always scanned images
    # (text-based judgments are typically 50-500 KB; scans are 1-15 MB).
    # Grey zone 0.9-2.5MB exists (long real text PDFs) — keep limit generous;
    # the text-length check below will still discard true scans.
    # Can be overridden via env CONGBO_PDF_MAX_BYTES (e.g. 20_000_000 for precedents).
    PDF_MAX_BYTES = int(os.environ.get("CONGBO_PDF_MAX_BYTES", "2500000"))

    def extract_pdf_text(self, doc_id: str, found_url: str = "") -> Optional[str]:
        """Download PDF and extract text. Returns None if failed, missing, or
        appears to be a scanned image (no extractable text)."""
        urls_to_try = []
        if found_url:
            urls_to_try.append(found_url)
        urls_to_try.append(f"{self.BASE_URL}/3ta{doc_id}t1cvn/")

        for pdf_url in urls_to_try:
            try:
                # Cheap HEAD check: skip very large PDFs (likely scans)
                try:
                    head = self._page.request.fetch(
                        pdf_url,
                        method="HEAD",
                        timeout=10000,
                    )
                    clen = int(head.headers.get("content-length", "0") or 0)
                    if clen and clen > self.PDF_MAX_BYTES:
                        log.debug(
                            "Skip %s: PDF too large (%d bytes, likely scan)",
                            doc_id,
                            clen,
                        )
                        continue
                except Exception:
                    pass  # HEAD failed; fall through to GET

                log.debug("Downloading PDF: %s", pdf_url)
                response = self._page.request.get(
                    pdf_url,
                    timeout=30000,
                    headers={"Referer": f"{self.BASE_URL}/{doc_id}.aspx"},
                )

                if response.status == 200 and b"%PDF" in response.body()[:100]:
                    body = response.body()
                    if len(body) > self.PDF_MAX_BYTES:
                        log.debug(
                            "Skip %s: downloaded PDF too large (%d bytes)",
                            doc_id,
                            len(body),
                        )
                        continue
                    doc = fitz.open(stream=body, filetype="pdf")
                    text = "".join(page.get_text() for page in doc)
                    doc.close()
                    if len(text.strip()) < 50:
                        log.debug(
                            "Skip %s: PDF has no extractable text (likely scan)",
                            doc_id,
                        )
                        continue
                    return text
                else:
                    log.debug(
                        "PDF download for %s (%s) returned status %d or invalid content",
                        doc_id,
                        pdf_url,
                        response.status,
                    )
            except Exception as exc:
                log.debug("PDF download error for %s (%s): %s", doc_id, pdf_url, exc)
                continue

        return None

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Lightweight scan: HTML metadata only, no PDF download.
    # ~1.5s/doc instead of 5-10s. Used to build an index for selection.
    # ------------------------------------------------------------------

    def scan_document(self, doc_id: str) -> Optional[dict]:
        """Fetch ONLY HTML metadata (no PDF). Returns minimal record or None."""
        meta = self.get_detail_metadata(doc_id, fast_mode=True)
        if not meta or (not meta.get("title") and not meta.get("case_number")):
            return None
        return {
            "id": doc_id,
            "title": meta.get("title", ""),
            "case_number": meta.get("case_number", ""),
            "date": meta.get("date", ""),
            "trial_level": meta.get("trial_level", ""),
            "case_type": meta.get("case_type", ""),
            "court": meta.get("court", ""),
            "legal_relation": meta.get("legal_relation", ""),
            "precedent_applied": meta.get("precedent_applied", ""),
            "url": meta.get("url", self.DETAIL_URL_TPL.format(doc_id=doc_id)),
        }

    def crawl_document(self, doc_id: str, title_hint: str = "") -> Optional[dict]:
        """Crawl a single document: metadata + PDF text.

        Returns None if the document is missing/empty (caller should just
        skip silently — there are plenty more on the portal).
        """
        # Step 1: Detail Metadata (HTML page — may be sparse)
        meta = self.get_detail_metadata(doc_id)

        # Step 2: PDF Text (skip the doc entirely if PDF is missing)
        body = self.extract_pdf_text(doc_id, found_url=meta.get("found_pdf_url", ""))
        if not body or len(body.strip()) < 50:
            log.debug("Skip %s: PDF missing or empty", doc_id)
            return None

        # Step 3: Backfill metadata from PDF text (more reliable than HTML page)
        pdf_meta = self._parse_metadata_from_pdf_text(body)
        for k, v in pdf_meta.items():
            if not meta.get(k) and v:
                meta[k] = v

        # If we still have no title, derive one from case_number
        if not meta.get("title"):
            if title_hint:
                meta["title"] = title_hint
            elif meta.get("case_number"):
                meta["title"] = meta["case_number"]
            else:
                log.debug("Skip %s: no title/metadata", doc_id)
                return None

        return {
            "id": doc_id,
            "title": meta.get("title", title_hint),
            "case_number": meta.get("case_number", ""),
            "date": meta.get("date", ""),
            "trial_level": meta.get("trial_level", ""),
            "case_type": meta.get("case_type", ""),
            "court": meta.get("court", ""),
            "doc_type": meta.get("doc_type", ""),
            "case_summary": meta.get("case_summary", ""),
            "legal_relation": meta.get("legal_relation", ""),
            "precedent_applied": meta.get("precedent_applied", ""),
            "url": meta.get("url", self.DETAIL_URL_TPL.format(doc_id=doc_id)),
            "body": body,
        }

    def crawl(
        self,
        case_type: str = "all",
        case_level: str = "all",
        doc_type: str = "all",
        max_pages: int = 5,
        limit: Optional[int] = None,
        batch_size: int = 20,
        strategy: str = "search",
        scan_only: bool = False,
        ids_file: Optional[str] = None,
        start_id: Optional[int] = None,
        index_filename: str = "index.jsonl",
    ) -> None:
        """
        Full crawl pipeline.

        Strategies:
            - 'search': Use the search form (default)
            - 'homepage': Scrape the latest from homepage
            - 'probe': Probe sequential IDs

        scan_only: if True, only fetch HTML metadata (no PDF) and append to
            an index.jsonl file. Much faster; used for the scan-then-select
            workflow.
        ids_file: optional path to a text file with one doc_id per line.
            When given, overrides `strategy` and crawls exactly those IDs.
        """
        self._start_browser()
        try:
            docs = []

            if ids_file:
                with open(ids_file, "r", encoding="utf-8") as f:
                    ids = [
                        line.strip()
                        for line in f
                        if line.strip() and not line.startswith("#")
                    ]
                docs = [{"doc_id": did, "title": ""} for did in ids]
                log.info("Loaded %d IDs from %s", len(docs), ids_file)

            elif strategy == "search":
                docs = self.get_listing(
                    case_type=case_type,
                    case_level=case_level,
                    doc_type=doc_type,
                    max_pages=max_pages,
                )
                if not docs:
                    log.warning("Search failed, trying homepage fallback …")
                    if not self._navigate(self.BASE_URL):
                        pass
                    time.sleep(2)
                    docs = self._parse_listing_page()

            elif strategy == "homepage":
                if self._navigate(self.BASE_URL):
                    time.sleep(2)
                    docs = self._parse_listing_page()

            elif strategy == "probe":
                if start_id is None:
                    start_id = self.discover_latest_id()
                else:
                    log.info("Using explicit start_id=%d", start_id)
                want = limit or 50
                # Over-provision: probe ~3x in case of gaps (deleted/missing IDs).
                probe_count = want * 3
                log.info(
                    "Probing IDs starting from %d backwards (want %d, probe up to %d) …",
                    start_id,
                    want,
                    probe_count,
                )
                docs = [
                    {"doc_id": str(start_id - i), "title": ""}
                    for i in range(probe_count)
                ]

            if not docs:
                log.error("No documents found with strategy: %s", strategy)
                return

            # In probe mode we keep extras so failed IDs can be skipped.
            # The inner loop stops when `processed >= limit`.
            if limit and strategy != "probe":
                docs = docs[:limit]

            log.info(
                "Will process up to %d documents (target OK: %d)",
                len(docs),
                limit or len(docs),
            )

            batch: list[dict] = []
            # Resume-safe batch numbering: start after the highest existing
            # batch file so a re-run never overwrites prior results.
            existing_batches = sorted(
                int(m.group(1))
                for f in os.listdir(self.data_dir)
                if (m := re.match(rf"congbobanan_{case_type}_batch_(\d+)\.json$", f))
            )
            batch_num = (existing_batches[-1] + 1) if existing_batches else 1
            if existing_batches:
                log.info(
                    "Resuming batch numbering from %d (found %d existing batches)",
                    batch_num,
                    len(existing_batches),
                )
            processed = 0
            failed = 0
            consecutive_fail = 0
            MAX_CONSECUTIVE_FAIL = 60  # stop probing after this many misses in a row

            target = limit or len(docs)
            mode_label = "Scanning" if scan_only else "Crawling"
            pbar = tqdm(
                total=target,
                desc=mode_label,
                unit="doc",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]",
            )
            pbar.set_postfix_str("OK=0 fail=0")

            # Scan mode appends to a single index file (no batches).
            index_path = os.path.join(self.data_dir, index_filename)
            index_fp = open(index_path, "a", encoding="utf-8") if scan_only else None

            try:
                for i, stub in enumerate(docs):
                    # Stop early if we already collected the requested amount
                    if processed >= target:
                        break

                    if scan_only:
                        result = self.scan_document(stub["doc_id"])
                    else:
                        result = self.crawl_document(
                            stub["doc_id"], stub.get("title", "")
                        )

                    if result:
                        if scan_only:
                            index_fp.write(
                                json.dumps(result, ensure_ascii=False) + "\n"
                            )
                            index_fp.flush()
                        else:
                            batch.append(result)
                        processed += 1
                        consecutive_fail = 0
                        pbar.update(1)
                    else:
                        failed += 1
                        consecutive_fail += 1
                        if (
                            strategy == "probe"
                            and not ids_file
                            and consecutive_fail >= MAX_CONSECUTIVE_FAIL
                        ):
                            tqdm.write(
                                f"⚠ Hit {consecutive_fail} consecutive failures, stopping."
                            )
                            break

                    pbar.set_postfix_str(
                        f"OK={processed} fail={failed} streak={consecutive_fail}"
                    )

                    if not scan_only and len(batch) >= batch_size:
                        self._save_batch(batch_num, batch, case_type)
                        batch_num += 1
                        batch = []

                    time.sleep(self.delay)
            finally:
                pbar.close()
                if index_fp:
                    index_fp.close()

            if batch and not scan_only:
                self._save_batch(batch_num, batch, case_type)

            if scan_only:
                log.info(
                    "Scan done! OK: %d, Failed: %d, Index: %s",
                    processed,
                    failed,
                    index_path,
                )
            else:
                log.info(
                    "Done! OK: %d, Failed: %d, Probed: %d", processed, failed, i + 1
                )

        finally:
            self._stop_browser()

    def _save_batch(self, batch_num: int, results: list[dict], case_type: str) -> None:
        """Save a batch of results."""
        filename = os.path.join(
            self.data_dir, f"congbobanan_{case_type}_batch_{batch_num}.json"
        )
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log.info("Saved %d docs to %s", len(results), filename)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Crawl court judgments from congbobanan.toaan.gov.vn"
    )
    parser.add_argument(
        "--case-type",
        choices=list(CongBoBanAnCrawler.CASE_TYPES.keys()),
        default="all",
        help="Filter by case type (default: all)",
    )
    parser.add_argument(
        "--case-level",
        choices=list(CongBoBanAnCrawler.CASE_LEVELS.keys()),
        default="all",
        help="Filter by trial level (e.g. phuc_tham, giam_doc_tham)",
    )
    parser.add_argument(
        "--doc-type",
        choices=list(CongBoBanAnCrawler.DOC_TYPES.keys()),
        default="all",
        help="Filter by document type (ban_an, quyet_dinh)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Max listing pages to crawl (default: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max documents to process",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Save every N documents (default: 20)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between requests (default: 2.0s)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PATHS.raw_judgments),
        help="Output directory",
    )

    parser.add_argument(
        "--strategy",
        choices=["search", "homepage", "probe"],
        default="search",
        help="Crawl strategy (default: search)",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Lightweight mode: fetch only HTML metadata (no PDF) into index.jsonl. "
        "Use this first, then `select-judgments`, then re-run with --ids-file.",
    )
    parser.add_argument(
        "--ids-file",
        help="Path to a text file with one doc_id per line. Overrides --strategy.",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        help="Probe-mode: explicit start ID instead of auto-discover. "
        "Use this to run multiple probe scanners on disjoint ID ranges in parallel.",
    )
    parser.add_argument(
        "--index-filename",
        default="index.jsonl",
        help="Output filename for scan-only mode (default: index.jsonl). "
        "Use unique names when running parallel scanners.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging (show DEBUG messages)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Tighter default delay for scan_only (no PDF download => faster).
    delay = args.delay
    if args.scan_only and delay >= 2.0:
        delay = 0.3

    crawler = CongBoBanAnCrawler(data_dir=args.output_dir, delay=delay)
    crawler.crawl(
        case_type=args.case_type,
        case_level=args.case_level,
        doc_type=args.doc_type,
        max_pages=args.max_pages,
        limit=args.limit,
        batch_size=args.batch_size,
        strategy=args.strategy,
        scan_only=args.scan_only,
        ids_file=args.ids_file,
        start_id=args.start_id,
        index_filename=args.index_filename,
    )


if __name__ == "__main__":
    main()
