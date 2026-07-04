"""
Dataset Generator for Legal Q&A pairs (OpenAI/Gemini).
Reads cleaned court judgments and uses LLM to generate
structured question-answer pairs at multiple complexity levels.
"""

import json
import os
import hashlib
import asyncio
import argparse
import re
import logging
import time
from typing import Optional
from dotenv import load_dotenv
from tqdm import tqdm

from ura_xlaw.config import PATHS
from ura_xlaw.generation.providers import AsyncOpenAIProvider, create_provider
from ura_xlaw.generation.validation import validate_sample

# Load environment variables from .env
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class DatasetGenerator:
    """Generate legal Q&A dataset from cleaned court judgments using LLM."""

    def __init__(
        self,
        prompt_path: str = str(PATHS.generation_prompt),
        output_dir: str = str(PATHS.processed),
    ):
        self.output_dir = output_dir
        self.prompt_path = prompt_path
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt template not found at {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def _format_prompt(self, item: dict) -> str:
        """Format prompt with court judgment data."""
        # Prioritize body_cleaned for the prompt
        main_content = item.get("body_cleaned") or item.get("body") or ""

        return self.prompt_template.format(
            title=item.get("title", "N/A"),
            case_number=item.get("case_number", "N/A"),
            date=item.get("date", "N/A"),
            trial_level=item.get("trial_level", "N/A"),
            case_type=item.get("case_type", "N/A"),
            court=item.get("court", "N/A"),
            body=main_content,
            url=item.get("url", "N/A"),
        )

    def _enrich_output(self, parsed: dict, item: dict, body: str) -> dict:
        """Override LLM-echoed metadata with verified source values and add
        traceability fields. Mutates and returns `parsed`."""
        doc_id = str(item.get("id") or item.get("doc_id") or "")
        parsed["doc_id"] = doc_id
        # Trust source over LLM echo for metadata
        for k in (
            "case_number",
            "date",
            "court",
            "case_type",
            "trial_level",
            "doc_type",
            "legal_relation",
            "precedent_applied",
            "is_precedent",
            "title",
        ):
            if k in item and item[k] not in (None, ""):
                parsed[k] = item[k]
        parsed["original_source"] = item.get("url") or parsed.get("original_source", "")
        # Grounding traceability: hash of the full body the LLM saw
        parsed["body_sha1"] = hashlib.sha1(
            (body or "").encode("utf-8", errors="ignore")
        ).hexdigest()[:16]
        parsed["body_chars"] = len(body or "")
        return parsed

    def _truncate_body(self, body: str, max_chars: int = 25000) -> str:
        """
        Smart truncation that respects court judgment structure.

        Strategy:
        1. If body fits, return as-is.
        2. Try to keep ALL major sections (parties, XÉT THẤY/NHẬN ĐỊNH,
           QUYẾT ĐỊNH/cite điều luật) intact, dropping only the long
           narrative middle when needed.
        3. Fallback: head/tail split on paragraph boundaries (not mid-sentence).
        """
        if not body or len(body) <= max_chars:
            return body

        # Section markers commonly found in Vietnamese judgments / decisions.
        # We try to anchor on the start of "reasoning" and "verdict" blocks.
        anchors = [
            r"X[ÉE]T TH[ẤA]Y",
            r"NH[ẬA]N\s+Đ[ỊI]NH",
            r"QUY[ẾE]T\s+Đ[ỊI]NH",
            r"C[ĂA]N\s+C[ỨU]",
        ]
        anchor_re = re.compile("|".join(anchors), re.IGNORECASE)

        # Find first reasoning anchor; everything before it = facts/parties.
        first_anchor = anchor_re.search(body)
        if not first_anchor:
            return self._head_tail_truncate(body, max_chars)

        head = body[: first_anchor.start()]  # parties + intro
        tail = body[first_anchor.start() :]  # reasoning + verdict + citations

        # Verdict block is the most legally critical -> always keep last.
        # Allocate ~30% to head, ~70% to tail.
        head_budget = int(max_chars * 0.30)
        tail_budget = max_chars - head_budget - 80  # reserve for marker

        if len(head) > head_budget:
            head = self._cut_on_paragraph(head, head_budget, prefer="start")
        if len(tail) > tail_budget:
            # Keep start of tail (xét thấy) AND end of tail (quyết định + laws).
            tail_head_budget = tail_budget // 2
            tail_tail_budget = tail_budget - tail_head_budget
            tail_start = self._cut_on_paragraph(tail, tail_head_budget, prefer="start")
            tail_end = self._cut_on_paragraph(tail, tail_tail_budget, prefer="end")
            tail = (
                tail_start
                + "\n\n[... narrative truncated for length ...]\n\n"
                + tail_end
            )

        return head + "\n\n" + tail

    @staticmethod
    def _cut_on_paragraph(text: str, max_chars: int, prefer: str = "start") -> str:
        """Cut a string at paragraph boundary closest to the budget."""
        if len(text) <= max_chars:
            return text
        if prefer == "start":
            chunk = text[:max_chars]
            cut = chunk.rfind("\n\n")
            return chunk[:cut] if cut > 0 else chunk
        else:  # end
            chunk = text[-max_chars:]
            cut = chunk.find("\n\n")
            return chunk[cut:] if cut > 0 else chunk

    @staticmethod
    def _head_tail_truncate(body: str, max_chars: int) -> str:
        """Fallback: split head/tail on paragraph boundaries."""
        half = max_chars // 2
        head = body[:half]
        tail = body[-half:]
        head_cut = head.rfind("\n\n")
        tail_cut = tail.find("\n\n")
        if head_cut > 0:
            head = head[:head_cut]
        if tail_cut > 0:
            tail = tail[tail_cut:]
        return head + "\n\n[... content truncated for context limits ...]\n\n" + tail

    def _parse_response(self, response_text: str) -> Optional[dict]:
        try:
            # Clean possible markdown formatting
            clean = re.sub(r"```json\s*|\s*```", "", response_text).strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            return None

    def _get_processed_ids(self, filepath: str) -> set:
        processed_ids = set()
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        # Primary: doc_id written by both sync & async paths
                        if record.get("doc_id"):
                            processed_ids.add(str(record["doc_id"]))
                            continue
                        # Legacy fallback: extract from original_source URL
                        if "original_source" in record:
                            match = re.search(
                                r"2ta(\d+)t1cvn", record["original_source"]
                            )
                            if match:
                                processed_ids.add(match.group(1))
                    except Exception:
                        continue
        return processed_ids

    def process_dataset(
        self,
        input_file: str,
        provider: str,
        api_key: str,
        limit: Optional[int] = None,
        max_body_chars: int = 25000,
        model: Optional[str] = None,
        max_retries: int = 5,
        strict_grounding: bool = True,
    ):
        provider_name = provider.lower()
        llm = create_provider(provider_name, api_key)
        output_file = os.path.join(
            self.output_dir, f"qa_generated_{provider_name}.jsonl"
        )
        rejected_file = os.path.join(
            self.output_dir, f"qa_generated_{provider_name}_rejected.jsonl"
        )

        # Load processed IDs to avoid duplicates
        processed_ids = self._get_processed_ids(output_file)
        logger.info(f"Loaded {len(processed_ids)} already processed documents.")

        if not os.path.exists(input_file):
            logger.error(f"Input file {input_file} not found.")
            return

        count = 0
        failed = 0
        rejected = 0

        # Count total input lines for progress bar
        with open(input_file, "r", encoding="utf-8") as f_count:
            total_lines = sum(1 for _ in f_count)
        target = min(limit, total_lines) if limit else total_lines

        with open(input_file, "r", encoding="utf-8") as f_in, open(
            output_file, "a", encoding="utf-8"
        ) as f_out, open(rejected_file, "a", encoding="utf-8") as f_rej:

            bar = tqdm(
                total=target,
                desc="Generating",
                unit="doc",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
            )
            bar.set_postfix_str("OK=0 rejected=0 skipped=0")
            skipped = 0

            for line in f_in:
                if limit and count >= limit:
                    break

                try:
                    item = json.loads(line)
                except:
                    continue

                doc_id = str(item.get("id", ""))
                if doc_id in processed_ids:
                    skipped += 1
                    bar.set_postfix_str(
                        f"OK={count} rejected={rejected} skipped={skipped}"
                    )
                    continue

                title = item.get("title", "N/A")
                body = item.get("body_cleaned") or item.get("body") or ""

                if not body or len(body) < 200:
                    skipped += 1
                    logger.debug(f"Skipping {doc_id} (body too short)")
                    bar.set_postfix_str(
                        f"OK={count} rejected={rejected} skipped={skipped}"
                    )
                    continue

                logger.debug(f"[{count+1}] Generating Q&A for: {title[:60]}...")

                # Truncate for prompt
                item_copy = item.copy()
                prompt_body = self._truncate_body(body, max_body_chars)
                item_copy["body"] = prompt_body
                item_copy["body_cleaned"] = prompt_body
                prompt = self._format_prompt(item_copy)

                parsed = None
                last_validation = None
                for attempt in range(1, max_retries + 1):
                    try:
                        default_model = (
                            "gpt-4.1"
                            if provider_name == "openai"
                            else "gemini-1.5-flash-latest"
                        )
                        response_text = llm.generate(prompt, model or default_model)
                    except Exception as e:
                        logger.debug(f"  ✗ API Error (attempt {attempt}): {e}")
                        time.sleep(2)
                        continue

                    candidate = self._parse_response(response_text)
                    if candidate is None:
                        logger.debug(f"  ⚠ JSON parse failed (attempt {attempt})")
                        continue

                    last_validation = validate_sample(
                        candidate, body, strict_grounding=strict_grounding
                    )
                    if last_validation.ok:
                        parsed = candidate
                        if last_validation.warnings:
                            for w in last_validation.warnings:
                                logger.debug(f"  ⚠ {w}")
                        break
                    else:
                        logger.debug(
                            f"  ⚠ Validation failed (attempt {attempt}): "
                            f"{last_validation.errors}"
                        )

                if parsed:
                    self._enrich_output(parsed, item, body)
                    f_out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                    f_out.flush()
                    count += 1
                    bar.update(1)
                else:
                    rejected += 1
                    rej_record = {
                        "doc_id": doc_id,
                        "title": title,
                        "errors": last_validation.errors if last_validation else [],
                    }
                    f_rej.write(json.dumps(rej_record, ensure_ascii=False) + "\n")
                    f_rej.flush()
                    bar.update(1)
                    tqdm.write(
                        f"  ✗ Rejected doc {doc_id} after {max_retries} attempts"
                    )

                bar.set_postfix_str(f"OK={count} rejected={rejected} skipped={skipped}")
                time.sleep(1)  # Basic safety

            bar.close()

        print(
            f"\nFinished. Generated: {count}, Rejected: {rejected}, "
            f"Skipped: {skipped}, Failed: {failed}"
        )

    # ------------------------------------------------------------------
    # Async pipeline (OpenAI only) — much faster for large batches.
    # ------------------------------------------------------------------

    async def process_dataset_async(
        self,
        input_file: str,
        api_key: str,
        limit: Optional[int] = None,
        max_body_chars: int = 25000,
        model: Optional[str] = None,
        max_retries: int = 5,
        strict_grounding: bool = True,
        concurrency: int = 8,
    ):
        """Async OpenAI pipeline. Runs `concurrency` requests in parallel."""
        provider = "openai"
        m = model or "gpt-4.1"
        output_file = os.path.join(self.output_dir, f"qa_generated_{provider}.jsonl")
        rejected_file = os.path.join(
            self.output_dir, f"qa_generated_{provider}_rejected.jsonl"
        )

        processed_ids = self._get_processed_ids(output_file)
        logger.info(f"Loaded {len(processed_ids)} already processed documents.")

        if not os.path.exists(input_file):
            logger.error(f"Input file {input_file} not found.")
            return

        # Pre-load eligible items
        items: list[dict] = []
        skipped = 0
        with open(input_file, "r", encoding="utf-8") as f_in:
            for line in f_in:
                if limit and len(items) >= limit:
                    break
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                doc_id = str(item.get("id", ""))
                if doc_id in processed_ids:
                    skipped += 1
                    continue
                body = item.get("body_cleaned") or item.get("body") or ""
                if not body or len(body) < 200:
                    skipped += 1
                    continue
                items.append(item)

        if not items:
            print(f"Nothing to do. Skipped {skipped}.")
            return

        target = len(items)
        logger.info(
            f"Async generating {target} docs (concurrency={concurrency}, model={m})"
        )

        llm = AsyncOpenAIProvider(api_key=api_key)
        sem = asyncio.Semaphore(concurrency)
        write_lock = asyncio.Lock()
        f_out = open(output_file, "a", encoding="utf-8")
        f_rej = open(rejected_file, "a", encoding="utf-8")

        bar = tqdm(
            total=target,
            desc="Generating",
            unit="doc",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )
        counters = {"ok": 0, "rejected": 0, "failed": 0}
        bar.set_postfix_str("OK=0 rejected=0 failed=0")

        async def worker(item: dict) -> None:
            doc_id = str(item.get("id", ""))
            title = item.get("title", "N/A")
            body = item.get("body_cleaned") or item.get("body") or ""

            item_copy = item.copy()
            prompt_body = self._truncate_body(body, max_body_chars)
            item_copy["body"] = prompt_body
            item_copy["body_cleaned"] = prompt_body
            prompt = self._format_prompt(item_copy)

            parsed = None
            last_validation = None
            last_err: Optional[str] = None

            async with sem:
                for attempt in range(1, max_retries + 1):
                    try:
                        response_text = await llm.generate(prompt, model=m)
                    except Exception as e:
                        last_err = str(e)
                        logger.debug(f"  ✗ API Error doc {doc_id} (#{attempt}): {e}")
                        # Parse retry-after for 429; else exponential backoff
                        wait = 2 * attempt
                        m429 = re.search(r"try again in ([\d.]+)s", last_err)
                        if m429:
                            wait = float(m429.group(1)) + 1.0
                        await asyncio.sleep(wait)
                        continue

                    candidate = self._parse_response(response_text)
                    if candidate is None:
                        last_err = "json_parse_failed"
                        continue

                    last_validation = validate_sample(
                        candidate, body, strict_grounding=strict_grounding
                    )
                    if last_validation.ok:
                        parsed = candidate
                        break

            async with write_lock:
                if parsed:
                    self._enrich_output(parsed, item, body)
                    f_out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                    f_out.flush()
                    counters["ok"] += 1
                else:
                    if last_validation is None and last_err:
                        counters["failed"] += 1
                        errs = [f"api: {last_err}"]
                    else:
                        counters["rejected"] += 1
                        errs = last_validation.errors if last_validation else []
                    f_rej.write(
                        json.dumps(
                            {"doc_id": doc_id, "title": title, "errors": errs},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    f_rej.flush()
                    tqdm.write(f"  ✗ Doc {doc_id}: {errs}")
                bar.update(1)
                bar.set_postfix_str(
                    f"OK={counters['ok']} rejected={counters['rejected']} "
                    f"failed={counters['failed']}"
                )

        try:
            await asyncio.gather(*(worker(it) for it in items))
        finally:
            bar.close()
            f_out.close()
            f_rej.close()
            await llm.close()

        print(
            f"\nFinished. Generated: {counters['ok']}, "
            f"Rejected: {counters['rejected']}, Failed: {counters['failed']}, "
            f"Pre-skipped: {skipped}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Legal Dataset Generator (LLM)")
    parser.add_argument("--provider", choices=["openai", "gemini"], required=True)
    parser.add_argument("--key", help="API Key (or use environment variable)")
    parser.add_argument("--input", default=str(PATHS.cleaned_judgments))
    parser.add_argument("--limit", type=int, help="Number of docs to process")
    parser.add_argument("--model", help="Specific model name (e.g. gpt-4o-mini)")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Parallel requests (OpenAI only). 1 = sync mode (default). "
        "Try 8-16 for fast batch generation.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)

    env_key = "OPENAI_API_KEY" if args.provider == "openai" else "GEMINI_API_KEY"
    key = args.key or os.environ.get(env_key)
    if not key:
        print("Error: API Key MUST be provided via --key or environment variable.")
        raise SystemExit(1)

    gen = DatasetGenerator()

    if args.concurrency > 1:
        if args.provider != "openai":
            print("--concurrency >1 is only supported for --provider openai.")
            raise SystemExit(1)
        asyncio.run(
            gen.process_dataset_async(
                input_file=args.input,
                api_key=key,
                limit=args.limit,
                model=args.model,
                concurrency=args.concurrency,
            )
        )
    else:
        gen.process_dataset(
            input_file=args.input,
            provider=args.provider,
            api_key=key,
            limit=args.limit,
            model=args.model,
        )


if __name__ == "__main__":
    main()
