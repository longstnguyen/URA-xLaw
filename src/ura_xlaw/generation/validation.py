"""
Validation layer for generated legal Q&A samples.

Checks:
1. JSON schema correctness
2. Grounding: every cited article in `law_applied` must appear in source body
3. Anti-hallucination: anonymized names in `situation` must come from source
4. Diversity: 3 entries should not be near-duplicates

Usage:
    from ura_xlaw.generation.validation import validate_sample, ValidationResult
    result = validate_sample(parsed_json, source_body)
    if not result.ok:
        print(result.errors)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional


REQUIRED_TOP_FIELDS = {"situation", "entries"}
REQUIRED_ENTRY_FIELDS = {
    "complexity_level",
    "question",
    "answer",
    "legal_reasoning",
    "law_applied",
}
COMPLEXITY_LEVELS = {"Simple", "Medium", "Complex"}


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _check_schema(sample: dict) -> list[str]:
    errs = []
    missing = REQUIRED_TOP_FIELDS - sample.keys()
    if missing:
        errs.append(f"Missing top-level fields: {missing}")

    entries = sample.get("entries")
    if not isinstance(entries, list):
        errs.append("`entries` must be a list")
        return errs

    if len(entries) != 3:
        errs.append(f"Expected 3 entries, got {len(entries)}")

    seen_levels = set()
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            errs.append(f"Entry {i} is not a dict")
            continue
        miss = REQUIRED_ENTRY_FIELDS - e.keys()
        if miss:
            errs.append(f"Entry {i} missing fields: {miss}")
        lvl = e.get("complexity_level")
        if lvl not in COMPLEXITY_LEVELS:
            errs.append(f"Entry {i} has invalid complexity_level: {lvl}")
        seen_levels.add(lvl)
        laws = e.get("law_applied")
        if not isinstance(laws, list) or not laws:
            errs.append(f"Entry {i} `law_applied` must be a non-empty list")

    if seen_levels and seen_levels != COMPLEXITY_LEVELS:
        errs.append(
            f"complexity_level coverage incomplete: got {seen_levels}, "
            f"expected {COMPLEXITY_LEVELS}"
        )

    return errs


# ---------------------------------------------------------------------------
# Grounding: cited articles must appear in source
# ---------------------------------------------------------------------------

# Match formats like "Điều 55", "điều 212", "Đ55" inside source text
_ARTICLE_IN_SOURCE_RE = re.compile(r"[ĐđDd]i[eề]u\s*(\d+)", re.IGNORECASE)
# Match enumerations after the first article: "Điều 55, 81, 82, 83 Luật ..."
# We capture the leading number with the head regex and a trailing run of
# ", N, N, N" before the law name.
_ARTICLE_ENUM_RE = re.compile(r"[ĐđDd]i[eề]u\s*(\d+(?:\s*,\s*\d+)+)", re.IGNORECASE)
# Match the structured citation `D<num>:...` or `D<num>:K<num>:...`
_CITATION_RE = re.compile(r"^D(\d+)(?::K(\d+))?(?::d([^:]+))?(?::(.+))?$")
# Án lệ (case law / precedent) citation: "AL11/2017/AL" or "AL11/2017/AL:Tên án lệ"
# Vietnamese precedents are referenced by court as "Án lệ số 11/2017/AL".
_AN_LE_CITATION_RE = re.compile(r"^AL\s*(\d+)\s*/\s*(\d{4})\s*/\s*AL(?::(.+))?$", re.I)
# In source body, precedents appear as "Án lệ số 11/2017/AL" or "án lệ 11/2017/AL"
_AN_LE_IN_SOURCE_RE = re.compile(
    r"[ÁA]n\s*l[ệe](?:\s*s[ốo])?\s*(\d+)\s*/\s*(\d{4})\s*/\s*AL", re.I
)


def _extract_source_articles(body: str) -> set[str]:
    """Extract article numbers explicitly cited in source body."""
    arts: set[str] = set()
    if not body:
        return arts
    arts.update(_ARTICLE_IN_SOURCE_RE.findall(body))
    # Also expand enumerations like "Điều 55, 81, 82, 83"
    for group in _ARTICLE_ENUM_RE.findall(body):
        for num in re.findall(r"\d+", group):
            arts.add(num)
    return arts


def _extract_source_precedents(body: str) -> set[tuple[str, str]]:
    """Extract precedent IDs (number, year) cited in source body, e.g. (11, 2017)."""
    return set(_AN_LE_IN_SOURCE_RE.findall(body or ""))


def _check_grounding(sample: dict, body: str) -> tuple[list[str], list[str]]:
    """Verify every law_applied article appears in source body."""
    errs, warns = [], []
    source_articles = _extract_source_articles(body)
    source_precedents = _extract_source_precedents(body)
    if not source_articles and not source_precedents:
        warns.append("No articles detected in source body — skipping grounding check")
        return errs, warns

    for i, entry in enumerate(sample.get("entries", [])):
        for cite in entry.get("law_applied", []):
            if not isinstance(cite, str):
                errs.append(f"Entry {i}: citation is not a string: {cite!r}")
                continue
            cite_str = cite.strip()
            # Try án lệ format first
            al_m = _AN_LE_CITATION_RE.match(cite_str)
            if al_m:
                key = (al_m.group(1), al_m.group(2))
                if key not in source_precedents:
                    errs.append(
                        f"Entry {i}: precedent AL{key[0]}/{key[1]}/AL cited but "
                        f"NOT found in source body (citation: {cite!r})"
                    )
                continue
            # Otherwise expect article format D<n>...
            m = _CITATION_RE.match(cite_str)
            if not m:
                errs.append(
                    f"Entry {i}: citation does not match format "
                    f"D<n>[:K<n>][:d<x>][:LawName] or AL<n>/<year>/AL[:Name] "
                    f"— got {cite!r}"
                )
                continue
            article_num = m.group(1)
            if article_num not in source_articles:
                errs.append(
                    f"Entry {i}: article D{article_num} cited but NOT found "
                    f"in source body (citation: {cite!r})"
                )
    return errs, warns


# ---------------------------------------------------------------------------
# Anti-hallucination on `situation`: extract anonymized names and check source
# ---------------------------------------------------------------------------

# Vietnamese anonymized name pattern: e.g. "Chị Nguyễn Thị Lệ Q", "Anh Phan Tiến D",
# "Ông A", "Bà B", "Công ty X". We focus on the trailing single-letter alias.
_ANON_NAME_RE = re.compile(
    r"\b(?:Ông|Bà|Anh|Chị|Em|Cháu|Công ty|Bị cáo|Nguyên đơn|Bị đơn)\s+"
    r"(?:[A-ZĐÀ-Ỹ][\wÀ-ỹ]*\s+){0,4}([A-ZĐÀ-Ỹ])\b"
)


def _check_situation_names(sample: dict, body: str) -> list[str]:
    """Each anonymized name alias used in situation must appear in source body."""
    warns = []
    situation = sample.get("situation", "")
    if not situation:
        return warns
    aliases_in_situation = set(_ANON_NAME_RE.findall(situation))
    aliases_in_source = set(_ANON_NAME_RE.findall(body or ""))

    missing = aliases_in_situation - aliases_in_source
    if missing:
        warns.append(
            f"Situation uses anonymized aliases not present in source: {missing}"
        )
    return warns


# ---------------------------------------------------------------------------
# Diversity: entries' questions should not be near-duplicates
# ---------------------------------------------------------------------------


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _check_diversity(sample: dict, threshold: float = 0.70) -> list[str]:
    warns = []
    entries = sample.get("entries", [])
    questions = [e.get("question", "") for e in entries]
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            sim = _similarity(questions[i], questions[j])
            if sim >= threshold:
                warns.append(
                    f"Entries {i} & {j} have similar questions "
                    f"(similarity={sim:.2f} ≥ {threshold})"
                )
    return warns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_sample(
    sample: dict,
    body: str,
    diversity_threshold: float = 0.70,
    strict_grounding: bool = True,
) -> ValidationResult:
    """
    Validate a generated Q&A sample against its source body.

    Args:
        sample: parsed JSON from the LLM
        body: cleaned source judgment text
        diversity_threshold: question similarity above which entries are
            flagged as near-duplicates
        strict_grounding: if True, ungrounded citations are errors;
            if False, they are warnings

    Returns:
        ValidationResult with `ok` flag, `errors`, and `warnings`.
    """
    errors: list[str] = []
    warnings: list[str] = []

    schema_errs = _check_schema(sample)
    errors.extend(schema_errs)

    # Skip deeper checks if schema is broken
    if not schema_errs:
        ground_errs, ground_warns = _check_grounding(sample, body)
        if strict_grounding:
            errors.extend(ground_errs)
        else:
            warnings.extend(ground_errs)
        warnings.extend(ground_warns)

        warnings.extend(_check_situation_names(sample, body))
        warnings.extend(_check_diversity(sample, diversity_threshold))

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)
