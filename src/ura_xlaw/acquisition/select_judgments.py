"""
Select doc IDs from a scan index for balanced/targeted deep crawl.

Workflow:
    1. Run crawler in scan mode to build index.jsonl:
        python -m ura_xlaw crawl-judgments --scan-only --strategy probe --limit 2000
    2. Run this script to pick IDs:
        python -m ura_xlaw select-judgments --precedent-only --out selected.txt
       or with explicit quotas:
        python -m ura_xlaw select-judgments --quota hinh_su=50,dan_su=50,hon_nhan=40 --out selected.txt
    3. Deep-crawl only the selected IDs:
        python -m ura_xlaw crawl-judgments --ids-file selected.txt
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Optional

from ura_xlaw.config import PATHS

# Map normalized case_type label -> short key used in --quota
CASE_TYPE_KEY = {
    "Hình sự": "hinh_su",
    "Dân sự": "dan_su",
    "Hôn nhân và gia đình": "hon_nhan",
    "Hành chính": "hanh_chinh",
    "Kinh doanh thương mại": "kinh_doanh",
    "Lao động": "lao_dong",
}
KEY_TO_LABEL = {v: k for k, v in CASE_TYPE_KEY.items()}


def load_index(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def has_precedent(row: dict) -> bool:
    val = (row.get("precedent_applied") or "").strip()
    if not val:
        return False
    # Some pages render an empty placeholder.
    return val.lower() not in {"không", "không có", "n/a", "-", "none"}


def parse_quota(spec: Optional[str]) -> dict[str, int]:
    """Parse 'hinh_su=50,dan_su=50' into {label: count}."""
    if not spec:
        return {}
    out: dict[str, int] = {}
    for part in spec.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        if k not in KEY_TO_LABEL:
            raise SystemExit(
                f"Unknown case_type key '{k}'. Valid: {list(KEY_TO_LABEL)}"
            )
        out[KEY_TO_LABEL[k]] = int(v)
    return out


def select_balanced(
    rows: list[dict],
    total: int,
    quota: dict[str, int],
    seed: int = 42,
) -> list[str]:
    """Pick `total` IDs respecting quota; remainder filled proportionally."""
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r.get("case_type") or "Khác"].append(r)
    for v in by_type.values():
        rng.shuffle(v)

    picked: list[str] = []

    # Apply explicit quotas first
    for label, n in quota.items():
        bucket = by_type.get(label, [])
        take = bucket[:n]
        picked.extend(r["id"] for r in take)
        by_type[label] = bucket[n:]

    if total and len(picked) < total:
        # Proportional fill from remaining pool
        remaining_pool = [r for bucket in by_type.values() for r in bucket]
        rng.shuffle(remaining_pool)
        need = total - len(picked)
        picked.extend(r["id"] for r in remaining_pool[:need])

    return picked


def main() -> None:
    p = argparse.ArgumentParser(description="Select doc IDs from scan index.")
    p.add_argument(
        "--index",
        default=str(PATHS.raw_judgments / "index.jsonl"),
        help="Path to the judgment scan index.",
    )
    p.add_argument("--out", help="Output text file (one id/line)")
    p.add_argument(
        "--precedent-only",
        action="store_true",
        help="Keep only docs that cite an án lệ (precedent).",
    )
    p.add_argument(
        "--no-precedent",
        action="store_true",
        help="Keep only docs WITHOUT a precedent citation.",
    )
    p.add_argument(
        "--case-type",
        action="append",
        default=[],
        help=f"Filter by case-type key (repeatable). Valid: {list(KEY_TO_LABEL)}",
    )
    p.add_argument(
        "--trial-level",
        action="append",
        default=[],
        help="Filter by trial level label (e.g. 'Sơ thẩm', 'Phúc thẩm'). Repeatable.",
    )
    p.add_argument(
        "--total",
        type=int,
        default=0,
        help="Total IDs to pick (0 = all that pass filters).",
    )
    p.add_argument(
        "--quota",
        help="Per-type quotas, e.g. 'hinh_su=50,dan_su=50,hon_nhan=40'. "
        "Combined with --total to fill the rest proportionally.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--stats", action="store_true", help="Print distribution stats and exit."
    )
    args = p.parse_args()

    if args.precedent_only and args.no_precedent:
        raise SystemExit("--precedent-only and --no-precedent are mutually exclusive")

    rows = load_index(args.index)
    print(f"Loaded {len(rows)} rows from {args.index}")

    if args.stats:
        if not rows:
            print("Index is empty.")
            return
        ct = Counter(r.get("case_type") or "Khác" for r in rows)
        tl = Counter(r.get("trial_level") or "Khác" for r in rows)
        prec = sum(1 for r in rows if has_precedent(r))
        print("\n=== case_type distribution ===")
        for k, v in ct.most_common():
            print(f"  {k:35s} {v:6d} ({v/len(rows)*100:.1f}%)")
        print("\n=== trial_level distribution ===")
        for k, v in tl.most_common():
            print(f"  {k:35s} {v:6d} ({v/len(rows)*100:.1f}%)")
        print(
            f"\nWith precedent (án lệ): {prec} / {len(rows)} "
            f"({prec/len(rows)*100:.2f}%)"
        )
        return

    if not args.out:
        p.error("--out is required unless --stats is used")

    # Apply filters
    filtered = rows
    if args.precedent_only:
        filtered = [r for r in filtered if has_precedent(r)]
    if args.no_precedent:
        filtered = [r for r in filtered if not has_precedent(r)]
    if args.case_type:
        wanted = {KEY_TO_LABEL[k] for k in args.case_type if k in KEY_TO_LABEL}
        filtered = [r for r in filtered if r.get("case_type") in wanted]
    if args.trial_level:
        wanted_levels = set(args.trial_level)
        filtered = [r for r in filtered if r.get("trial_level") in wanted_levels]

    print(f"After filters: {len(filtered)} rows")

    quota = parse_quota(args.quota)
    total = args.total or len(filtered)
    picked = select_balanced(filtered, total, quota, seed=args.seed)

    # Distribution report
    by_type = Counter()
    by_id = {r["id"]: r for r in filtered}
    for did in picked:
        by_type[(by_id.get(did) or {}).get("case_type") or "Khác"] += 1
    print(f"\nSelected {len(picked)} IDs:")
    for k, v in by_type.most_common():
        print(f"  {k:35s} {v:6d}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for did in picked:
            f.write(f"{did}\n")
    print(f"\nWrote {len(picked)} IDs to {args.out}")


if __name__ == "__main__":
    main()
