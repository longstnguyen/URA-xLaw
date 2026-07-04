"""Central project metadata and filesystem paths.

All paths are anchored at the repository root, so commands work regardless of
the caller's current directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DATASET_NAME = "URA-xLaw"


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    workspace: Path
    raw: Path
    processed: Path
    release: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        root = Path(
            os.environ.get("URA_XLAW_ROOT", Path(__file__).resolve().parents[2])
        ).resolve()
        workspace = root / "data"
        return cls(
            root=root,
            workspace=workspace,
            raw=workspace / "raw",
            processed=workspace / "processed",
            release=root / "dataset",
        )

    @property
    def raw_judgments(self) -> Path:
        return self.raw / "congbobanan"

    @property
    def cleaned_judgments(self) -> Path:
        return self.processed / "judgments_cleaned.jsonl"

    @property
    def generation_prompt(self) -> Path:
        return Path(__file__).resolve().parent / "prompts" / "qa_generation.txt"

    @property
    def full_corpus(self) -> Path:
        return self.release / "corpus" / "full.parquet"

    @property
    def retrieval_corpus(self) -> Path:
        return self.release / "corpus" / "retrieval.parquet"


PATHS = ProjectPaths.discover()
