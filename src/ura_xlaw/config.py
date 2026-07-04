"""Central project metadata and filesystem paths.

All paths are anchored at the repository root, so commands work regardless of
the caller's current directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DATASET_NAME = "URA-xLaw"


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    raw: Path
    processed: Path
    prompts: Path
    release: Path
    crawl_metadata: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        root = Path(__file__).resolve().parents[2]
        data = root / "data"
        return cls(
            root=root,
            data=data,
            raw=data / "raw",
            processed=data / "processed",
            prompts=data / "prompts",
            release=data,
            crawl_metadata=data / "metadata" / "crawl",
        )

    @property
    def raw_judgments(self) -> Path:
        return self.raw / "congbobanan"

    @property
    def cleaned_judgments(self) -> Path:
        return self.processed / "judgments_cleaned.jsonl"

    @property
    def generation_prompt(self) -> Path:
        return self.prompts / "legal_gen_prompt.txt"


PATHS = ProjectPaths.discover()
