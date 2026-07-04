"""Unified command router for URA-xLaw."""

from __future__ import annotations

import importlib
import sys


COMMANDS = {
    "crawl-judgments": (
        "ura_xlaw.acquisition.court_judgments",
        "Crawl court judgments and metadata.",
    ),
    "select-judgments": (
        "ura_xlaw.acquisition.select_judgments",
        "Select a balanced set of crawled judgments.",
    ),
    "clean-judgments": (
        "ura_xlaw.preprocessing.clean_judgments",
        "Clean and deduplicate raw judgments.",
    ),
    "generate-qa": (
        "ura_xlaw.generation.generator",
        "Generate grounded QA with an LLM provider.",
    ),
    "build-corpus": (
        "ura_xlaw.corpus.build_corpus",
        "Build the normalized legal corpus.",
    ),
    "build-supplemental-corpus": (
        "ura_xlaw.corpus.build_supplemental_corpus",
        "Build supplemental chunks for missing laws.",
    ),
    "build-precedent-corpus": (
        "ura_xlaw.corpus.build_precedent_corpus",
        "Add Vietnamese precedent chunks.",
    ),
    "crawl-missing-laws": (
        "ura_xlaw.acquisition.laws.missing_laws",
        "Crawl legal documents missing from the corpus.",
    ),
    "map-citations": (
        "ura_xlaw.corpus.map_citations",
        "Map legal citations to corpus chunks.",
    ),
    "export-answerable": (
        "ura_xlaw.dataset.export_answerable",
        "Export QA whose citations are fully grounded.",
    ),
    "split": (
        "ura_xlaw.dataset.split_dataset",
        "Create leakage-safe train and test splits.",
    ),
    "flatten": (
        "ura_xlaw.dataset.flatten_qa",
        "Flatten nested generated QA for inspection.",
    ),
    "build-unanswerable": (
        "ura_xlaw.dataset.build_unanswerable",
        "Build judgment-derived unanswerable QA.",
    ),
    "normalize-grounding": (
        "ura_xlaw.dataset.normalize_grounding",
        "Normalize split article chunks and gold IDs.",
    ),
    "package-release": (
        "ura_xlaw.dataset.package_release",
        "Package processed artifacts into dataset/.",
    ),
    "validate-release": (
        "ura_xlaw.dataset.validate_release",
        "Validate checksums, schemas, leakage, and grounding.",
    ),
}


def print_help() -> None:
    print("URA-xLaw dataset construction toolkit\n")
    print("Usage: ura-xlaw <command> [options]\n")
    print("Commands:")
    width = max(map(len, COMMANDS))
    for command, (_, description) in COMMANDS.items():
        print(f"  {command:<{width}}  {description}")
    print("\nRun a command with --help for its options.")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print_help()
        return

    command = sys.argv[1]
    spec = COMMANDS.get(command)
    if spec is None:
        print(f"Unknown command: {command}\n", file=sys.stderr)
        print_help()
        raise SystemExit(2)

    module_name, _ = spec
    module = importlib.import_module(module_name)
    sys.argv = [f"ura-xlaw {command}", *sys.argv[2:]]
    module.main()


if __name__ == "__main__":
    main()
