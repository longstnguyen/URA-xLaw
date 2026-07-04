"""Unified command router for URA-xLaw."""

from __future__ import annotations

import importlib
import sys


COMMANDS = {
    "crawl-judgments": "ura_xlaw.acquisition.judgments",
    "select-judgments": "ura_xlaw.acquisition.select_judgments",
    "clean-judgments": "ura_xlaw.preprocessing.judgments",
    "generate-qa": "ura_xlaw.generation.service",
    "build-corpus": "ura_xlaw.corpus.build",
    "map-citations": "ura_xlaw.corpus.citations",
    "export-answerable": "ura_xlaw.dataset.export",
    "split": "ura_xlaw.dataset.split",
    "flatten": "ura_xlaw.dataset.flatten",
    "build-real-unanswerable": "ura_xlaw.dataset.unanswerable",
    "normalize-grounding": "ura_xlaw.dataset.normalize_grounding",
    "package-release": "ura_xlaw.dataset.release",
    "validate-release": "ura_xlaw.dataset.validate_release",
}


def print_help() -> None:
    print("URA-xLaw dataset construction toolkit\n")
    print("Usage: python -m ura_xlaw <command> [options]\n")
    print("Commands:")
    for command in COMMANDS:
        print(f"  {command}")
    print("\nRun a command with --help for its options.")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print_help()
        return

    command = sys.argv[1]
    module_name = COMMANDS.get(command)
    if module_name is None:
        print(f"Unknown command: {command}\n", file=sys.stderr)
        print_help()
        raise SystemExit(2)

    module = importlib.import_module(module_name)
    sys.argv = [f"ura-xlaw {command}", *sys.argv[2:]]
    module.main()


if __name__ == "__main__":
    main()
