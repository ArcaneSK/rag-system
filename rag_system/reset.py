from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .config import Settings


def _count_contents(root: Path) -> tuple[int, int]:
    if not root.exists():
        return 0, 0

    file_count = 0
    dir_count = 0
    for path in root.rglob("*"):
        if path.name == ".gitkeep":
            continue
        if path.is_dir():
            dir_count += 1
        else:
            file_count += 1
    return file_count, dir_count


def _clear_directory(root: Path) -> dict[str, int]:
    files_before, dirs_before = _count_contents(root)

    if root.exists():
        for child in root.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()

    root.mkdir(parents=True, exist_ok=True)
    return {
        "deleted_files": files_before,
        "deleted_dirs": dirs_before,
    }


def reset_local_storage(settings: Settings) -> dict[str, object]:
    settings.ensure_directories()
    chroma_summary = _clear_directory(settings.chroma_dir)
    state_summary = _clear_directory(settings.state_dir)

    return {
        "pdf_dir_preserved": str(settings.pdf_dir),
        "chroma_dir_cleared": str(settings.chroma_dir),
        "state_dir_cleared": str(settings.state_dir),
        "deleted_files": chroma_summary["deleted_files"] + state_summary["deleted_files"],
        "deleted_dirs": chroma_summary["deleted_dirs"] + state_summary["deleted_dirs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clear the local ChromaDB store and indexing state while preserving PDFs."
    )
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    args = parser.parse_args()

    settings = Settings.from_env()
    if not args.yes:
        prompt = (
            f"This will delete local index data in {settings.chroma_dir} and {settings.state_dir} "
            "but keep your PDFs. Continue? [y/N]: "
        )
        if input(prompt).strip().lower() not in {"y", "yes"}:
            raise SystemExit("Cancelled.")

    summary = reset_local_storage(settings)
    print(json.dumps(summary, indent=2))
