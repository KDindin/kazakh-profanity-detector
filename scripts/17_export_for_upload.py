"""Create a clean export of the repository for GitHub's web uploader.

Copies exactly the files that .gitignore would keep into
``~/publish/kazakh-profanity-detector`` (outside the project, so the export is never
picked up by a later export), and refuses to run if the destination already exists.

    python scripts/17_export_for_upload.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = Path.home() / "publish" / "kazakh-profanity-detector"

SKIP_DIRS = {
    ".git", ".venv", ".pytest_cache", "__pycache__", "kazroberta_kk_profanity",
    "raw", "interim", "_hf",
}
SKIP_SUFFIXES = {".pyc"}
SKIP_NAMES = {"Thumbs.db", ".DS_Store"}


def keep(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    parts = set(rel.parts[:-1])
    if parts & SKIP_DIRS:
        return False
    if any(p.endswith(".egg-info") for p in rel.parts):
        return False
    if path.suffix in SKIP_SUFFIXES or path.name in SKIP_NAMES:
        return False
    # data/raw and data/interim are excluded by directory name above
    return True


def main() -> int:
    if DEST.exists():
        print(f"refusing to overwrite {DEST}")
        return 1
    files = [p for p in ROOT.rglob("*") if p.is_file() and keep(p, ROOT)]
    total = 0
    for src in files:
        dst = DEST / src.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        total += src.stat().st_size
    print(f"exported {len(files)} files, {total / 1e6:.2f} MB -> {DEST}")
    for p in sorted(DEST.iterdir()):
        print("  ", p.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
