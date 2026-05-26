"""Helpers for resolving repo-relative data paths from any cwd."""

from __future__ import annotations

from pathlib import Path


def resolve_data_path(value: str) -> Path:
    """Resolve a data path that is robust to whichever cwd the API was launched from.

    Defaults like ``data/corpus/chunks.jsonl`` work when uvicorn is launched from the
    repo root but break when launched from ``apps/api/``. This helper tries the path
    as-is, then walks up from this module to find a project root containing the path.
    """
    candidate = Path(value)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    here = Path(__file__).resolve()
    for ancestor in [here.parent, *here.parents]:
        anchored = ancestor / candidate
        if anchored.exists():
            return anchored
    return candidate
