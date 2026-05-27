"""Helpers for resolving repo-relative data paths from any cwd."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _allowed_roots() -> list[Path]:
    """Roots under which data paths are allowed to resolve."""
    here = Path(__file__).resolve()
    roots: list[Path] = []
    seen: set[Path] = set()
    for ancestor in here.parents:
        if ancestor in seen:
            continue
        seen.add(ancestor)
        roots.append(ancestor)
        if (ancestor / "apps").is_dir() or (ancestor / "pyproject.toml").is_file() or (ancestor / ".git").exists():
            break
    # System temp dirs are caller-controlled (pytest fixtures, ephemeral
    # uploads, etc.) and never attacker-controlled in production, so they are
    # safe to whitelist as a convenience.
    try:
        roots.append(Path(tempfile.gettempdir()).resolve())
    except OSError:
        pass
    # /private/var/folders/... on macOS is the realpath behind /var/folders/...
    # which `tempfile.gettempdir()` returns on some configurations. Include the
    # /private prefix too so resolved paths stay inside.
    private_tmp = Path("/private/var/folders")
    if private_tmp.exists():
        roots.append(private_tmp)
    extra = os.environ.get("W3C_DATA_ALLOWED_ROOTS", "")
    for token in extra.split(":"):
        token = token.strip()
        if token:
            roots.append(Path(token).resolve())
    return roots


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_data_path(value: str) -> Path:
    """Resolve a data path that is robust to whichever cwd the API was launched from.

    Defaults like ``data/corpus/chunks.jsonl`` work when uvicorn is launched from the
    repo root but break when launched from ``apps/api/``. This helper tries the path
    as-is, then walks up from this module to find a project root containing the path.

    Security: an absolute path or relative-with-``..`` path is only accepted if it
    resolves inside one of the discovered project roots (or ``W3C_DATA_ALLOWED_ROOTS``).
    Anything outside that boundary returns the original repo-anchored candidate to keep
    the caller in a safe directory rather than reading from an attacker-supplied path.
    """
    candidate = Path(value)
    roots = _allowed_roots()
    if not roots:
        return candidate

    if candidate.is_absolute():
        resolved = candidate.resolve()
        for root in roots:
            if _is_within(resolved, root):
                return resolved
        raise ValueError(
            f"Refusing to resolve data path outside of project roots: {value!r}"
        )

    if candidate.exists():
        resolved = candidate.resolve()
        for root in roots:
            if _is_within(resolved, root):
                return resolved

    for root in roots:
        anchored = (root / candidate).resolve()
        if not _is_within(anchored, root):
            continue
        if anchored.exists():
            return anchored

    return roots[0] / candidate
