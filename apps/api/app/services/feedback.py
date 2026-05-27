"""Append-only JSONL feedback store.

User feedback (thumbs up/down + optional comment + audit trail) is the most
valuable ground-truth signal we can collect from Member-only alpha users.
This module persists each submission to a single JSONL file so it can be:
- inspected by hand
- grepped/jq'd from the CLI
- migrated to a relational store later without losing data
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # POSIX only; on Windows we degrade to in-process locking
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]


# Within-process serialization (kept even on POSIX so concurrent threads in
# one worker don't race on the file descriptor between open() and flock()).
_LOCK = threading.Lock()


class FeedbackStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        enriched = {
            "received_at": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        line = json.dumps(enriched, ensure_ascii=False) + "\n"
        with _LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                # POSIX advisory lock survives across uvicorn workers (separate
                # OS processes). Without this, concurrent writes from multiple
                # workers can produce interleaved JSONL lines.
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.write(line)
                    handle.flush()
                finally:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return enriched

    def read_all(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with _LOCK, self.path.open("r", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                lines = handle.readlines()
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        records: list[dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if limit is not None:
            records = records[-limit:]
        return records

    def stats(self) -> dict[str, Any]:
        records = self.read_all()
        total = len(records)
        up = sum(1 for r in records if r.get("rating") == "up")
        down = sum(1 for r in records if r.get("rating") == "down")
        with_comment = sum(1 for r in records if (r.get("comment") or "").strip())
        return {
            "total": total,
            "thumbs_up": up,
            "thumbs_down": down,
            "with_comment": with_comment,
            "approval_rate": round(up / total, 4) if total else 0.0,
        }
