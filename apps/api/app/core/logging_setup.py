"""Structured (JSON-lines) logging with per-request correlation.

This module is the single place stdlib ``logging`` gets wired in
production. It does three things:

1. Replaces the default text formatter with a JSON formatter so every
   log line is a structured record. Log aggregators (Datadog, Loki,
   Cloudwatch Insights) treat each line as a queryable document.

2. Threads a per-request ``request_id`` through every record via a
   ``contextvars.ContextVar``. The middleware in ``main.py`` sets the
   id at request start and adds it to the response headers so a user
   can quote it back when reporting a bug. Background threads spawned
   from ``concurrent.futures`` inherit the var automatically; the SSE
   streaming worker thread captures and restores it explicitly.

3. Exposes a ``log_event`` helper for the workflow to emit one record
   per pipeline stage with timing and status in structured fields,
   rather than ad-hoc ``logger.info("scope done in %.2fs ok")`` text.

Usage:
    from app.core.logging_setup import log_event, get_request_id
    log_event(logger, "retriever", duration_ms=42.1, citations=12)
"""

from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid
from typing import Any


# Single contextvar for the current request id. Default "-" makes it
# obvious in logs when something runs outside any request scope
# (startup, background tasks, eval scripts).
_REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def set_request_id(rid: str | None) -> str:
    rid = rid or new_request_id()
    _REQUEST_ID.set(rid)
    return rid


def get_request_id() -> str:
    return _REQUEST_ID.get()


# Fields the JSON formatter promotes to top level when a LogRecord has
# them set via ``logger.info(..., extra={...})``. Anything else lands
# under ``extra``.
_STRUCTURED_FIELDS = frozenset({
    "stage", "duration_ms", "status", "error_type",
    "citations", "candidates", "tokens_in", "tokens_out",
    "model", "provider", "degraded",
})


class JsonFormatter(logging.Formatter):
    """Render each LogRecord as one JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", None) or _REQUEST_ID.get(),
        }
        extra_bag: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key in _STRUCTURED_FIELDS:
                payload[key] = value
            elif key.startswith("ev_"):
                # Workflow stage events use the ``ev_`` prefix so they
                # don't collide with stdlib LogRecord field names.
                payload[key[3:]] = value
        if extra_bag:
            payload["extra"] = extra_bag
        if record.exc_info:
            payload["exc_type"] = (
                record.exc_info[0].__name__ if record.exc_info[0] else None
            )
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    """Install the JSON formatter on the root logger.

    Idempotent: a second call is a no-op so reload-style tooling
    (uvicorn --reload, pytest fixtures) doesn't stack handlers.
    """
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler.formatter, JsonFormatter):
            return  # already set up
    # Strip default handlers so we don't double-emit (one JSON + one text).
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    # Quiet the noisier libraries; uvicorn's access log stays separate.
    for noisy in ("httpx", "urllib3", "asyncio", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(
    logger_obj: logging.Logger,
    stage: str,
    *,
    status: str = "ok",
    duration_ms: float | None = None,
    error_type: str | None = None,
    **fields: Any,
) -> None:
    """Emit one structured INFO record for a workflow stage.

    ``fields`` is forwarded to the formatter as top-level JSON keys
    when the field name is in ``_STRUCTURED_FIELDS``; otherwise prefix
    the field with ``ev_`` (e.g. ``ev_top_score=0.84``).
    """
    extras: dict[str, Any] = {"stage": stage, "status": status}
    if duration_ms is not None:
        extras["duration_ms"] = round(float(duration_ms), 2)
    if error_type:
        extras["error_type"] = error_type
    for key, value in fields.items():
        extras[key if key in _STRUCTURED_FIELDS else f"ev_{key}"] = value
    logger_obj.info(stage, extra=extras)
