"""Structured logging. Never log embeddings or raw camera frames."""

from __future__ import annotations

import logging
import sys
from typing import Iterable

_BLOCKED_KEYS = {
    "embedding",
    "embeddings",
    "face_embedding",
    "body_embedding",
    "frame",
    "crop",
    "image",
    "pixels",
}


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, dict):
            record.args = {
                k: "<redacted>" if k.lower() in _BLOCKED_KEYS else v
                for k, v in record.args.items()
            }
        msg = str(record.getMessage())
        for key in _BLOCKED_KEYS:
            if key in msg.lower() and "embedding" in key:
                record.msg = "[redacted biometric field]"
                record.args = ()
                break
        return True


def setup_logging(level: int = logging.INFO, extra_filters: Iterable | None = None) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RedactingFilter())
    for filt in extra_filters or []:
        handler.addFilter(filt)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[handler],
        force=True,
    )
