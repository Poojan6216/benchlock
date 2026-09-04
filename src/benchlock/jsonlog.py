"""Structured JSON logging to stderr.

stdout carries the product of a command (a verdict block, a report, JSON); stderr
carries the diagnostics. Keeping them apart is what lets `benchlock report > pr.md`
work in a CI job without the log lines landing in the artifact.

No telemetry: this writes to your terminal and nowhere else (Hard Rule 11).
"""

from __future__ import annotations

import json
import os
import sys
import time
from enum import IntEnum
from typing import Any, TextIO


class Level(IntEnum):
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40


_LEVEL_NAMES = {Level.DEBUG: "debug", Level.INFO: "info", Level.WARN: "warn", Level.ERROR: "error"}


class Logger:
    """A deliberately small structured logger. No handlers, no global config."""

    def __init__(
        self,
        stream: TextIO | None = None,
        level: Level = Level.INFO,
        *,
        timestamps: bool = True,
    ) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self.level = level
        self._timestamps = timestamps

    def bind(self, stream: TextIO) -> Logger:
        return Logger(stream, self.level, timestamps=self._timestamps)

    def log(self, level: Level, event: str, **fields: Any) -> None:
        if level < self.level:
            return
        record: dict[str, Any] = {"level": _LEVEL_NAMES[level], "event": event}
        if self._timestamps:
            record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        record.update(fields)
        print(json.dumps(record, default=str, sort_keys=False), file=self._stream, flush=True)

    def debug(self, event: str, **fields: Any) -> None:
        self.log(Level.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self.log(Level.INFO, event, **fields)

    def warn(self, event: str, **fields: Any) -> None:
        self.log(Level.WARN, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self.log(Level.ERROR, event, **fields)


def _level_from_env() -> Level:
    name = os.environ.get("BENCHLOCK_LOG_LEVEL", "info").strip().lower()
    return {"debug": Level.DEBUG, "info": Level.INFO, "warn": Level.WARN, "error": Level.ERROR}.get(
        name, Level.INFO
    )


#: Process-wide logger. Rebindable in tests; never reads or writes anything but stderr.
log = Logger(level=_level_from_env(), timestamps=os.environ.get("BENCHLOCK_LOG_TS") != "0")
