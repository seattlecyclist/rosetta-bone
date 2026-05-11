"""structlog configuration. Call configure_logging() once at CLI entry.

At the default INFO level, this suppresses chatty third-party loggers
(httpx, huggingface_hub, sentence_transformers, transformers, etc.) so
the user only sees Rosetta Bone's own structured events plus genuine
warnings. Pass `--verbose` / `-v` on the CLI (or `level="DEBUG"`
programmatically) to re-enable all of it for debugging.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

import structlog

# Third-party loggers that emit INFO-level chatter we don't want by default.
# Each download or HTTP request from these libraries produces a line.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
    "urllib3",
    "datasets",
    "filelock",
    "fsspec",
    "PIL",
)

# Warning messages we always want to suppress (the API rename and the
# Hub anonymous-access nag don't carry actionable information).
_NOISY_WARNINGS = (
    r".*get_sentence_embedding_dimension.*",
    r".*unauthenticated requests.*",
)


def _quiet_third_party(level: str) -> None:
    """Silence noisy third-party loggers + warnings unless level is DEBUG."""
    if level.upper() == "DEBUG":
        return

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    for pat in _NOISY_WARNINGS:
        warnings.filterwarnings("ignore", message=pat)

    # These knobs must be set before transformers / huggingface_hub are
    # imported. configure_logging() runs in the Typer root callback, so
    # we get there before any subcommand body triggers those imports.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def configure_logging(level: str = "INFO") -> None:
    _quiet_third_party(level)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper()),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
