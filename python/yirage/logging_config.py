# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
YiRage Logging Configuration

Provides unified logging and error handling across all YiRage modules.

Usage:
    from yirage.logging_config import get_logger, YirageError
    
    logger = get_logger('yirage.kernel')
    logger.info("Starting optimization")
    
    try:
        result = optimize(graph)
    except YirageError as e:
        logger.error(f"Optimization failed: {e}")
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from enum import Enum


# ============================================================
# Error Classes
# ============================================================


class ErrorCode(Enum):
    """Standard error codes for YiRage."""

    # General errors (1000-1999)
    UNKNOWN_ERROR = 1000
    INVALID_ARGUMENT = 1001
    NOT_IMPLEMENTED = 1002
    INTERNAL_ERROR = 1003

    # Core errors (2000-2999)
    CORE_NOT_AVAILABLE = 2000
    CORE_INITIALIZATION_FAILED = 2001
    GRAPH_INVALID = 2002
    GRAPH_NOT_FOUND = 2003

    # Search errors (3000-3999)
    SEARCH_FAILED = 3000
    SEARCH_TIMEOUT = 3001
    NO_VALID_KERNEL = 3002
    VERIFICATION_FAILED = 3003

    # Backend errors (4000-4999)
    BACKEND_NOT_AVAILABLE = 4000
    BACKEND_INITIALIZATION_FAILED = 4001
    BACKEND_EXECUTION_FAILED = 4002

    # Compilation errors (5000-5999)
    COMPILATION_FAILED = 5000
    CODE_GENERATION_FAILED = 5001
    MLIR_LOWERING_FAILED = 5002

    # RL errors (6000-6999)
    RL_CONTEXT_FAILED = 6000
    RL_ACTION_INVALID = 6001
    RL_VERIFICATION_FAILED = 6002


class YirageError(Exception):
    """Base exception for all YiRage errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[dict] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.utcnow()

    def __str__(self) -> str:
        return f"[{self.code.name}] {self.message}"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "error": self.code.name,
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class CoreError(YirageError):
    """Error related to C++ core module."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, code=ErrorCode.CORE_NOT_AVAILABLE, **kwargs)


class SearchError(YirageError):
    """Error during kernel search."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, code=ErrorCode.SEARCH_FAILED, **kwargs)


class BackendError(YirageError):
    """Error related to backend operations."""

    def __init__(self, message: str, backend: str = "unknown", **kwargs):
        super().__init__(message, code=ErrorCode.BACKEND_NOT_AVAILABLE, **kwargs)
        self.details["backend"] = backend


class CompilationError(YirageError):
    """Error during compilation."""

    def __init__(self, message: str, stage: str = "unknown", **kwargs):
        super().__init__(message, code=ErrorCode.COMPILATION_FAILED, **kwargs)
        self.details["stage"] = stage


class RLError(YirageError):
    """Error in RL search."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, code=ErrorCode.RL_CONTEXT_FAILED, **kwargs)


# ============================================================
# Logger Configuration
# ============================================================


class LogLevel(Enum):
    """Log levels."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class ColorFormatter(logging.Formatter):
    """Colored log formatter for terminal output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class LogConfig:
    """Global logging configuration."""

    _initialized = False
    _log_dir: Optional[Path] = None
    _file_handler: Optional[logging.Handler] = None

    @classmethod
    def initialize(
        cls,
        level: LogLevel = LogLevel.INFO,
        log_to_file: bool = False,
        log_dir: Optional[str] = None,
        colored: bool = True,
    ):
        """
        Initialize logging configuration.

        Args:
            level: Minimum log level
            log_to_file: Whether to log to file
            log_dir: Directory for log files
            colored: Use colored output
        """
        if cls._initialized:
            return

        # Create root logger
        root_logger = logging.getLogger("yirage")
        root_logger.setLevel(level.value)

        # Console handler
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level.value)

        if colored and sys.stderr.isatty():
            formatter = ColorFormatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )

        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler
        if log_to_file:
            cls._log_dir = Path(log_dir or os.path.expanduser("~/.yirage/logs"))
            cls._log_dir.mkdir(parents=True, exist_ok=True)

            log_file = cls._log_dir / f"yirage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

            cls._file_handler = logging.FileHandler(log_file)
            cls._file_handler.setLevel(logging.DEBUG)  # File logs everything
            cls._file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
                )
            )
            root_logger.addHandler(cls._file_handler)

        cls._initialized = True

    @classmethod
    def set_level(cls, level: LogLevel):
        """Set global log level."""
        logging.getLogger("yirage").setLevel(level.value)

    @classmethod
    def get_log_dir(cls) -> Optional[Path]:
        """Get current log directory."""
        return cls._log_dir


def get_logger(name: str = "yirage") -> logging.Logger:
    """
    Get a logger for a module.

    Args:
        name: Logger name (e.g., 'yirage.kernel')

    Returns:
        Logger instance
    """
    # Ensure initialized
    if not LogConfig._initialized:
        LogConfig.initialize()

    return logging.getLogger(name)


# ============================================================
# Performance Logging
# ============================================================


class PerfLogger:
    """Performance logging utility."""

    def __init__(self, name: str = "yirage.perf"):
        self.logger = get_logger(name)
        self._timings = {}

    def start(self, event: str):
        """Start timing an event."""
        import time

        self._timings[event] = time.perf_counter()

    def end(self, event: str, log: bool = True) -> float:
        """
        End timing an event.

        Args:
            event: Event name
            log: Whether to log the result

        Returns:
            Elapsed time in seconds
        """
        import time

        if event not in self._timings:
            return 0.0

        elapsed = time.perf_counter() - self._timings[event]
        del self._timings[event]

        if log:
            self.logger.info(f"{event}: {elapsed*1000:.2f}ms")

        return elapsed

    def measure(self, event: str):
        """Context manager for timing."""

        class TimingContext:
            def __init__(ctx, perf_logger, event):
                ctx.perf_logger = perf_logger
                ctx.event = event

            def __enter__(ctx):
                ctx.perf_logger.start(ctx.event)
                return ctx

            def __exit__(ctx, *args):
                ctx.elapsed = ctx.perf_logger.end(ctx.event)

        return TimingContext(self, event)


# ============================================================
# Structured Logging
# ============================================================


class StructuredLogger:
    """Structured logging for machine-readable output."""

    def __init__(self, name: str = "yirage"):
        self.logger = get_logger(name)

    def log_event(self, event: str, level: LogLevel = LogLevel.INFO, **kwargs):
        """Log a structured event."""
        import json

        data = {"event": event, "timestamp": datetime.utcnow().isoformat(), **kwargs}

        self.logger.log(level.value, json.dumps(data))

    def log_search(self, backend: str, iterations: int, best_latency_ms: float, **kwargs):
        """Log search results."""
        self.log_event(
            "search_complete",
            backend=backend,
            iterations=iterations,
            best_latency_ms=best_latency_ms,
            **kwargs,
        )

    def log_compile(self, backend: str, compile_time_seconds: float, success: bool, **kwargs):
        """Log compilation results."""
        self.log_event(
            "compile_complete",
            backend=backend,
            compile_time_seconds=compile_time_seconds,
            success=success,
            **kwargs,
        )


# ============================================================
# Initialize on import
# ============================================================

# Check environment for log level
_env_level = os.environ.get("YIRAGE_LOG_LEVEL", "INFO").upper()
_default_level = getattr(LogLevel, _env_level, LogLevel.INFO)

# Check if file logging requested
_log_to_file = os.environ.get("YIRAGE_LOG_FILE", "").lower() in ("1", "true", "yes")

LogConfig.initialize(
    level=_default_level,
    log_to_file=_log_to_file,
)
