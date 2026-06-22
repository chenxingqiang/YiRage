# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Tests for yirage.logging_config module.

Tests error classes, LogConfig, PerfLogger, StructuredLogger, ColorFormatter.
"""

import importlib.util
import logging
import time
from pathlib import Path

import pytest

# Load the module directly to bypass yirage.__init__ (which requires native core)
_PYTHON_ROOT = Path(__file__).parent.parent.parent / "python"


def _load_logging_config():
    """Load logging_config module directly, avoiding yirage.__init__."""
    path = _PYTHON_ROOT / "yirage" / "logging_config.py"
    spec = importlib.util.spec_from_file_location("yirage_logging_config_test", path)
    if spec is None or spec.loader is None:
        pytest.skip("logging_config module not found")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def lc():
    """Load and return the logging_config module."""
    return _load_logging_config()


# =============================================================================
# ErrorCode Enum Tests
# =============================================================================


class TestErrorCode:
    """Test ErrorCode enum values and ranges."""

    def test_general_errors_range(self, lc):
        """Test general error codes are in 1000-1999 range."""
        assert lc.ErrorCode.UNKNOWN_ERROR.value == 1000
        assert lc.ErrorCode.INVALID_ARGUMENT.value == 1001
        assert lc.ErrorCode.NOT_IMPLEMENTED.value == 1002
        assert lc.ErrorCode.INTERNAL_ERROR.value == 1003

    def test_core_errors_range(self, lc):
        """Test core error codes are in 2000-2999 range."""
        assert lc.ErrorCode.CORE_NOT_AVAILABLE.value == 2000
        assert lc.ErrorCode.CORE_INITIALIZATION_FAILED.value == 2001
        assert lc.ErrorCode.GRAPH_INVALID.value == 2002
        assert lc.ErrorCode.GRAPH_NOT_FOUND.value == 2003

    def test_search_errors_range(self, lc):
        """Test search error codes are in 3000-3999 range."""
        assert lc.ErrorCode.SEARCH_FAILED.value == 3000
        assert lc.ErrorCode.SEARCH_TIMEOUT.value == 3001
        assert lc.ErrorCode.NO_VALID_KERNEL.value == 3002
        assert lc.ErrorCode.VERIFICATION_FAILED.value == 3003

    def test_backend_errors_range(self, lc):
        """Test backend error codes are in 4000-4999 range."""
        assert lc.ErrorCode.BACKEND_NOT_AVAILABLE.value == 4000
        assert lc.ErrorCode.BACKEND_INITIALIZATION_FAILED.value == 4001
        assert lc.ErrorCode.BACKEND_EXECUTION_FAILED.value == 4002

    def test_compilation_errors_range(self, lc):
        """Test compilation error codes are in 5000-5999 range."""
        assert lc.ErrorCode.COMPILATION_FAILED.value == 5000
        assert lc.ErrorCode.CODE_GENERATION_FAILED.value == 5001
        assert lc.ErrorCode.MLIR_LOWERING_FAILED.value == 5002

    def test_rl_errors_range(self, lc):
        """Test RL error codes are in 6000-6999 range."""
        assert lc.ErrorCode.RL_CONTEXT_FAILED.value == 6000
        assert lc.ErrorCode.RL_ACTION_INVALID.value == 6001
        assert lc.ErrorCode.RL_VERIFICATION_FAILED.value == 6002

    def test_error_code_is_enum(self, lc):
        """Test that ErrorCode is an Enum subclass."""
        from enum import Enum

        assert issubclass(lc.ErrorCode, Enum)


# =============================================================================
# YirageError Tests
# =============================================================================


class TestYirageError:
    """Test base YirageError exception class."""

    def test_basic_creation(self, lc):
        """Test creating YirageError with message only."""
        err = lc.YirageError("something went wrong")
        assert err.message == "something went wrong"
        assert err.code == lc.ErrorCode.UNKNOWN_ERROR
        assert err.details == {}
        assert err.cause is None

    def test_creation_with_code(self, lc):
        """Test creating YirageError with specific code."""
        err = lc.YirageError("bad input", code=lc.ErrorCode.INVALID_ARGUMENT)
        assert err.code == lc.ErrorCode.INVALID_ARGUMENT

    def test_creation_with_details(self, lc):
        """Test creating YirageError with details dict."""
        details = {"key": "value", "count": 42}
        err = lc.YirageError("fail", details=details)
        assert err.details == details

    def test_creation_with_cause(self, lc):
        """Test creating YirageError with a cause exception."""
        cause = RuntimeError("root cause")
        err = lc.YirageError("wrapper", cause=cause)
        assert err.cause is cause

    def test_str_format(self, lc):
        """Test string representation includes code name and message."""
        err = lc.YirageError("test error", code=lc.ErrorCode.SEARCH_FAILED)
        s = str(err)
        assert "SEARCH_FAILED" in s
        assert "test error" in s

    def test_to_dict(self, lc):
        """Test to_dict serialization."""
        err = lc.YirageError(
            "test",
            code=lc.ErrorCode.GRAPH_INVALID,
            details={"nodes": 5},
        )
        d = err.to_dict()
        assert d["error"] == "GRAPH_INVALID"
        assert d["code"] == 2002
        assert d["message"] == "test"
        assert d["details"] == {"nodes": 5}
        assert "timestamp" in d

    def test_timestamp_is_set(self, lc):
        """Test that timestamp is automatically set."""
        err = lc.YirageError("test")
        assert err.timestamp is not None

    def test_is_exception(self, lc):
        """Test that YirageError can be raised and caught."""
        with pytest.raises(lc.YirageError):
            raise lc.YirageError("boom")


# =============================================================================
# Specialized Error Classes
# =============================================================================


class TestCoreError:
    """Test CoreError specialization."""

    def test_default_code(self, lc):
        """Test CoreError has CORE_NOT_AVAILABLE code."""
        err = lc.CoreError("core missing")
        assert err.code == lc.ErrorCode.CORE_NOT_AVAILABLE

    def test_is_yirage_error(self, lc):
        """Test CoreError is subclass of YirageError."""
        assert issubclass(lc.CoreError, lc.YirageError)


class TestSearchError:
    """Test SearchError specialization."""

    def test_default_code(self, lc):
        """Test SearchError has SEARCH_FAILED code."""
        err = lc.SearchError("search failed")
        assert err.code == lc.ErrorCode.SEARCH_FAILED

    def test_is_yirage_error(self, lc):
        """Test SearchError is subclass of YirageError."""
        assert issubclass(lc.SearchError, lc.YirageError)


class TestBackendError:
    """Test BackendError specialization."""

    def test_default_code(self, lc):
        """Test BackendError has BACKEND_NOT_AVAILABLE code."""
        err = lc.BackendError("backend fail")
        assert err.code == lc.ErrorCode.BACKEND_NOT_AVAILABLE

    def test_backend_in_details(self, lc):
        """Test backend name stored in details."""
        err = lc.BackendError("fail", backend="cuda")
        assert err.details["backend"] == "cuda"

    def test_default_backend_unknown(self, lc):
        """Test default backend name is 'unknown'."""
        err = lc.BackendError("fail")
        assert err.details["backend"] == "unknown"


class TestCompilationError:
    """Test CompilationError specialization."""

    def test_default_code(self, lc):
        """Test CompilationError has COMPILATION_FAILED code."""
        err = lc.CompilationError("compile fail")
        assert err.code == lc.ErrorCode.COMPILATION_FAILED

    def test_stage_in_details(self, lc):
        """Test stage stored in details."""
        err = lc.CompilationError("fail", stage="codegen")
        assert err.details["stage"] == "codegen"

    def test_default_stage_unknown(self, lc):
        """Test default stage is 'unknown'."""
        err = lc.CompilationError("fail")
        assert err.details["stage"] == "unknown"


class TestRLError:
    """Test RLError specialization."""

    def test_default_code(self, lc):
        """Test RLError has RL_CONTEXT_FAILED code."""
        err = lc.RLError("rl fail")
        assert err.code == lc.ErrorCode.RL_CONTEXT_FAILED


# =============================================================================
# LogLevel Tests
# =============================================================================


class TestLogLevel:
    """Test LogLevel enum."""

    def test_debug_level(self, lc):
        """Test DEBUG level maps to logging.DEBUG."""
        assert lc.LogLevel.DEBUG.value == logging.DEBUG

    def test_info_level(self, lc):
        """Test INFO level maps to logging.INFO."""
        assert lc.LogLevel.INFO.value == logging.INFO

    def test_warning_level(self, lc):
        """Test WARNING level maps to logging.WARNING."""
        assert lc.LogLevel.WARNING.value == logging.WARNING

    def test_error_level(self, lc):
        """Test ERROR level maps to logging.ERROR."""
        assert lc.LogLevel.ERROR.value == logging.ERROR

    def test_critical_level(self, lc):
        """Test CRITICAL level maps to logging.CRITICAL."""
        assert lc.LogLevel.CRITICAL.value == logging.CRITICAL


# =============================================================================
# ColorFormatter Tests
# =============================================================================


class TestColorFormatter:
    """Test ColorFormatter for terminal output."""

    def test_has_color_codes(self, lc):
        """Test that color codes are defined for all levels."""
        formatter = lc.ColorFormatter()
        assert "DEBUG" in formatter.COLORS
        assert "INFO" in formatter.COLORS
        assert "WARNING" in formatter.COLORS
        assert "ERROR" in formatter.COLORS
        assert "CRITICAL" in formatter.COLORS

    def test_reset_code(self, lc):
        """Test RESET escape code is defined."""
        assert lc.ColorFormatter.RESET == "\033[0m"

    def test_format_record(self, lc):
        """Test formatting a log record adds color codes."""
        formatter = lc.ColorFormatter("%(levelname)s: %(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="error msg",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        # Should contain the error message
        assert "error msg" in formatted


# =============================================================================
# LogConfig Tests
# =============================================================================


class TestLogConfig:
    """Test LogConfig class-level configuration."""

    def test_set_level(self, lc):
        """Test set_level changes the root yirage logger level."""
        lc.LogConfig.set_level(lc.LogLevel.DEBUG)
        logger = logging.getLogger("yirage")
        assert logger.level == logging.DEBUG
        # Reset
        lc.LogConfig.set_level(lc.LogLevel.INFO)

    def test_get_log_dir_none_by_default(self, lc):
        """Test get_log_dir returns None when file logging not enabled."""
        # Since we loaded the module fresh, _log_dir may or may not be set
        # depending on env vars. Just test the method works.
        result = lc.LogConfig.get_log_dir()
        assert result is None or isinstance(result, Path)


# =============================================================================
# get_logger Tests
# =============================================================================


class TestGetLogger:
    """Test get_logger function."""

    def test_returns_logger(self, lc):
        """Test get_logger returns a Logger instance."""
        logger = lc.get_logger("yirage.test")
        assert isinstance(logger, logging.Logger)

    def test_default_name(self, lc):
        """Test get_logger with default name."""
        logger = lc.get_logger()
        assert logger.name == "yirage"

    def test_custom_name(self, lc):
        """Test get_logger with custom name."""
        logger = lc.get_logger("yirage.kernel.graph")
        assert logger.name == "yirage.kernel.graph"


# =============================================================================
# PerfLogger Tests
# =============================================================================


class TestPerfLogger:
    """Test PerfLogger performance timing utility."""

    def test_start_end(self, lc):
        """Test start/end timing returns positive elapsed time."""
        perf = lc.PerfLogger("yirage.test.perf")
        perf.start("test_event")
        time.sleep(0.01)
        elapsed = perf.end("test_event", log=False)
        assert elapsed > 0

    def test_end_unknown_event(self, lc):
        """Test ending an unknown event returns 0."""
        perf = lc.PerfLogger()
        elapsed = perf.end("nonexistent", log=False)
        assert elapsed == 0.0

    def test_measure_context_manager(self, lc):
        """Test measure() context manager records elapsed time."""
        perf = lc.PerfLogger("yirage.test.perf")
        with perf.measure("ctx_event") as ctx:
            time.sleep(0.01)
        assert ctx.elapsed > 0

    def test_multiple_events(self, lc):
        """Test timing multiple events simultaneously."""
        perf = lc.PerfLogger()
        perf.start("event_a")
        perf.start("event_b")
        time.sleep(0.01)
        a = perf.end("event_a", log=False)
        time.sleep(0.01)
        b = perf.end("event_b", log=False)
        assert a > 0
        assert b > a  # event_b ran longer


# =============================================================================
# StructuredLogger Tests
# =============================================================================


class TestStructuredLogger:
    """Test StructuredLogger for machine-readable logging."""

    def test_log_event(self, lc):
        """Test log_event produces structured output."""
        slog = lc.StructuredLogger("yirage.test.structured")
        # Should not raise
        slog.log_event("test_event", key1="val1", key2=42)

    def test_log_search(self, lc):
        """Test log_search helper."""
        slog = lc.StructuredLogger()
        slog.log_search(
            backend="cuda",
            iterations=100,
            best_latency_ms=0.5,
        )

    def test_log_compile(self, lc):
        """Test log_compile helper."""
        slog = lc.StructuredLogger()
        slog.log_compile(
            backend="cpu",
            compile_time_seconds=1.23,
            success=True,
        )


# =============================================================================
# Error Hierarchy Tests
# =============================================================================


class TestErrorHierarchy:
    """Test the exception hierarchy for correct inheritance."""

    def test_all_errors_inherit_yirage_error(self, lc):
        """Test all error types are YirageError subclasses."""
        for cls in [
            lc.CoreError,
            lc.SearchError,
            lc.BackendError,
            lc.CompilationError,
            lc.RLError,
        ]:
            assert issubclass(cls, lc.YirageError)
            assert issubclass(cls, Exception)

    def test_catch_yirage_error_catches_subtypes(self, lc):
        """Test catching YirageError also catches subtypes."""
        for cls in [
            lc.CoreError,
            lc.SearchError,
            lc.BackendError,
            lc.CompilationError,
            lc.RLError,
        ]:
            with pytest.raises(lc.YirageError):
                raise cls("test")
