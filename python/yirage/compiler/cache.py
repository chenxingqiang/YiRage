# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Compile Cache Management

Provides persistent caching for compiled kernels to avoid redundant compilation.
"""

import os
import json
import hashlib
import shutil
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any, List
from pathlib import Path
import threading


@dataclass
class CacheEntry:
    """A single cache entry."""

    graph_hash: str
    backend: str
    created_at: float
    last_accessed: float
    access_count: int
    latency_ms: Optional[float]
    compile_time_seconds: float
    mlir_code: Optional[str]
    generated_code: Optional[str]
    metadata: Dict[str, Any]


class CompileCache:
    """
    Persistent cache for compiled kernels.

    Features:
    - Disk-based persistence
    - LRU eviction
    - Thread-safe operations
    - Statistics tracking

    Usage:
        cache = CompileCache()

        # Check cache
        entry = cache.get(graph_hash, backend)
        if entry:
            print("Cache hit!")
        else:
            # Compile and cache
            result = compile(graph)
            cache.put(graph_hash, backend, result)
    """

    DEFAULT_CACHE_DIR = os.path.expanduser("~/.yirage/compile_cache")
    MAX_CACHE_SIZE_MB = 1024  # 1GB default max
    MAX_ENTRIES = 10000

    _instance: Optional["CompileCache"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_size_mb: int = MAX_CACHE_SIZE_MB,
        max_entries: int = MAX_ENTRIES,
    ):
        self.cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR)
        self.max_size_mb = max_size_mb
        self.max_entries = max_entries

        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "entries").mkdir(exist_ok=True)
        (self.cache_dir / "code").mkdir(exist_ok=True)

        # In-memory index
        self._index: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

        # Statistics
        self._hits = 0
        self._misses = 0

        # Load index
        self._load_index()

    @classmethod
    def get_instance(cls, **kwargs) -> "CompileCache":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance

    def get(self, graph_hash: str, backend: str) -> Optional[CacheEntry]:
        """
        Get a cached entry.

        Args:
            graph_hash: Hash of the graph
            backend: Target backend

        Returns:
            CacheEntry if found, None otherwise
        """
        key = self._make_key(graph_hash, backend)

        with self._lock:
            if key in self._index:
                entry = self._index[key]
                entry.last_accessed = time.time()
                entry.access_count += 1
                self._hits += 1
                self._save_entry(entry)
                return entry

            self._misses += 1
            return None

    def put(
        self,
        graph_hash: str,
        backend: str,
        latency_ms: Optional[float] = None,
        compile_time_seconds: float = 0.0,
        mlir_code: Optional[str] = None,
        generated_code: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> CacheEntry:
        """
        Store a compiled result in cache.

        Args:
            graph_hash: Hash of the graph
            backend: Target backend
            latency_ms: Measured latency
            compile_time_seconds: Compilation time
            mlir_code: Generated MLIR (optional)
            generated_code: Generated target code (optional)
            metadata: Additional metadata

        Returns:
            Created CacheEntry
        """
        key = self._make_key(graph_hash, backend)
        now = time.time()

        entry = CacheEntry(
            graph_hash=graph_hash,
            backend=backend,
            created_at=now,
            last_accessed=now,
            access_count=1,
            latency_ms=latency_ms,
            compile_time_seconds=compile_time_seconds,
            mlir_code=mlir_code,
            generated_code=generated_code,
            metadata=metadata or {},
        )

        with self._lock:
            self._index[key] = entry
            self._save_entry(entry)

            # Evict if necessary
            self._evict_if_needed()

        return entry

    def remove(self, graph_hash: str, backend: str) -> bool:
        """Remove an entry from cache."""
        key = self._make_key(graph_hash, backend)

        with self._lock:
            if key in self._index:
                del self._index[key]
                entry_file = self.cache_dir / "entries" / f"{key}.json"
                code_file = self.cache_dir / "code" / f"{key}.code"
                entry_file.unlink(missing_ok=True)
                code_file.unlink(missing_ok=True)
                return True
            return False

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._index.clear()

            # Remove all files
            shutil.rmtree(self.cache_dir / "entries", ignore_errors=True)
            shutil.rmtree(self.cache_dir / "code", ignore_errors=True)

            # Recreate directories
            (self.cache_dir / "entries").mkdir(exist_ok=True)
            (self.cache_dir / "code").mkdir(exist_ok=True)

            # Reset stats
            self._hits = 0
            self._misses = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / max(1, total_requests)

        # Calculate size
        total_size = 0
        for f in (self.cache_dir / "entries").glob("*.json"):
            total_size += f.stat().st_size
        for f in (self.cache_dir / "code").glob("*.code"):
            total_size += f.stat().st_size

        return {
            "entries": len(self._index),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size_mb": total_size / (1024 * 1024),
            "max_size_mb": self.max_size_mb,
        }

    def list_entries(self, backend: Optional[str] = None) -> List[CacheEntry]:
        """List all cache entries, optionally filtered by backend."""
        with self._lock:
            entries = list(self._index.values())

        if backend:
            entries = [e for e in entries if e.backend == backend]

        return sorted(entries, key=lambda e: e.last_accessed, reverse=True)

    def _make_key(self, graph_hash: str, backend: str) -> str:
        """Create a cache key from graph hash and backend."""
        return f"{backend}_{graph_hash}"

    def _load_index(self):
        """Load index from disk."""
        entries_dir = self.cache_dir / "entries"

        for entry_file in entries_dir.glob("*.json"):
            try:
                with open(entry_file) as f:
                    data = json.load(f)
                entry = CacheEntry(**data)
                key = self._make_key(entry.graph_hash, entry.backend)
                self._index[key] = entry
            except Exception:
                # Skip invalid entries
                pass

    def _save_entry(self, entry: CacheEntry):
        """Save entry to disk."""
        key = self._make_key(entry.graph_hash, entry.backend)
        entry_file = self.cache_dir / "entries" / f"{key}.json"

        # Save entry metadata
        data = asdict(entry)
        # Don't store large code in JSON
        data["mlir_code"] = None
        data["generated_code"] = None

        with open(entry_file, "w") as f:
            json.dump(data, f)

        # Save code separately
        if entry.generated_code:
            code_file = self.cache_dir / "code" / f"{key}.code"
            with open(code_file, "w") as f:
                f.write(entry.generated_code)

    def _evict_if_needed(self):
        """Evict oldest entries if cache is too large."""
        if len(self._index) <= self.max_entries:
            return

        # Sort by last_accessed, evict oldest
        sorted_entries = sorted(self._index.items(), key=lambda x: x[1].last_accessed)

        num_to_evict = len(self._index) - self.max_entries + 100  # Evict extra

        for key, entry in sorted_entries[:num_to_evict]:
            del self._index[key]
            entry_file = self.cache_dir / "entries" / f"{key}.json"
            code_file = self.cache_dir / "code" / f"{key}.code"
            entry_file.unlink(missing_ok=True)
            code_file.unlink(missing_ok=True)


# Convenience functions


def get_compile_cache(**kwargs) -> CompileCache:
    """Get the global compile cache instance."""
    return CompileCache.get_instance(**kwargs)


def clear_compile_cache():
    """Clear all cached compilations."""
    cache = get_compile_cache()
    cache.clear()
