# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Hardware Registry — singleton that stores :class:`ChipArchitecture` instances.

Usage::

    from yirage.hardware import HardwareRegistry

    reg = HardwareRegistry.instance()
    reg.register(my_chip)                  # register a new chip
    chip = reg.get("nvidia_h100")          # look up by chip_id
    chips = reg.list_by_vendor("nvidia")   # list all NVIDIA chips
"""

from __future__ import annotations

import json
import threading
from enum import Enum
from pathlib import Path
from typing import Callable

from .chip_arch import ChipArchitecture


class HardwareRegistry:
    """Thread-safe singleton registry for chip architectures."""

    _instance: HardwareRegistry | None = None
    _lock = threading.Lock()

    # ---------------------------------------------------------------- singleton

    def __init__(self) -> None:
        # Prevent direct construction; use .instance()
        self._chips: dict[str, ChipArchitecture] = {}
        self._backend_index: dict[str, list[str]] = {}
        self._vendor_index: dict[str, list[str]] = {}
        self._category_index: dict[str, list[str]] = {}
        self._callbacks: list[Callable[[ChipArchitecture], None]] = []
        self._mutex = threading.Lock()

    @classmethod
    def instance(cls) -> HardwareRegistry:
        """Return the global singleton (created on first call)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton — mainly useful in tests."""
        with cls._lock:
            cls._instance = None

    # ----------------------------------------------------------- registration

    def register(
        self,
        chip: ChipArchitecture,
        *,
        overwrite: bool = False,
    ) -> bool:
        """
        Register a chip architecture.

        Args:
            chip: The :class:`ChipArchitecture` to register.
            overwrite: If *True*, silently replace an existing entry with
                the same ``chip_id``.  Otherwise return *False* if the id
                already exists.

        Returns:
            *True* if the chip was registered, *False* if it already existed
            (and ``overwrite`` was *False*).
        """
        if not chip.chip_id:
            raise ValueError("chip.chip_id must be a non-empty string")

        with self._mutex:
            if chip.chip_id in self._chips and not overwrite:
                return False

            self._chips[chip.chip_id] = chip

            # Update indices
            self._backend_index.setdefault(chip.backend, [])
            if chip.chip_id not in self._backend_index[chip.backend]:
                self._backend_index[chip.backend].append(chip.chip_id)

            # Use Enum.value (not str(enum)): on Python <3.11, str(ChipVendor.X) is
            # "ChipVendor.X", which breaks list_by_vendor after cross-import copies.
            vendor_key = (
                str(chip.vendor.value).lower()
                if isinstance(chip.vendor, Enum)
                else str(chip.vendor).lower()
            )
            self._vendor_index.setdefault(vendor_key, [])
            if chip.chip_id not in self._vendor_index[vendor_key]:
                self._vendor_index[vendor_key].append(chip.chip_id)

            cat_key = (
                str(chip.category.value).lower()
                if isinstance(chip.category, Enum)
                else str(chip.category).lower()
            )
            self._category_index.setdefault(cat_key, [])
            if chip.chip_id not in self._category_index[cat_key]:
                self._category_index[cat_key].append(chip.chip_id)

        # Fire callbacks (outside lock)
        for cb in self._callbacks:
            try:
                cb(chip)
            except Exception:
                pass  # best-effort

        return True

    def unregister(self, chip_id: str) -> bool:
        """Remove a chip by id.  Returns *True* if found."""
        with self._mutex:
            chip = self._chips.pop(chip_id, None)
            if chip is None:
                return False
            # Clean indices
            for idx in (self._backend_index, self._vendor_index, self._category_index):
                for lst in idx.values():
                    if chip_id in lst:
                        lst.remove(chip_id)
            return True

    # ---------------------------------------------------------------- queries

    def get(self, chip_id: str) -> ChipArchitecture | None:
        """Look up a chip by its unique id."""
        with self._mutex:
            return self._chips.get(chip_id)

    def __contains__(self, chip_id: str) -> bool:
        with self._mutex:
            return chip_id in self._chips

    def list_all(self) -> list[ChipArchitecture]:
        """Return all registered chips (snapshot)."""
        with self._mutex:
            return list(self._chips.values())

    def list_ids(self) -> list[str]:
        """Return all registered chip ids."""
        with self._mutex:
            return list(self._chips.keys())

    def list_by_backend(self, backend: str) -> list[ChipArchitecture]:
        """Return chips mapped to a specific YiRage backend name."""
        with self._mutex:
            ids = self._backend_index.get(backend, [])
            return [self._chips[cid] for cid in ids if cid in self._chips]

    def list_by_vendor(self, vendor: str) -> list[ChipArchitecture]:
        """Return chips from a specific vendor."""
        with self._mutex:
            key = vendor.lower()
            ids = self._vendor_index.get(key, [])
            return [self._chips[cid] for cid in ids if cid in self._chips]

    def list_by_category(self, category: str) -> list[ChipArchitecture]:
        """Return chips in a specific category (gpu, cpu, npu, …)."""
        with self._mutex:
            key = category.lower()
            ids = self._category_index.get(key, [])
            return [self._chips[cid] for cid in ids if cid in self._chips]

    @property
    def size(self) -> int:
        with self._mutex:
            return len(self._chips)

    # ------------------------------------------------------------ callbacks

    def on_register(self, callback: Callable[[ChipArchitecture], None]) -> None:
        """
        Subscribe to registration events.

        The *callback* is invoked **after** a chip is successfully registered
        (including overwrites).  Use this to react to new chip registrations.
        """
        self._callbacks.append(callback)

    # ------------------------------------------------------ import / export

    def export_json(self, path: str | None = None) -> str:
        """
        Serialise the whole registry to JSON.

        Args:
            path: If given, also write to this file path.

        Returns:
            JSON string.
        """
        with self._mutex:
            data = [chip.to_dict() for chip in self._chips.values()]

        text = json.dumps(data, indent=2, ensure_ascii=False)
        if path:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def import_json(self, source: str, *, overwrite: bool = False) -> int:
        """
        Load chips from a JSON string or file path.

        Args:
            source: Either a JSON string (starts with ``[``) or a file path.
            overwrite: Whether to replace existing entries.

        Returns:
            Number of chips imported.
        """
        if source.lstrip().startswith("["):
            data = json.loads(source)
        else:
            data = json.loads(Path(source).read_text(encoding="utf-8"))

        count = 0
        for item in data:
            chip = ChipArchitecture.from_dict(item)
            if self.register(chip, overwrite=overwrite):
                count += 1
        return count

    # ---------------------------------------------------------------- clear

    def clear(self) -> None:
        """Remove all registered chips."""
        with self._mutex:
            self._chips.clear()
            self._backend_index.clear()
            self._vendor_index.clear()
            self._category_index.clear()
