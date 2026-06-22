# Changelog

All notable changes to YiRage will be documented in this file.

## Unreleased

- Added this changelog as the canonical release-notes target referenced by
  package metadata.
- Clarified repository configuration and documentation inconsistencies found
  during project audit.
- Removed the broken `yirage` console-script entry that pointed at a
  non-existent `yirage.cli:main`, which previously installed an unusable
  `yirage` command.
- Aligned the `setup.py` package description with `pyproject.toml` so both
  surfaces note the native runtime requirement consistently.
- Added a regression test that every `[project.scripts]` entry in
  `pyproject.toml` resolves to a real, importable target.
