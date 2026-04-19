"""Atomic file-write helpers.

The classic on-disk corruption failure mode is a naive ``open(path, "w")``
truncate-and-write: if the process dies between the truncate and the
completion of the write, the destination file is half-written or empty.
A subsequent load either raises (fail-closed, merely annoying) or silently
parses a partial result (fail-open, silent data loss).

The canonical idiom to avoid this is:

    write → flush → fsync → os.replace

Each step is load-bearing:

* **write** to a *temp file in the same directory* as the destination.
  "Same directory" matters — ``os.replace`` is only atomic when source and
  destination are on the same filesystem. Putting the temp in ``/tmp`` and
  renaming to ``/home`` silently downgrades to copy-then-unlink and
  re-opens the corruption window.
* **flush** empties the Python buffer into the OS page cache.
* **fsync** tells the OS to actually push the bytes to the storage device.
  Without it, a power loss between fsync-missing and the device's own
  flush still loses data even after ``os.replace`` "succeeded."
* **os.replace** is atomic on POSIX and Windows (Python 3.3+). The
  destination either exists as old content or new content — never as a
  partial write.

All writes in the persistence layer (corpus, chats, sessions, dream
logs) should go through these helpers.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp-and-rename).

    On failure, the temp file is best-effort removed so we don't leave
    ``.foo.yaml.abc123.tmp`` cruft around the tree.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temp filename so concurrent saves of the same path (rare but
    # possible, e.g. two API handlers racing) don't clobber each other's
    # in-progress write — they each get their own temp and the last
    # os.replace wins (fine, YAMLs are idempotent final-state snapshots).
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_yaml(path: Path, data: Any) -> None:
    """Serialize ``data`` as YAML and write atomically.

    Uses the same ``yaml.dump`` options as the corpus/session layer:
    block style, unicode preserved, key order preserved (not sorted).
    """
    text = yaml.dump(
        data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    atomic_write_text(path, text)


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Serialize ``data`` as JSON and write atomically.

    Uses ``default=str`` so pydantic datetimes / Path objects serialize
    without an explicit pre-dump step — matches existing chat/session
    writer behavior.
    """
    text = json.dumps(
        data,
        indent=indent,
        default=str,
        ensure_ascii=False,
    )
    atomic_write_text(path, text)
