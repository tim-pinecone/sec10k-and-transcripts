"""Resumability and observability primitives.

Every long loop in this project goes through these two classes, because a 13.6M-record
embedding job will be interrupted — by a 429, an OOM, a closed laptop — and restarting
from zero means paying the OpenAI bill twice.

- `Checkpoint` records completed units so a re-run skips them.
- `Progress` prints flushed position/rate/ETA and mirrors it to a tailable status file.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write to a temp file in the same directory, then rename.

    A crash mid-write leaves the old file intact rather than a truncated one that a
    resume would mistake for complete work.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Yield a temp path to write into; rename it over `path` on clean exit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class Checkpoint:
    """A manifest of completed work units, keyed by a caller-chosen string.

    Units are whatever is cheap to redo and expensive to lose — a dataset shard, a
    year, one staged parquet file. Keep the unit small enough that losing one is
    annoying rather than painful.
    """

    def __init__(self, name: str, state_dir: Path, flush_every: int = 20) -> None:
        self.path = Path(state_dir) / f"{name}.checkpoint.json"
        self.flush_every = flush_every
        self._done: dict[str, Any] = {}
        self._pending = 0
        if self.path.exists():
            try:
                self._done = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                # A corrupt checkpoint must not silently reset progress to zero.
                raise SystemExit(
                    f"Checkpoint {self.path} is corrupt. Inspect it, then either "
                    f"repair or delete it (deleting means redoing all work)."
                )

    def __len__(self) -> int:
        return len(self._done)

    def is_done(self, key: str) -> bool:
        return key in self._done

    def pending(self, keys: list[str]) -> list[str]:
        """The subset of `keys` not yet completed, in the given order."""
        return [k for k in keys if k not in self._done]

    def mark(self, key: str, **info: Any) -> None:
        self._done[key] = info or True
        self._pending += 1
        if self._pending >= self.flush_every:
            self.flush()

    def info(self, key: str) -> Any:
        return self._done.get(key)

    def totals(self, field: str) -> int:
        """Sum a numeric field recorded across all completed units."""
        return sum(
            v.get(field, 0) for v in self._done.values() if isinstance(v, dict)
        )

    def flush(self) -> None:
        atomic_write_bytes(self.path, json.dumps(self._done, indent=1).encode())
        self._pending = 0


class Progress:
    """Flushed live progress with a denominator, rate, and ETA, plus a status file.

    Deliberately not just tqdm: background and redirected runs need something to
    `cat`, and block-buffered stdout makes a healthy job look hung.
    """

    def __init__(
        self,
        label: str,
        total: int,
        status_path: Path | None = None,
        done: int = 0,
        report_every: float = 2.0,
        stream: Any = None,
    ) -> None:
        self.label = label
        self.total = total
        self.done = done
        self.resumed_at = done
        self.status_path = Path(status_path) if status_path else None
        self.report_every = report_every
        self.stream = stream or sys.stderr
        self.started = time.time()
        self._last_report = 0.0
        self.current = ""
        self.extra: dict[str, Any] = {}

    def advance(self, n: int = 1, current: str = "", **extra: Any) -> None:
        self.done += n
        if current:
            self.current = current
        self.extra.update(extra)
        if time.time() - self._last_report >= self.report_every:
            self.report()

    def _stats(self) -> tuple[float, float]:
        elapsed = max(time.time() - self.started, 1e-9)
        processed = self.done - self.resumed_at
        rate = processed / elapsed
        remaining = max(self.total - self.done, 0)
        eta = remaining / rate if rate > 0 else float("inf")
        return rate, eta

    def report(self) -> None:
        rate, eta = self._stats()
        pct = (self.done / self.total * 100) if self.total else 100.0
        bits = [
            self.label,
            f"{self.done:,}/{self.total:,} ({pct:.0f}%)",
            f"{rate:.1f}/s",
            f"ETA {_fmt_duration(eta)}",
        ]
        if self.current:
            bits.append(self.current)
        if self.extra:
            bits.append(" ".join(f"{k}={v}" for k, v in self.extra.items()))
        print(" · ".join(bits), file=self.stream, flush=True)
        self._write_status(rate, eta)
        self._last_report = time.time()

    def _write_status(self, rate: float, eta: float) -> None:
        if not self.status_path:
            return
        payload = {
            "label": self.label,
            "done": self.done,
            "total": self.total,
            "pct": round(self.done / self.total * 100, 2) if self.total else 100.0,
            "rate_per_sec": round(rate, 3),
            "eta_seconds": None if eta == float("inf") else round(eta),
            "current": self.current,
            "elapsed_seconds": round(time.time() - self.started),
            "resumed_from": self.resumed_at,
            **self.extra,
        }
        atomic_write_bytes(self.status_path, json.dumps(payload, indent=1).encode())

    def finish(self, summary: str = "") -> None:
        self.report()
        elapsed = _fmt_duration(time.time() - self.started)
        line = f"{self.label} · done {self.done:,}/{self.total:,} in {elapsed}"
        if summary:
            line += f" · {summary}"
        print(line, file=self.stream, flush=True)


def _fmt_duration(seconds: float) -> str:
    if seconds == float("inf"):
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
