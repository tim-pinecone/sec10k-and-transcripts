"""The resumability guarantee: kill it, re-run it, and completed work is skipped."""

import json

import pytest

from finvec.progress import Checkpoint, Progress, atomic_path, atomic_write_bytes


def test_checkpoint_skips_completed_work_after_restart(tmp_path):
    ck = Checkpoint("stage-sec", tmp_path)
    for shard in ["0", "1", "2"]:
        ck.mark(shard, records=100)
    ck.flush()

    # Simulate the process dying and the command being re-run.
    reloaded = Checkpoint("stage-sec", tmp_path)
    assert len(reloaded) == 3
    assert reloaded.pending(["0", "1", "2", "3", "4"]) == ["3", "4"]
    assert reloaded.totals("records") == 300


def test_checkpoint_flushes_before_reaching_the_cadence(tmp_path):
    ck = Checkpoint("x", tmp_path, flush_every=2)
    ck.mark("a")
    ck.mark("b")  # hits the cadence, writes to disk
    assert len(Checkpoint("x", tmp_path)) == 2


def test_corrupt_checkpoint_fails_loudly_instead_of_resetting(tmp_path):
    # Silently starting over would re-pay for every embedding already bought.
    (tmp_path / "bad.checkpoint.json").write_text("{not json")
    with pytest.raises(SystemExit, match="corrupt"):
        Checkpoint("bad", tmp_path)


def test_atomic_write_leaves_no_temp_files(tmp_path):
    target = tmp_path / "nested" / "out.json"
    atomic_write_bytes(target, b'{"ok":true}')
    assert json.loads(target.read_text()) == {"ok": True}
    assert list(tmp_path.rglob("*.tmp")) == []


def test_atomic_path_discards_partial_output_on_failure(tmp_path):
    target = tmp_path / "part.parquet"
    with pytest.raises(RuntimeError):
        with atomic_path(target) as tmp:
            tmp.write_bytes(b"half-written")
            raise RuntimeError("crash mid-write")
    # A resume must not mistake a truncated file for completed work.
    assert not target.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_progress_reports_position_total_and_eta(tmp_path, capsys):
    status = tmp_path / "status.json"
    prog = Progress("staging sec", total=1380, status_path=status, report_every=0)
    prog.advance(current="shard 1", ticker="NVDA")
    out = capsys.readouterr().err
    assert "1/1,380" in out
    assert "ETA" in out
    assert "ticker=NVDA" in out

    payload = json.loads(status.read_text())
    assert payload["done"] == 1 and payload["total"] == 1380
    assert payload["current"] == "shard 1"


def test_progress_resumes_the_denominator_not_just_the_counter(tmp_path):
    status = tmp_path / "status.json"
    prog = Progress("staging", total=1380, status_path=status, done=400,
                    report_every=0)
    prog.advance()
    payload = json.loads(status.read_text())
    assert payload["done"] == 401
    assert payload["resumed_from"] == 400
    assert 29 < payload["pct"] < 30
