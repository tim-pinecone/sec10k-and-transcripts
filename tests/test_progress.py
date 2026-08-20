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


def test_stage_shard_embeds_the_whole_shard_in_one_call(tmp_path, monkeypatch):
    """Per-namespace embedding silently caps concurrency at ~2 batches.

    A shard spans ~9 fiscal years, so embedding per namespace sees only a few hundred
    records at a time and never fills the request pool no matter what --concurrency is
    set to. This asserts one shard-wide call, and that vectors stay aligned with the
    records they belong to after the split back out by namespace.
    """
    import json

    import pyarrow.parquet as pq

    from finvec import stage as stage_mod
    from finvec.sources.base import Record

    def fake_shard(shard):
        for year in (2023, 2024):
            for chunk in range(3):
                text = f"{year}-{chunk} " * 20
                yield Record(
                    id=f"AAA_{year}_10K_CHUNK_{chunk}",
                    namespace=str(year),
                    text=text,
                    metadata={
                        "ticker": "AAA", "cik": "1", "fiscal_year": year,
                        "sic": "1", "sic_description": "x", "chunk_id": chunk,
                        "is_table": False, "is_boilerplate": False,
                        "token_count": len(text), "text": text,
                    },
                    token_count=len(text),
                )

    monkeypatch.setattr(stage_mod.sec10k, "iter_shard", fake_shard)

    class CountingEmbedder:
        def __init__(self):
            self.calls = 0
            self.seen = []

        def embed(self, items):
            self.calls += 1
            self.seen.append(len(items))
            # Encode the input index into the vector so alignment is checkable.
            return [[float(i)] * stage_mod.EMBED_DIMS for i in range(len(items))]

    embedder = CountingEmbedder()
    counts = stage_shard_counts = stage_mod.stage_shard(
        7, tmp_path, embedder, dataset="sec"
    )

    assert embedder.calls == 1, f"expected one embed call, got {embedder.calls}"
    assert sum(stage_shard_counts.values()) == embedder.seen[0]

    # Alignment: the vector written for each record must be the one generated for its
    # position in the single shard-wide call.
    position = 0
    for namespace in sorted(counts):
        part = next((tmp_path / "sec" / f"import-{namespace}" / namespace).glob("*.parquet"))
        table = pq.read_table(part)
        for value in table["values"].to_pylist():
            assert value[0] == float(position), (
                f"vector/record misalignment at position {position}"
            )
            position += 1
        for meta in table["metadata"].to_pylist():
            assert json.loads(meta)["fiscal_year"] == int(namespace)
