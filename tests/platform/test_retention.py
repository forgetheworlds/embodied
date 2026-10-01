"""Retention: what a scored run keeps on disk (the runner's retention rule).

A scored run's receipt-level files are the record; its stereo pairs, MAVLink and
feed streams are the raw material that record was made from. The rule under
test: when a run is scored, runs of the same stage that have fallen outside the
newest-``WINDOW_RUNS`` window lose their bulky artifacts and keep the receipt
level, unless a citation, a sensor capture or the operator's keep flag protects
them.
"""

import json
import os
import time
from pathlib import Path

import pytest

from embodied import retention


def _touch(path: Path, size: int = 64) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def _make_run(stage: Path, name: str, *, receipt: bool = True, attempt: str = "run-a") -> Path:
    """A scored run with one bulky attempt and the receipt-level files."""
    run_dir = stage / name
    _touch(run_dir / attempt / "pairs" / "00001-left.ppm", 614 * 1024)
    _touch(run_dir / attempt / "pairs" / "00001-right.ppm", 614 * 1024)
    _touch(run_dir / attempt / "mavlink.jsonl", 1024)
    _touch(run_dir / attempt / "estimator-feed.jsonl", 512)
    _touch(run_dir / attempt / "checks.json", 32)
    _touch(run_dir / attempt / "bring-up.json", 32)
    _touch(run_dir / "manifest.json", 16)
    _touch(run_dir / "preflight.json", 16)
    if receipt:
        artifacts = [
            {"path": f"{attempt}/checks.json"},
            {"path": f"{attempt}/bring-up.json"},
            {"path": f"{attempt}/mavlink.jsonl"},
            {"path": f"{attempt}/estimator-feed.jsonl"},
            {"path": f"{attempt}/pairs/00001-left.ppm"},
            {"path": "manifest.json"},
            {"path": "preflight.json"},
        ]
        (run_dir / "receipt.json").write_text(json.dumps({"artifacts": artifacts}))
    return run_dir


class TestPrunablePaths:
    def test_bulky_artifacts_are_listed_and_receipt_level_files_are_not(self, tmp_path: Path):
        run_dir = _make_run(tmp_path, "p01l-x-1")
        prunable = {path.relative_to(run_dir).as_posix() for path in retention.prunable_paths(run_dir)}
        assert "run-a/pairs" in prunable
        assert "run-a/mavlink.jsonl" in prunable
        assert "run-a/estimator-feed.jsonl" in prunable
        for kept in (
            "receipt.json",
            "manifest.json",
            "preflight.json",
            "run-a/checks.json",
            "run-a/bring-up.json",
        ):
            assert kept not in prunable

    def test_orphan_attempt_directories_are_prunable(self, tmp_path: Path):
        run_dir = _make_run(tmp_path, "accept-x")
        _touch(run_dir / "run-b" / "mavlink.jsonl", 1024)
        prunable = {path.relative_to(run_dir).as_posix() for path in retention.prunable_paths(run_dir)}
        assert "run-b" in prunable


class TestPruneRun:
    def test_receipt_level_files_survive_and_prune_json_records_the_removal(self, tmp_path: Path):
        run_dir = _make_run(tmp_path, "p01l-x-1")
        record = retention.prune_run(run_dir)
        assert not (run_dir / "run-a" / "pairs").exists()
        assert not (run_dir / "run-a" / "mavlink.jsonl").exists()
        assert not (run_dir / "run-a" / "estimator-feed.jsonl").exists()
        assert (run_dir / "run-a" / "checks.json").exists()
        assert (run_dir / "receipt.json").exists()
        assert (run_dir / "manifest.json").exists()
        written = json.loads((run_dir / retention.PRUNE_FILENAME).read_text())
        assert written["retention_version"] == retention.RETENTION_VERSION
        assert written["run"] == "p01l-x-1"
        removed = {entry["path"] for entry in written["removed"]}
        assert "run-a/pairs" in removed
        assert written["bytes_freed"] > 0

    def test_orphan_attempt_directory_is_removed_wholesale(self, tmp_path: Path):
        run_dir = _make_run(tmp_path, "accept-x")
        _touch(run_dir / "run-b" / "sitl.log", 1024)
        retention.prune_run(run_dir)
        assert not (run_dir / "run-b").exists()
        assert (run_dir / "run-a" / "checks.json").exists()

    def test_a_sensor_capture_run_is_kept_whole(self, tmp_path: Path):
        run_dir = _make_run(tmp_path, "p01l-capture-1")
        _touch(run_dir / "run-a" / "sensor-capture" / "records.jsonl", 1024)
        record = retention.prune_run(run_dir)
        assert record["pruned"] is False
        assert "capture" in record["reason"]
        assert (run_dir / "run-a" / "pairs").exists()
        assert not (run_dir / retention.PRUNE_FILENAME).exists()

    def test_the_keep_marker_protects_a_run(self, tmp_path: Path):
        run_dir = _make_run(tmp_path, "p01l-under-diagnosis")
        (run_dir / retention.KEEP_MARKER).write_text('{"reason": "operator --keep-artifacts"}\n')
        record = retention.prune_run(run_dir)
        assert record["pruned"] is False
        assert (run_dir / "run-a" / "pairs").exists()

    def test_captures_prunable_lets_the_apply_pass_take_a_capture(self, tmp_path: Path):
        run_dir = _make_run(tmp_path, "p01l-capture-old")
        _touch(run_dir / "run-a" / "sensor-capture" / "records.jsonl", 1024)
        record = retention.prune_run(run_dir, captures_prunable=True)
        assert record["pruned"] is True
        assert not (run_dir / "run-a" / "sensor-capture").exists()
        assert (run_dir / "run-a" / "checks.json").exists()

    def test_a_run_cited_by_a_document_is_kept_whole(self, tmp_path: Path):
        run_dir = _make_run(tmp_path, "p01l-cited-20260927T042005Z")
        other = _make_run(tmp_path, "p01l-plain")
        source = tmp_path / "LEARNED-FAILURES.md"
        source.write_text(
            "The frozen pose was found in run `p01l-cited-20260927T042005Z`, "
            "whose feed rows told the whole story.\n"
        )
        record = retention.prune_run(run_dir, citation_sources=(source,))
        assert record["pruned"] is False
        assert "cited" in record["reason"]
        assert (run_dir / "run-a" / "pairs").exists()
        # The uncited sibling prunes normally with the same sources.
        assert retention.prune_run(other, citation_sources=(source,))["pruned"] is True

    def test_a_cited_bulky_file_survives_inside_a_pruned_run(self, tmp_path: Path):
        stage = tmp_path / "p01-localization"
        run_dir = _make_run(stage, "p01l-zupt5-20260927T042005Z")
        source = tmp_path / "LEARNED-FAILURES.md"
        source.write_text(
            "row 25: `p01l-zupt5-20260927T042005Z/run-a/estimator-feed.jsonl` is the evidence.\n"
        )
        protected = retention.citation_paths((source,), stage, tmp_path)
        record = retention.prune_run(run_dir, protected_paths=protected)
        assert record["pruned"] is True
        # The file a claim depends on survives; the same run's uncited bulk goes.
        assert (run_dir / "run-a" / "estimator-feed.jsonl").exists()
        assert not (run_dir / "run-a" / "pairs").exists()
        assert not (run_dir / "run-a" / "mavlink.jsonl").exists()


class TestCitationScanning:
    def test_brace_series_and_numeric_ranges_expand_to_run_names(self, tmp_path: Path):
        stage = tmp_path / "p01-localization"
        for index in (1, 2, 3, 4, 5):
            _make_run(stage, f"p01l-clamp-{index}-20260930T1600{index}Z")
        source = tmp_path / "report.md"
        source.write_text("measured on `p01l-clamp-{1..5}-20260930T16` and `p01l-clamp-{1,3}`\n")
        mentioned = retention.cited_run_names((source,), stage)
        for index in (1, 2, 3, 4, 5):
            assert any(name.startswith(f"p01l-clamp-{index}-") for name in mentioned)

    def test_elided_run_ids_resolve_by_prefix_and_suffix(self, tmp_path: Path):
        stage = tmp_path / "p01-localization"
        _make_run(stage, "p01l-replay-ref-20261001T034502Z")
        source = tmp_path / "report.md"
        source.write_text("flight 1 (`p01l-replay-ref-\u202634502Z`, P6)\n")
        mentioned = retention.cited_run_names((source,), stage)
        assert any("034502Z" in name for name in mentioned)

    def test_document_scoped_bare_references_protect_named_runs(self, tmp_path: Path):
        stage = tmp_path / "p00-airframe"
        mission_08 = _make_run(stage, "accept-mission-08-2026-09-25T1622Z")
        mission_10 = _make_run(stage, "accept-mission-10-2026-09-25T1700Z")
        for run_dir in (mission_08, mission_10):
            _touch(run_dir / "run-a" / "imu.jsonl", 256)
            _touch(run_dir / "run-b" / "mavlink.jsonl", 256)
            _touch(run_dir / "run-b" / "imu.jsonl", 256)
        source = tmp_path / "final-fix-report.md"
        source.write_text(
            "accept-mission-08-2026-09-25T1622Z receipt, checks.json, "
            "run-{a,b}/mavlink.jsonl, motion.jsonl. Each run's `imu.jsonl` "
            "was re-read; accept-mission-10-2026-09-25T1700Z is the control.\n"
        )
        cited = retention.citation_paths((source,), stage, tmp_path)
        assert (stage / "accept-mission-08-2026-09-25T1622Z" / "run-a" / "mavlink.jsonl") in cited
        assert (stage / "accept-mission-08-2026-09-25T1622Z" / "run-b" / "mavlink.jsonl") in cited
        assert (stage / "accept-mission-08-2026-09-25T1622Z" / "run-a" / "imu.jsonl") in cited
        assert (stage / "accept-mission-10-2026-09-25T1700Z" / "run-a" / "imu.jsonl") in cited
        # The named runs themselves are cited too, so they stay whole.
        assert retention._run_is_cited(stage / "accept-mission-08-2026-09-25T1622Z", (source,))


class TestEnforceWindow:
    def test_newest_window_runs_stay_whole_and_older_runs_prune(self, tmp_path: Path):
        stage = tmp_path / "p01-localization"
        runs = []
        for index in range(retention.WINDOW_RUNS + 3):
            run_dir = _make_run(stage, f"p01l-old-{index:02d}")
            stamp = time.time() - (1000 - index)
            os.utime(run_dir / "receipt.json", (stamp, stamp))
            runs.append(run_dir)
        records = retention.enforce_window(stage, citation_sources=())
        pruned = {record["run"] for record in records if record["pruned"]}
        assert pruned == {runs[index].name for index in range(3)}
        for index in range(3):
            assert not (runs[index] / "run-a" / "pairs").exists()
        for run_dir in runs[3:]:
            assert (run_dir / "run-a" / "pairs").exists()

    def test_protected_and_cited_runs_do_not_consume_window_slots(self, tmp_path: Path):
        stage = tmp_path / "p01-localization"
        cited = _make_run(stage, "p01l-cited-1")
        capture = _make_run(stage, "p01l-capture-1")
        _touch(capture / "run-a" / "sensor-capture" / "records.jsonl", 16)
        plain = []
        for index in range(retention.WINDOW_RUNS + 1):
            plain.append(_make_run(stage, f"p01l-plain-{index:02d}"))
        stamp = time.time() - 2000
        for run_dir in (cited, capture, *plain):
            os.utime(run_dir / "receipt.json", (stamp, stamp))
            stamp += 1
        source = tmp_path / "docs.md"
        source.write_text("`p01l-cited-1` backs a live claim.\n")
        retention.enforce_window(stage, citation_sources=(source,))
        assert (cited / "run-a" / "pairs").exists()
        assert (capture / "run-a" / "pairs").exists()
        # The protected runs did not eat window slots: exactly one plain run pruned.
        assert sum(not (run_dir / "run-a" / "pairs").exists() for run_dir in plain) == 1

    def test_runs_without_a_receipt_order_by_directory_time(self, tmp_path: Path):
        stage = tmp_path / "p01-localization"
        # Ten scored runs fill the window; a receipt-less failed launch that
        # predates them all is the run that leaves it.
        fresh_runs = []
        for index in range(retention.WINDOW_RUNS):
            fresh_runs.append(_make_run(stage, f"p01l-fresh-{index:02d}"))
        failed_old = _make_run(stage, "p01l-failed-old", receipt=False)
        os.utime(failed_old, (time.time() - 5000, time.time() - 5000))
        for run_dir in fresh_runs:
            os.utime(run_dir / "receipt.json", (time.time(), time.time()))
        retention.enforce_window(stage, citation_sources=())
        assert not (failed_old / "run-a" / "pairs").exists()
        assert all((run_dir / "run-a" / "pairs").exists() for run_dir in fresh_runs)


class TestApplyOnScore:
    def _repo(self, tmp_path: Path) -> Path:
        (tmp_path / "README.md").write_text("# repo\n")
        return tmp_path

    def test_scoring_a_run_prunes_the_run_that_left_the_window(self, tmp_path: Path):
        root = self._repo(tmp_path)
        stage = root / "work" / "runs" / "p01-localization"
        stage.mkdir(parents=True)
        for index in range(retention.WINDOW_RUNS):
            run_dir = _make_run(stage, f"p01l-old-{index:02d}")
            stamp = time.time() - (1000 - index)
            os.utime(run_dir / "receipt.json", (stamp, stamp))
        fresh = _make_run(stage, "p01l-new")
        stamp = time.time()
        os.utime(fresh / "receipt.json", (stamp, stamp))
        retention.apply_on_score(fresh, root, keep_all=False)
        assert (fresh / "run-a" / "pairs").exists()
        assert not (stage / "p01l-old-00" / "run-a" / "pairs").exists()
        assert (stage / "p01l-old-01" / "run-a" / "pairs").exists()

    def test_keep_all_writes_the_marker_and_prunes_nothing(self, tmp_path: Path):
        root = self._repo(tmp_path)
        stage = root / "work" / "runs" / "p01-localization"
        stage.mkdir(parents=True)
        runs = []
        for index in range(retention.WINDOW_RUNS + 2):
            run_dir = _make_run(stage, f"p01l-old-{index:02d}")
            stamp = time.time() - (1000 - index)
            os.utime(run_dir / "receipt.json", (stamp, stamp))
            runs.append(run_dir)
        retention.apply_on_score(runs[-1], root, keep_all=True)
        marker = json.loads((runs[-1] / retention.KEEP_MARKER).read_text())
        assert marker["reason"] == "operator --keep-artifacts"
        for run_dir in runs:
            assert (run_dir / "run-a" / "pairs").exists()

    def test_an_output_outside_the_runs_tree_enforces_no_window(self, tmp_path: Path):
        root = self._repo(tmp_path)
        stage = root / "work" / "runs" / "p01-localization"
        stage.mkdir(parents=True)
        run_dir = _make_run(stage, "p01l-kept")
        outside = _make_run(tmp_path / "scratch", "scratch-run")
        stamp = time.time() - 5000
        os.utime(run_dir / "receipt.json", (stamp, stamp))
        retention.apply_on_score(outside, root, keep_all=False)
        assert (run_dir / "run-a" / "pairs").exists()
        assert (outside / "run-a" / "pairs").exists()


class TestRunnerWiring:
    """The rule is the runner's: every command carries the flag, and scoring
    through the CLI enforces the window on the stage the run scored into."""

    def test_the_flag_is_accepted_after_a_nested_subcommand(self, tmp_path: Path):
        import embodied.bench.cli  # registers the bench command
        from embodied import cli

        episode = (
            Path(__file__).resolve().parents[1] / "fixtures" / "bench" / "hand-checkable-episode"
        )
        output = tmp_path / "flag-run"
        code = cli.main(
            ["bench", "replay", "--episode", str(episode), "--output", str(output),
             "--keep-artifacts"]
        )
        assert code == 0
        marker = json.loads((output / retention.KEEP_MARKER).read_text())
        assert marker["reason"] == "operator --keep-artifacts"
        plain = tmp_path / "plain-run"
        code = cli.main(["bench", "replay", "--episode", str(episode), "--output", str(plain)])
        assert code == 0
        assert not (plain / retention.KEEP_MARKER).exists()

    def test_scoring_through_the_cli_prunes_the_run_leaving_the_window(
        self, tmp_path: Path, monkeypatch
    ):
        import embodied.bench.cli  # registers the bench command
        from embodied import cli

        episode = Path(__file__).resolve().parents[1] / "fixtures" / "bench" / "hand-checkable-episode"
        root = tmp_path
        (root / "README.md").write_text("# repo\n")
        stage = root / "work" / "runs" / "p02-bench"
        stage.mkdir(parents=True)
        for index in range(retention.WINDOW_RUNS):
            run_dir = _make_run(stage, f"p02-old-{index:02d}")
            stamp = time.time() - (1000 - index)
            os.utime(run_dir / "receipt.json", (stamp, stamp))
        monkeypatch.setattr(cli, "repository_root", lambda: root)
        output = stage / "p02-new"
        code = cli.main(
            ["bench", "replay", "--episode", str(episode), "--output", str(output)]
        )
        assert code == 0
        assert (output / "receipt.json").is_file()
        # The stale run that left the window dropped to receipt level; the
        # scored run kept everything it wrote.
        assert not (stage / "p02-old-00" / "run-a" / "pairs").exists()
        assert (stage / "p02-old-01" / "run-a" / "pairs").exists()
        assert (stage / "p02-old-09" / "run-a" / "pairs").exists()
