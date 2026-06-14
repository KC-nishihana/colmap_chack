"""伝播の pause / resume / cancel テスト (Fake Worker・torch不要)。"""

import os
from pathlib import Path

import pytest

from ai.process_manager import SamProcessManager
from ai.propagation_protocol import PropagationCommand, PropagationEvent

FAKE_WORKER = Path(__file__).resolve().parent / "fake_sam_worker.py"


@pytest.fixture(autouse=True)
def _clean_mode():
    yield
    os.environ.pop("FAKE_SAM_MODE", None)


def _mgr(tmp_path, mode):
    os.environ["FAKE_SAM_MODE"] = mode
    os.environ["COLMAP_MASK_EDITOR_RUNTIME_DIR"] = str(tmp_path / "rt")
    return SamProcessManager(worker_main_path=FAKE_WORKER)


def _frames(n):
    return [{"frame_index": i, "entry_key": f"e{i}", "source_path": f"p{i}"} for i in range(n)]


def _start_all(qtbot, mgr, events, n):
    with qtbot.waitSignal(mgr.worker_started, timeout=15000):
        assert mgr.start()
    mgr.send_command(PropagationCommand.START, frames=_frames(n),
                     reference_frame_index=0, reference_mask_path="r", checkpoint_path="c",
                     direction="forward")
    qtbot.waitUntil(lambda: any(e.get("event") == PropagationEvent.STARTED for e in events),
                    timeout=15000)
    return next(e for e in events if e.get("event") == PropagationEvent.STARTED)["job_id"]


def _events(mgr):
    ev = []
    mgr.event_received.connect(ev.append)
    mgr.error_received.connect(ev.append)
    return ev


def test_pause_then_resume_completes(qtbot, tmp_path):
    mgr = _mgr(tmp_path, "propagation_slow")
    events = _events(mgr)
    job_id = _start_all(qtbot, mgr, events, n=6)

    mgr.send_command(PropagationCommand.PAUSE, job_id=job_id)
    qtbot.waitUntil(lambda: any(e.get("event") == PropagationEvent.PAUSED for e in events),
                    timeout=15000)
    mgr.send_command(PropagationCommand.RESUME, job_id=job_id)
    qtbot.waitUntil(lambda: any(e.get("event") == PropagationEvent.RESUMED for e in events),
                    timeout=15000)
    qtbot.waitUntil(lambda: any(e.get("event") == PropagationEvent.COMPLETED for e in events),
                    timeout=20000)
    mgr.stop()


def test_cancel_keeps_completed_results(qtbot, tmp_path):
    mgr = _mgr(tmp_path, "propagation_slow")
    events = _events(mgr)
    job_id = _start_all(qtbot, mgr, events, n=8)

    # 少なくとも1フレーム完了するまで待つ
    qtbot.waitUntil(lambda: any(e.get("event") == PropagationEvent.FRAME_READY for e in events),
                    timeout=15000)
    done_paths = [e["result_mask_path"] for e in events
                  if e.get("event") == PropagationEvent.FRAME_READY]

    mgr.send_command(PropagationCommand.CANCEL, job_id=job_id)
    qtbot.waitUntil(lambda: any(e.get("event") == PropagationEvent.CANCELLED for e in events),
                    timeout=15000)

    # 完成済み結果は保持される & Worker は維持
    assert all(Path(p).exists() for p in done_paths)
    assert mgr.is_running()
    mgr.stop()


def test_status_returns_progress(qtbot, tmp_path):
    mgr = _mgr(tmp_path, "propagation_slow")
    events = _events(mgr)
    job_id = _start_all(qtbot, mgr, events, n=6)
    mgr.send_command(PropagationCommand.STATUS, job_id=job_id)
    qtbot.waitUntil(lambda: any(e.get("event") == PropagationEvent.PROGRESS
                                and "total" in e for e in events), timeout=15000)
    snap = next(e for e in events if e.get("event") == PropagationEvent.PROGRESS and "total" in e)
    assert snap["total"] == 6
    mgr.stop()
