"""伝播の QProcess 統合テスト (Fake Worker 使用・torch不要)。"""

import os
from pathlib import Path

import pytest

from ai.process_manager import SamProcessManager
from ai.propagation_protocol import PropagationCommand, PropagationEvent
from ai import protocol

FAKE_WORKER = Path(__file__).resolve().parent / "fake_sam_worker.py"


@pytest.fixture(autouse=True)
def _clean_mode():
    yield
    os.environ.pop("FAKE_SAM_MODE", None)


def _mgr(tmp_path, mode="propagation_normal"):
    os.environ["FAKE_SAM_MODE"] = mode
    os.environ["COLMAP_MASK_EDITOR_RUNTIME_DIR"] = str(tmp_path / "rt")
    return SamProcessManager(worker_main_path=FAKE_WORKER)


def _collector(mgr):
    events = []
    mgr.event_received.connect(events.append)
    mgr.error_received.connect(events.append)
    mgr.ready.connect(events.append)
    return events


def _frames(n=3):
    return [{"frame_index": i, "entry_key": f"e{i}", "source_path": f"p{i}"} for i in range(n)]


def _start(qtbot, mgr):
    with qtbot.waitSignal(mgr.worker_started, timeout=15000):
        assert mgr.start()


def _start_prop(qtbot, mgr, events, n=3):
    rid = mgr.send_command(PropagationCommand.START, frames=_frames(n),
                           reference_frame_index=0, reference_mask_path="ref.png",
                           model_id="sam2.1_hiera_small", checkpoint_path="ckpt.pt",
                           direction="forward")
    qtbot.waitUntil(lambda: any(e.get("event") == PropagationEvent.STARTED for e in events),
                    timeout=15000)
    return next(e for e in events if e.get("event") == PropagationEvent.STARTED)


def test_propagation_normal_streams_and_completes(qtbot, tmp_path):
    mgr = _mgr(tmp_path, "propagation_normal")
    events = _collector(mgr)
    _start(qtbot, mgr)
    started = _start_prop(qtbot, mgr, events, n=3)
    job_id = started["job_id"]
    assert started["frame_count"] == 3

    qtbot.waitUntil(lambda: any(e.get("event") == PropagationEvent.COMPLETED
                                and e.get("job_id") == job_id for e in events), timeout=15000)
    frame_evts = [e for e in events if e.get("event") == PropagationEvent.FRAME_READY]
    assert {e["frame_index"] for e in frame_evts} == {0, 1, 2}
    for e in frame_evts:
        assert Path(e["result_mask_path"]).exists()
    mgr.stop()


def test_propagation_busy_rejects_second_start(qtbot, tmp_path):
    mgr = _mgr(tmp_path, "propagation_slow")
    events = _collector(mgr)
    _start(qtbot, mgr)
    _start_prop(qtbot, mgr, events, n=5)
    # 実行中に2回目の start -> BUSY
    mgr.send_command(PropagationCommand.START, frames=_frames(3),
                     reference_frame_index=0, reference_mask_path="r", checkpoint_path="c")
    qtbot.waitUntil(lambda: any(e.get("error_code") == "PROPAGATION_BUSY" for e in events),
                    timeout=15000)
    mgr.stop()


def test_propagation_frame_failure_keeps_worker(qtbot, tmp_path):
    mgr = _mgr(tmp_path, "propagation_frame_failure")
    events = _collector(mgr)
    _start(qtbot, mgr)
    _start_prop(qtbot, mgr, events, n=4)
    qtbot.waitUntil(lambda: any(e.get("error_code") == "PROPAGATION_PREDICT_FAILED"
                                for e in events), timeout=15000)
    # Worker は維持される
    assert mgr.is_running()
    mgr.stop()


def test_propagation_crash_emits_worker_crashed(qtbot, tmp_path):
    mgr = _mgr(tmp_path, "propagation_crash")
    events = _collector(mgr)
    _start(qtbot, mgr)
    with qtbot.waitSignal(mgr.worker_crashed, timeout=15000):
        mgr.send_command(PropagationCommand.START, frames=_frames(3),
                         reference_frame_index=0, reference_mask_path="r", checkpoint_path="c")
    assert not mgr.is_running()


def test_propagation_warning_codes_forwarded(qtbot, tmp_path):
    mgr = _mgr(tmp_path, "propagation_warning")
    events = _collector(mgr)
    _start(qtbot, mgr)
    started = _start_prop(qtbot, mgr, events, n=3)
    job_id = started["job_id"]
    qtbot.waitUntil(lambda: any(e.get("event") == PropagationEvent.COMPLETED
                                and e.get("job_id") == job_id for e in events), timeout=15000)
    warned = [e for e in events if e.get("event") == PropagationEvent.FRAME_READY
              and e.get("warning_codes")]
    assert any("LOW_IOU" in e["warning_codes"] for e in warned)
    mgr.stop()
