"""伝播コントローラ (GUI側) のテスト: Fake Worker 経由 + 古いjob_id破棄 (torch不要)。"""

import os
from pathlib import Path

import numpy as np
import pytest

from ai.process_manager import SamProcessManager
from ai.propagation_order import PropagationOrder
from ai.propagation_protocol import PropagationDirection, PropagationEvent
from ai.propagation_session import FrameState, PropagationFrame, PropagationUiState
from ui.propagation_controller import PropagationController

FAKE_WORKER = Path(__file__).resolve().parent / "fake_sam_worker.py"


@pytest.fixture(autouse=True)
def _clean():
    yield
    os.environ.pop("FAKE_SAM_MODE", None)


def _frames(n=3):
    return [PropagationFrame(i, f"e{i}", f"p{i}") for i in range(n)]


def test_controller_full_run_builds_review_session(qtbot, tmp_path):
    os.environ["FAKE_SAM_MODE"] = "propagation_normal"
    os.environ["COLMAP_MASK_EDITOR_RUNTIME_DIR"] = str(tmp_path / "rt")
    mgr = SamProcessManager(worker_main_path=FAKE_WORKER)
    with qtbot.waitSignal(mgr.worker_started, timeout=15000):
        assert mgr.start()

    ctrl = PropagationController(mgr)
    ref_mask = np.zeros((48, 64), np.uint8)
    ref_mask[5:20, 5:20] = 255

    with qtbot.waitSignal(ctrl.completed, timeout=15000):
        ctrl.start(frames=_frames(3), reference_frame_index=0, reference_mask=ref_mask,
                   order_mode=PropagationOrder.CURRENT_LIST, direction=PropagationDirection.FORWARD,
                   model_id="sam2.1_hiera_small", checkpoint_path="ckpt.pt")

    assert ctrl.state == PropagationUiState.REVIEW
    s = ctrl.session
    assert s is not None and s.job_id
    done = [f for f in s.frames if f.state in (FrameState.DONE, FrameState.WARNING)]
    assert len(done) == 3
    assert all(f.result_mask_path for f in done)
    mgr.stop()


def test_controller_discards_foreign_job_id(qtbot, tmp_path):
    os.environ["FAKE_SAM_MODE"] = "propagation_normal"
    os.environ["COLMAP_MASK_EDITOR_RUNTIME_DIR"] = str(tmp_path / "rt")
    mgr = SamProcessManager(worker_main_path=FAKE_WORKER)
    with qtbot.waitSignal(mgr.worker_started, timeout=15000):
        assert mgr.start()
    ctrl = PropagationController(mgr)
    ref_mask = np.zeros((48, 64), np.uint8)
    ref_mask[5:20, 5:20] = 255
    with qtbot.waitSignal(ctrl.completed, timeout=15000):
        ctrl.start(frames=_frames(3), reference_frame_index=0, reference_mask=ref_mask,
                   order_mode=PropagationOrder.CURRENT_LIST, direction=PropagationDirection.FORWARD,
                   model_id="sam2.1_hiera_small", checkpoint_path="ckpt.pt")

    # 古い job_id の frame_ready は反映されない (一時結果は削除される)
    stale = tmp_path / "stale.png"
    stale.write_bytes(b"x")
    before = [f.result_mask_path for f in ctrl.session.frames]
    ctrl._on_event({"event": PropagationEvent.FRAME_READY, "job_id": "prop-OLD",
                    "frame_index": 0, "result_mask_path": str(stale)})
    after = [f.result_mask_path for f in ctrl.session.frames]
    assert before == after            # セッション不変
    assert not stale.exists()         # 古い結果は削除
    mgr.stop()
