"""伝播の状態ガード判定 (controller predicates) のテスト (torch不要)。"""

from pathlib import Path

from ai.process_manager import SamProcessManager
from ai.propagation_order import PropagationOrder
from ai.propagation_protocol import PropagationDirection
from ai.propagation_session import PropagationSession, PropagationUiState
from ui.propagation_controller import PropagationController

FAKE_WORKER = Path(__file__).resolve().parent / "fake_sam_worker.py"


def _ctrl(qtbot):
    # Worker は起動しない (シグナル接続のためにマネージャだけ用意)
    mgr = SamProcessManager(worker_main_path=FAKE_WORKER)
    return PropagationController(mgr)


def _session(state):
    return PropagationSession(
        job_id="prop-1", state=state, reference_entry_key="e0",
        reference_frame_index=0, reference_mask_path="r.png",
        order_mode=PropagationOrder.CURRENT_LIST, direction=PropagationDirection.BOTH,
        frames=[], model_id="m", precision="bf16", device="cuda:0",
    )


def test_idle_not_active_no_unapplied(qtbot):
    c = _ctrl(qtbot)
    assert not c.is_active()
    assert not c.has_unapplied_results()


def test_running_is_active(qtbot):
    c = _ctrl(qtbot)
    c._session = _session(PropagationUiState.RUNNING)
    assert c.is_active()
    assert not c.has_unapplied_results()


def test_paused_is_active(qtbot):
    c = _ctrl(qtbot)
    c._session = _session(PropagationUiState.PAUSED)
    assert c.is_active()


def test_review_has_unapplied(qtbot):
    c = _ctrl(qtbot)
    c._session = _session(PropagationUiState.REVIEW)
    assert not c.is_active()
    assert c.has_unapplied_results()


def test_discard_clears(qtbot):
    c = _ctrl(qtbot)
    c._session = _session(PropagationUiState.REVIEW)
    c._active_job_id = "prop-1"
    c.discard_session()
    assert c.session is None
    assert not c.has_unapplied_results()
