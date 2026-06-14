"""伝播セッションモデルのテスト (torch不要)。"""

from ai.propagation_order import PropagationOrder
from ai.propagation_protocol import PropagationDirection
from ai.propagation_session import (
    FrameState,
    PropagationFrame,
    PropagationSession,
    PropagationUiState,
)


def _session():
    frames = [
        PropagationFrame(0, "e0", "p/0", result_mask_path="r/0.png", state=FrameState.DONE),
        PropagationFrame(1, "e1", "p/1", result_mask_path="r/1.png", state=FrameState.WARNING,
                         warning_codes=["LOW_IOU"]),
        PropagationFrame(2, "e2", "p/2", result_mask_path="r/2.png", state=FrameState.DONE),  # 基準
        PropagationFrame(3, "e3", "p/3", state=FrameState.FAILED, error_message="x"),
    ]
    return PropagationSession(
        job_id="prop-1", state=PropagationUiState.REVIEW,
        reference_entry_key="e2", reference_frame_index=2, reference_mask_path="ref.png",
        order_mode=PropagationOrder.CURRENT_LIST, direction=PropagationDirection.BOTH,
        frames=frames, model_id="sam2.1_hiera_small", precision="bf16", device="cuda:0",
    )


def test_frame_by_index():
    s = _session()
    assert s.frame_by_index(1).entry_key == "e1"
    assert s.frame_by_index(99) is None


def test_is_reviewable():
    s = _session()
    assert s.frame_by_index(0).is_reviewable
    assert s.frame_by_index(1).is_reviewable      # WARNING も採否可
    assert not s.frame_by_index(3).is_reviewable  # FAILED は不可


def test_accepted_frames_excludes_reference_and_failed():
    s = _session()
    keys = [f.entry_key for f in s.accepted_frames()]
    assert keys == ["e0", "e1"]   # e2=基準除外, e3=failed除外


def test_accepted_frames_respects_accept_flag():
    s = _session()
    s.frame_by_index(1).accepted = False
    assert [f.entry_key for f in s.accepted_frames()] == ["e0"]


def test_recompute_counts():
    s = _session()
    s.recompute_counts()
    assert s.completed_count == 3   # done/warning
    assert s.warning_count == 1
    assert s.failed_count == 1
