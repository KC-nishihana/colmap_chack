"""レビュー画面と採否ロジックのテスト (pytest-qt・torch不要)。"""

import numpy as np

from ai.ai_mask_ops import APPLY_EXCLUDE
from ai.propagation_order import PropagationOrder
from ai.propagation_protocol import PropagationDirection
from ai.propagation_session import (
    FrameState,
    PropagationFrame,
    PropagationSession,
    PropagationUiState,
)
from ai.propagation_staging import write_mask_png_atomic
from ui.propagation_review_dialog import PropagationReviewDialog


def _session(tmp_path):
    frames = []
    for i in range(4):
        m = np.zeros((32, 48), np.uint8)
        m[5:15, 5 + i:15 + i] = 255
        rp = tmp_path / f"{i:06d}.png"
        write_mask_png_atomic(rp, m)
        state = FrameState.WARNING if i == 1 else FrameState.DONE
        warns = ["LOW_IOU"] if i == 1 else []
        frames.append(PropagationFrame(i, f"e{i}", f"p{i}", result_mask_path=str(rp),
                                       state=state, warning_codes=warns))
    frames[3].state = FrameState.FAILED
    frames[3].result_mask_path = None
    return PropagationSession(
        job_id="prop-r", state=PropagationUiState.REVIEW,
        reference_entry_key="e2", reference_frame_index=2, reference_mask_path="ref.png",
        order_mode=PropagationOrder.CURRENT_LIST, direction=PropagationDirection.BOTH,
        frames=frames, model_id="sam2.1_hiera_small", precision="bf16", device="cuda:0",
    )


def test_dialog_constructs_and_lists(qtbot, tmp_path):
    dlg = PropagationReviewDialog(_session(tmp_path))
    qtbot.addWidget(dlg)
    assert dlg._list.count() == 4
    # 既定で 基準(e2)/失敗(e3) を除く e0,e1 が採用対象
    assert {f.entry_key for f in dlg.accepted_frames()} == {"e0", "e1"}


def test_bulk_no_warn_excludes_warned(qtbot, tmp_path):
    dlg = PropagationReviewDialog(_session(tmp_path))
    qtbot.addWidget(dlg)
    dlg._bulk_accept("no_warn")
    keys = {f.entry_key for f in dlg.accepted_frames()}
    assert keys == {"e0"}   # e1 は警告ありで除外、e2 基準, e3 失敗


def test_bulk_none_then_all(qtbot, tmp_path):
    dlg = PropagationReviewDialog(_session(tmp_path))
    qtbot.addWidget(dlg)
    dlg._bulk_accept("none")
    assert dlg.accepted_frames() == []
    dlg._bulk_accept("all")
    assert {f.entry_key for f in dlg.accepted_frames()} == {"e0", "e1"}


def test_apply_mode_returned(qtbot, tmp_path):
    dlg = PropagationReviewDialog(_session(tmp_path))
    qtbot.addWidget(dlg)
    idx = dlg._mode.findData(APPLY_EXCLUDE)
    dlg._mode.setCurrentIndex(idx)
    assert dlg.apply_mode() == APPLY_EXCLUDE
