"""画像伝播パネルのスモークテスト (pytest-qt・torch不要)。"""

from ai.propagation_order import PropagationOrder
from ai.propagation_protocol import PropagationDirection
from ai.propagation_session import PropagationUiState
from ui.propagation_panel import PropagationPanel


def test_panel_constructs_and_options(qtbot):
    p = PropagationPanel()
    qtbot.addWidget(p)
    opt = p.options()
    assert opt["direction"] in PropagationDirection.ALL
    assert isinstance(opt["order_mode"], PropagationOrder)
    assert opt["model_id"] == "sam2.1_hiera_small"
    assert opt["use_ai_candidate"] is True
    assert opt["count"] == 10
    assert opt["offload_video_to_cpu"] is True


def test_panel_state_enables(qtbot):
    p = PropagationPanel()
    qtbot.addWidget(p)

    p.set_state(PropagationUiState.IDLE)
    assert p._btn_start.isEnabled()
    assert not p._btn_pause.isEnabled()

    p.set_state(PropagationUiState.RUNNING)
    assert not p._btn_start.isEnabled()
    assert p._btn_pause.isEnabled()
    assert p._btn_cancel.isEnabled()

    p.set_state(PropagationUiState.PAUSED)
    assert p._btn_resume.isEnabled()

    p.set_state(PropagationUiState.REVIEW)
    assert p._btn_review.isEnabled()
    assert p._btn_start.isEnabled() is False or p._btn_start.isEnabled() is True  # review!=idle
    assert not p._btn_start.isEnabled()


def test_panel_order_preview_and_progress(qtbot):
    p = PropagationPanel()
    qtbot.addWidget(p)
    p.set_order_preview(["0 a.jpg", "1 b.jpg ← 基準"])
    assert p._order_list.count() == 2
    p.set_progress(3, 10)
    p.set_counts(2, 1, 0)
    p.set_vram(1234)
