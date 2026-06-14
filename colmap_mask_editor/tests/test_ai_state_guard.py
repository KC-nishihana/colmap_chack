"""
AI未確定状態ガード (Phase 15) のテスト。

GrabCut の未確定ガードと同様に、_resolve_pending_ai_session が
apply / discard / cancel の3択で正しく振る舞うことを確認する。
"""

import numpy as np
import pytest

from core.mask_ops import MaskEditor
from ui.main_window import MainWindow
from ai.ai_session import AiUiState
from ai.ai_mask_ops import AiCandidate, AiPredictionResult


def _make_result(h=60, w=80):
    m = np.zeros((h, w), dtype=np.uint8); m[10:40, 10:50] = 255
    cands = [AiCandidate(0, m, 0.9, int((m == 255).sum()), float((m == 255).mean()))]
    return AiPredictionResult(request_id=1, image_key="k", width=w, height=h, candidates=cands)


def _setup_preview(win):
    editor = MaskEditor(np.zeros((60, 80), dtype=np.uint8))
    win._editor = editor
    win._canvas.set_editor(editor)
    sess = win._ai_session
    sess._image_key = "k"
    sess._model_id = "sam2.1_hiera_small"
    sess._result = _make_result()
    sess._selected_candidate = 0
    sess._set_state(AiUiState.PREVIEW)


def _setup_prompt_only(win):
    editor = MaskEditor(np.zeros((60, 80), dtype=np.uint8))
    win._editor = editor
    win._canvas.set_editor(editor)
    sess = win._ai_session
    sess._image_key = "k"
    sess._model_id = "sam2.1_hiera_small"
    sess._set_state(AiUiState.PROMPT_EDITING)
    sess.prompts.add_point(20, 20, positive=True)


def test_no_dialog_when_idle(qtbot, monkeypatch):
    win = MainWindow()
    qtbot.addWidget(win)
    called = [False]
    monkeypatch.setattr(win, "_ask_pending_ai",
                        lambda: (called.__setitem__(0, True) or "cancel"))
    assert win._resolve_pending_ai_session("テスト") is True
    assert called[0] is False


def test_preview_apply_updates_mask(qtbot, monkeypatch):
    win = MainWindow()
    qtbot.addWidget(win)
    _setup_preview(win)
    before = win._editor.mask.copy()
    monkeypatch.setattr(win, "_ask_pending_ai", lambda: "apply")
    assert win._resolve_pending_ai_session("テスト") is True
    assert not np.array_equal(win._editor.mask, before)
    assert win._ai_session.result is None


def test_preview_discard_keeps_mask(qtbot, monkeypatch):
    win = MainWindow()
    qtbot.addWidget(win)
    _setup_preview(win)
    before = win._editor.mask.copy()
    monkeypatch.setattr(win, "_ask_pending_ai", lambda: "discard")
    assert win._resolve_pending_ai_session("テスト") is True
    assert np.array_equal(win._editor.mask, before)
    assert win._ai_session.result is None


def test_preview_cancel_aborts(qtbot, monkeypatch):
    win = MainWindow()
    qtbot.addWidget(win)
    _setup_preview(win)
    monkeypatch.setattr(win, "_ask_pending_ai", lambda: "cancel")
    assert win._resolve_pending_ai_session("テスト") is False
    # プレビューが維持される
    assert win._ai_session.state == AiUiState.PREVIEW
    assert win._ai_session.result is not None


def test_prompt_only_discard_keeps_mask(qtbot, monkeypatch):
    win = MainWindow()
    qtbot.addWidget(win)
    _setup_prompt_only(win)
    before = win._editor.mask.copy()
    monkeypatch.setattr(win, "_ask_pending_ai", lambda: "discard")
    assert win._resolve_pending_ai_session("テスト") is True
    assert np.array_equal(win._editor.mask, before)
    assert win._ai_session.prompts.is_empty()


def test_prompt_only_cancel_aborts(qtbot, monkeypatch):
    win = MainWindow()
    qtbot.addWidget(win)
    _setup_prompt_only(win)
    monkeypatch.setattr(win, "_ask_pending_ai", lambda: "cancel")
    assert win._resolve_pending_ai_session("テスト") is False
    assert win._ai_session.prompts.has_any()


def test_ai_running_passes_when_not_predicting(qtbot):
    """PREDICTING でなければ _resolve_ai_running はダイアログなしで True。"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert win._ai_session.state != AiUiState.PREDICTING
    assert win._resolve_ai_running("テスト") is True
