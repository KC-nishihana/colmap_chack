"""
AIセグメントタブの GUI テスト (Worker を起動せず、セッション状態を直接組み立てて検証)。

確認:
  - AIタブのウィジェットが存在する
  - MainWindow は AI未導入 (Worker未起動) でも起動する
  - 適用前は通常マスクが変わらない
  - 追加/除外/置換 適用でマスクが変化し Undo 可能
  - キャンセルで通常マスクが変わらない
  - 候補切替で選択マスクが変わる (再推論しない)
"""

import numpy as np
import pytest

from core.mask_ops import MaskEditor
from ui.image_canvas import GrabCutUiState  # noqa: F401 (import 健全性)
from ui.main_window import MainWindow
from ai.ai_session import AiUiState
from ai.ai_mask_ops import AiCandidate, AiPredictionResult


def _make_result(h=60, w=80):
    m0 = np.zeros((h, w), dtype=np.uint8); m0[10:40, 10:50] = 255
    m1 = np.zeros((h, w), dtype=np.uint8); m1[5:55, 5:70] = 255
    m2 = np.zeros((h, w), dtype=np.uint8); m2[20:30, 20:40] = 255
    cands = [
        AiCandidate(0, m0, 0.95, int((m0 == 255).sum()), float((m0 == 255).mean())),
        AiCandidate(1, m1, 0.88, int((m1 == 255).sum()), float((m1 == 255).mean())),
        AiCandidate(2, m2, 0.70, int((m2 == 255).sum()), float((m2 == 255).mean())),
    ]
    return AiPredictionResult(request_id=1, image_key="k", width=w, height=h, candidates=cands)


def _setup_preview(win, h=60, w=80):
    """Worker 抜きで PREVIEW 状態を組み立てる。"""
    editor = MaskEditor(np.zeros((h, w), dtype=np.uint8))
    win._editor = editor
    win._canvas.set_editor(editor)
    sess = win._ai_session
    sess._image_key = "k"
    sess._model_id = "sam2.1_hiera_small"
    sess._result = _make_result(h, w)
    sess._selected_candidate = sess._result.best_index()
    sess._set_state(AiUiState.PREVIEW)
    win._refresh_ai_overlay()


# ----- ウィジェット存在 -----

def test_ai_tab_widgets_exist(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    for attr in ["_btn_ai_start_worker", "_btn_ai_restart_worker", "_btn_ai_load_model",
                 "_btn_ai_unload_model", "_btn_ai_predict", "_btn_ai_apply_add",
                 "_btn_ai_apply_exclude", "_btn_ai_apply_replace", "_btn_ai_cancel",
                 "_ai_model_combo", "_ai_precision_combo", "_ai_candidate_btns"]:
        assert hasattr(win, attr), attr
    assert len(win._ai_candidate_btns) == 3


def test_mainwindow_starts_without_worker(qtbot):
    """AI Worker を起動しなくても MainWindow は問題なく構築できる。"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert win._ai_session.state == AiUiState.DISABLED
    assert not win._ai_session.is_worker_running()


def test_model_combo_has_two_models(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    ids = [win._ai_model_combo.itemData(i) for i in range(win._ai_model_combo.count())]
    assert "sam2.1_hiera_small" in ids
    assert "sam2.1_hiera_base_plus" in ids
    # 初期選択が Small
    assert win._ai_model_combo.itemData(0) == "sam2.1_hiera_small"


# ----- 適用 -----

def test_apply_add_changes_mask_and_undoable(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    _setup_preview(win)
    before = win._editor.mask.copy()

    from ai.ai_mask_ops import APPLY_ADD
    win._apply_ai(APPLY_ADD)

    assert not np.array_equal(win._editor.mask, before)
    assert win._editor.can_undo()
    # 適用後はプレビューが破棄され PROMPT_EDITING へ
    assert win._ai_session.state in (AiUiState.PROMPT_EDITING, AiUiState.MODEL_READY)


def test_apply_exclude(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    _setup_preview(win)
    win._editor.mask[:] = 255
    before = win._editor.mask.copy()
    from ai.ai_mask_ops import APPLY_EXCLUDE
    win._apply_ai(APPLY_EXCLUDE)
    assert not np.array_equal(win._editor.mask, before)
    # 候補0領域が0になっている
    assert win._editor.mask[20, 20] == 0


def test_apply_replace(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    _setup_preview(win)
    win._editor.mask[:] = 255
    from ai.ai_mask_ops import APPLY_REPLACE
    win._apply_ai(APPLY_REPLACE)
    assert set(np.unique(win._editor.mask)).issubset({0, 255})
    # 候補0外側は0
    assert win._editor.mask[0, 0] == 0


def test_cancel_does_not_change_mask(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    _setup_preview(win)
    before = win._editor.mask.copy()
    win._on_ai_cancel()
    assert np.array_equal(win._editor.mask, before)
    assert win._ai_session.result is None


def test_preview_does_not_change_mask_before_apply(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    before_zero = np.zeros((60, 80), dtype=np.uint8)
    _setup_preview(win)
    # プレビュー状態でマスクはまだ全0のまま
    assert np.array_equal(win._editor.mask, before_zero)


def test_candidate_switch_changes_selected_mask(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    _setup_preview(win)
    m_best = win._ai_session.selected_mask().copy()
    # 候補2へ切替
    win._select_ai_candidate(2)
    m_two = win._ai_session.selected_mask()
    assert not np.array_equal(m_best, m_two)
    assert win._ai_session.selected_candidate_index == 2
