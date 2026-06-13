"""
GrabCut補正UIの存在・有効/無効状態のGUIスモークテスト (v0.4B)

MainWindowのGrabCut補正グループとボタンの存在確認、
状態遷移による有効/無効切替を検証する。
pytestmark で xdist非対応 (GUI テストはシングルスレッド)。
"""

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGroupBox, QPushButton

from core.mask_ops import MaskEditor
from ui.image_canvas import GrabCutUiState, ImageCanvas
from ui.main_window import MainWindow


# ------------------------------------------------------------------ #
# テスト用ヘルパー
# ------------------------------------------------------------------ #

def make_image(h: int = 60, w: int = 80) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[5:h - 5, 5:w - 5] = (140, 140, 140)
    return img


def make_mask(h: int = 60, w: int = 80) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


# ------------------------------------------------------------------ #
# MainWindow の GrabCut補正グループ存在確認
# ------------------------------------------------------------------ #

def test_gc_correction_group_exists(qtbot):
    """GrabCut補正グループが存在する"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert hasattr(win, "_gc_correction_group")
    assert isinstance(win._gc_correction_group, QGroupBox)


def test_hint_buttons_exist(qtbot):
    """FG/BG/消去ヒントボタンが存在する"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert hasattr(win, "_btn_hint_fg")
    assert hasattr(win, "_btn_hint_bg")
    assert hasattr(win, "_btn_hint_erase")
    assert isinstance(win._btn_hint_fg, QPushButton)
    assert isinstance(win._btn_hint_bg, QPushButton)
    assert isinstance(win._btn_hint_erase, QPushButton)


def test_refine_apply_cancel_buttons_exist(qtbot):
    """再推定/適用/キャンセルボタンが存在する"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert hasattr(win, "_btn_refine")
    assert hasattr(win, "_btn_gc_apply")
    assert hasattr(win, "_btn_gc_cancel")
    assert isinstance(win._btn_refine, QPushButton)
    assert isinstance(win._btn_gc_apply, QPushButton)
    assert isinstance(win._btn_gc_cancel, QPushButton)


def test_hint_undo_redo_clear_buttons_exist(qtbot):
    """ヒントUndо/Redo/全消去ボタンが存在する"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert hasattr(win, "_btn_hint_undo")
    assert hasattr(win, "_btn_hint_redo")
    assert hasattr(win, "_btn_hint_clear")


def test_hint_radius_controls_exist(qtbot):
    """ヒントブラシサイズのSpinBoxとSliderが存在する"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert hasattr(win, "_hint_radius_spin")
    assert hasattr(win, "_hint_radius_slider")


# ------------------------------------------------------------------ #
# 初期状態では補正UIが無効
# ------------------------------------------------------------------ #

def test_gc_correction_group_disabled_initially(qtbot):
    """初期状態では補正グループが無効 (セッションなし)"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert not win._gc_correction_group.isEnabled()


def test_hint_buttons_disabled_initially(qtbot):
    """初期状態ではヒントボタンが無効"""
    win = MainWindow()
    qtbot.addWidget(win)
    # グループが無効なので個別ボタンも無効
    assert not win._gc_correction_group.isEnabled()


# ------------------------------------------------------------------ #
# GrabCut状態変化によるUI有効/無効切替
# ------------------------------------------------------------------ #

def test_correction_group_enabled_on_preview_state(qtbot):
    """PREVIEWstate + session有りで補正グループが有効になる"""
    win = MainWindow()
    qtbot.addWidget(win)

    # セッションダミーを設定
    from tests.test_grabcut_refine_worker import make_session as _make_session
    win._gc_session = _make_session()

    # PREVIEW状態をシミュレート
    win._on_grabcut_state_changed(GrabCutUiState.PREVIEW)

    assert win._gc_correction_group.isEnabled()


def test_correction_group_enabled_on_hint_editing(qtbot):
    """HINT_EDITING状態で補正グループが有効になる"""
    win = MainWindow()
    qtbot.addWidget(win)

    from tests.test_grabcut_refine_worker import make_session as _make_session
    win._gc_session = _make_session()

    win._on_grabcut_state_changed(GrabCutUiState.HINT_EDITING)

    assert win._gc_correction_group.isEnabled()


def test_correction_group_disabled_on_idle(qtbot):
    """IDLE状態では補正グループが無効になる"""
    win = MainWindow()
    qtbot.addWidget(win)

    from tests.test_grabcut_refine_worker import make_session as _make_session
    win._gc_session = _make_session()

    # まずPREVIEWで有効にする
    win._on_grabcut_state_changed(GrabCutUiState.PREVIEW)
    assert win._gc_correction_group.isEnabled()

    # IDLEで無効になる
    win._gc_session = None
    win._on_grabcut_state_changed(GrabCutUiState.IDLE)
    assert not win._gc_correction_group.isEnabled()


def test_correction_group_disabled_while_processing(qtbot):
    """INITIAL_RUNNING/REFINE_RUNNING中は補正グループが無効"""
    win = MainWindow()
    qtbot.addWidget(win)

    from tests.test_grabcut_refine_worker import make_session as _make_session
    win._gc_session = _make_session()

    for state in (GrabCutUiState.INITIAL_RUNNING, GrabCutUiState.REFINE_RUNNING):
        win._on_grabcut_state_changed(state)
        assert not win._gc_correction_group.isEnabled(), f"{state} で補正グループが有効になっている"


# ------------------------------------------------------------------ #
# ImageCanvas のヒントツール操作
# ------------------------------------------------------------------ #

def test_canvas_has_gc_ui_state(qtbot):
    """ImageCanvasがgc_ui_stateプロパティを持つ"""
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    assert hasattr(canvas, "gc_ui_state")
    assert canvas.gc_ui_state == GrabCutUiState.IDLE


def test_canvas_set_hint_label(qtbot):
    """set_hint_label でヒントラベルが設定される"""
    from core.grabcut_tool import GrabCutHintLabel
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)

    canvas.set_hint_label(GrabCutHintLabel.FOREGROUND)
    assert canvas._gc_hint_label == GrabCutHintLabel.FOREGROUND
    assert canvas._gc_hint_is_active


def test_canvas_set_hint_label_none_erase(qtbot):
    """set_hint_label(None) で消去モードになる"""
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)

    canvas.set_hint_label(None)
    assert canvas._gc_hint_label is None
    assert canvas._gc_hint_is_active


def test_canvas_hint_radius_set_get(qtbot):
    """set_hint_radius / get_hint_radius が正しく動作する"""
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)

    canvas.set_hint_radius(50)
    assert canvas.get_hint_radius() == 50


def test_canvas_gc_undo_redo_hint(qtbot):
    """gc_undo_hint / gc_redo_hint がクラッシュしない"""
    from core.grabcut_tool import GrabCutHintLabel, HintStroke
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)

    # 手動でストロークを追加
    stroke = HintStroke(label=GrabCutHintLabel.FOREGROUND, points=[(10, 10)], radius=5)
    canvas._gc_hint_strokes.append(stroke)

    # Undo
    canvas.gc_undo_hint()
    assert len(canvas._gc_hint_strokes) == 0
    assert len(canvas._gc_hint_redo_stack) == 1

    # Redo
    canvas.gc_redo_hint()
    assert len(canvas._gc_hint_strokes) == 1
    assert len(canvas._gc_hint_redo_stack) == 0


def test_canvas_gc_clear_hints(qtbot):
    """gc_clear_hints でストロークとRedoスタックがクリアされる"""
    from core.grabcut_tool import GrabCutHintLabel, HintStroke
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)

    for i in range(3):
        canvas._gc_hint_strokes.append(
            HintStroke(label=GrabCutHintLabel.FOREGROUND, points=[(i, i)], radius=3)
        )
    canvas._gc_hint_redo_stack.append(
        HintStroke(label=GrabCutHintLabel.BACKGROUND, points=[(5, 5)], radius=3)
    )

    canvas.gc_clear_hints()

    assert len(canvas._gc_hint_strokes) == 0
    assert len(canvas._gc_hint_redo_stack) == 0


# ------------------------------------------------------------------ #
# ヒントブラシサイズの同期
# ------------------------------------------------------------------ #

def test_hint_radius_spin_slider_sync(qtbot):
    """ヒントブラシSpinBoxを変更するとSliderが同期する"""
    win = MainWindow()
    qtbot.addWidget(win)

    win._hint_radius_spin.setValue(75)

    assert win._hint_radius_slider.value() == 75
    assert win._canvas.get_hint_radius() == 75


def test_hint_radius_slider_spin_sync(qtbot):
    """ヒントブラシSliderを変更するとSpinBoxが同期する"""
    win = MainWindow()
    qtbot.addWidget(win)

    win._hint_radius_slider.setValue(120)

    assert win._hint_radius_spin.value() == 120
    assert win._canvas.get_hint_radius() == 120


# ------------------------------------------------------------------ #
# window title がバージョン表記を含むこと
# ------------------------------------------------------------------ #

def test_window_title_contains_v04b(qtbot):
    """ウィンドウタイトルにv0.5が含まれる"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert "v0.5" in win.windowTitle()


# ------------------------------------------------------------------ #
# 既存マスク背景制約チェックボックス
# ------------------------------------------------------------------ #

def test_use_existing_mask_checkbox_exists(qtbot):
    """既存マスク背景制約チェックボックスが存在する"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert hasattr(win, "_grabcut_use_existing_mask_cb")


def test_use_existing_mask_checkbox_default_unchecked(qtbot):
    """既存マスク背景制約チェックボックスのデフォルトはOFF"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert not win._grabcut_use_existing_mask_cb.isChecked()
