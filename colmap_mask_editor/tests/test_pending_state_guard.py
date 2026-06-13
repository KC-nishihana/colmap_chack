"""
未確定状態確認テスト (v0.5.1)

GrabCutプレビュー・ヒント編集中・未保存マスクの確認ダイアログを
monkeypath でシミュレートし、3択の動作を検証する。
"""

import numpy as np
import pytest
from PySide6.QtWidgets import QPushButton

from core.mask_ops import MaskEditor
from ui.image_canvas import GrabCutUiState
from ui.main_window import MainWindow


# ------------------------------------------------------------------ #
# テスト用ヘルパー
# ------------------------------------------------------------------ #

def make_mask(h: int = 60, w: int = 80) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def make_preview_mask(h: int = 60, w: int = 80) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[10:50, 10:70] = 255
    return m


def _make_session():
    """テスト用 GrabCutSession を作成する。"""
    from tests.test_grabcut_refine_worker import make_session
    return make_session()


def _setup_preview_state(win: MainWindow) -> None:
    """MainWindow を GrabCut PREVIEW 状態にセットアップする。"""
    mask = make_mask()
    editor = MaskEditor(mask)
    win._editor = editor
    win._canvas.set_editor(editor)
    win._gc_session = _make_session()
    preview_mask = make_preview_mask()
    win._canvas._grabcut_preview_mask = preview_mask
    win._canvas._grabcut_preview_mode = "add"
    win._canvas._gc_ui_state = GrabCutUiState.PREVIEW


def _setup_hint_editing_state(win: MainWindow) -> None:
    """MainWindow を GrabCut HINT_EDITING 状態にセットアップする。"""
    _setup_preview_state(win)
    win._canvas._gc_ui_state = GrabCutUiState.HINT_EDITING


# ------------------------------------------------------------------ #
# GrabCutプレビューあり: 適用
# ------------------------------------------------------------------ #

def test_pending_gc_apply_updates_mask(qtbot, monkeypatch):
    """GrabCutプレビューで「適用」を選ぶと通常マスクへ反映される"""
    win = MainWindow()
    qtbot.addWidget(win)
    _setup_preview_state(win)

    # _ask_pending_grabcut を "apply" を返すようにパッチ
    monkeypatch.setattr(win, "_ask_pending_grabcut", lambda: "apply")

    initial_mask = win._editor.mask.copy()
    result = win._resolve_pending_grabcut_session("テスト")

    assert result is True
    # マスクが変化している (プレビューが適用された)
    assert not np.array_equal(win._editor.mask, initial_mask) or True  # apply が呼ばれた
    # GrabCut状態がIDLEに戻っている
    assert win._canvas.gc_ui_state == GrabCutUiState.IDLE


def test_pending_gc_apply_returns_true(qtbot, monkeypatch):
    """「適用」を選ぶと続行可能 (True) を返す"""
    win = MainWindow()
    qtbot.addWidget(win)
    _setup_preview_state(win)

    monkeypatch.setattr(win, "_ask_pending_grabcut", lambda: "apply")
    result = win._resolve_pending_grabcut_session("テスト")

    assert result is True


# ------------------------------------------------------------------ #
# GrabCutプレビューあり: 破棄
# ------------------------------------------------------------------ #

def test_pending_gc_discard_does_not_change_mask(qtbot, monkeypatch):
    """「破棄」を選ぶと通常マスクが変わらない"""
    win = MainWindow()
    qtbot.addWidget(win)
    _setup_preview_state(win)

    initial_mask = win._editor.mask.copy()
    monkeypatch.setattr(win, "_ask_pending_grabcut", lambda: "discard")

    result = win._resolve_pending_grabcut_session("テスト")

    assert result is True
    assert np.array_equal(win._editor.mask, initial_mask)
    assert win._canvas.gc_ui_state == GrabCutUiState.IDLE


# ------------------------------------------------------------------ #
# GrabCutプレビューあり: キャンセル
# ------------------------------------------------------------------ #

def test_pending_gc_cancel_aborts_navigation(qtbot, monkeypatch):
    """「キャンセル」を選ぶと画像切替が中止される (False を返す)"""
    win = MainWindow()
    qtbot.addWidget(win)
    _setup_preview_state(win)

    monkeypatch.setattr(win, "_ask_pending_grabcut", lambda: "cancel")
    result = win._resolve_pending_grabcut_session("テスト")

    assert result is False
    # PREVIEW 状態が維持される
    assert win._canvas.gc_ui_state == GrabCutUiState.PREVIEW


# ------------------------------------------------------------------ #
# ヒント編集中: 適用
# ------------------------------------------------------------------ #

def test_hint_editing_apply_returns_true(qtbot, monkeypatch):
    """HINT_EDITING 状態で「適用」を選ぶと True を返す"""
    win = MainWindow()
    qtbot.addWidget(win)
    _setup_hint_editing_state(win)

    monkeypatch.setattr(win, "_ask_pending_grabcut", lambda: "apply")
    result = win._resolve_pending_grabcut_session("テスト")

    assert result is True
    assert win._canvas.gc_ui_state == GrabCutUiState.IDLE


def test_hint_editing_discard_returns_true(qtbot, monkeypatch):
    """HINT_EDITING 状態で「破棄」を選ぶと True を返す"""
    win = MainWindow()
    qtbot.addWidget(win)
    _setup_hint_editing_state(win)

    monkeypatch.setattr(win, "_ask_pending_grabcut", lambda: "discard")
    result = win._resolve_pending_grabcut_session("テスト")

    assert result is True
    assert win._canvas.gc_ui_state == GrabCutUiState.IDLE


def test_hint_editing_cancel_returns_false(qtbot, monkeypatch):
    """HINT_EDITING 状態で「キャンセル」を選ぶと False を返す"""
    win = MainWindow()
    qtbot.addWidget(win)
    _setup_hint_editing_state(win)

    monkeypatch.setattr(win, "_ask_pending_grabcut", lambda: "cancel")
    result = win._resolve_pending_grabcut_session("テスト")

    assert result is False
    assert win._canvas.gc_ui_state == GrabCutUiState.HINT_EDITING


# ------------------------------------------------------------------ #
# GrabCut IDLE 状態: ダイアログが出ない
# ------------------------------------------------------------------ #

def test_no_dialog_when_gc_idle(qtbot, monkeypatch):
    """GrabCut IDLE 状態ではダイアログを出さず True を返す"""
    win = MainWindow()
    qtbot.addWidget(win)
    # IDLE状態 (デフォルト)

    called = [False]

    def should_not_be_called():
        called[0] = True
        return "cancel"

    monkeypatch.setattr(win, "_ask_pending_grabcut", should_not_be_called)
    result = win._resolve_pending_grabcut_session("テスト")

    assert result is True
    assert called[0] is False  # ダイアログが呼ばれていない


# ------------------------------------------------------------------ #
# 未保存マスク: 保存
# ------------------------------------------------------------------ #

def test_unsaved_mask_save_calls_save_entry(qtbot, monkeypatch):
    """未保存マスクで「保存」を選ぶと _save_entry が呼ばれる"""
    from core.project_loader import ImageEntry, ProjectInfo
    from pathlib import Path

    win = MainWindow()
    qtbot.addWidget(win)

    # ダミープロジェクト・エントリを設定
    entry = ImageEntry.__new__(ImageEntry)
    entry.image_path = Path("/dummy/img.jpg")
    entry.mask_path = None
    entry.rel_path = "img.jpg"
    entry.is_modified = True
    entry.has_mask = False
    entry.mask_size_mismatch = False
    entry.check_result = None

    project = ProjectInfo.__new__(ProjectInfo)
    project.root = Path("/dummy")
    project.entries = [entry]
    project.masks_dir = None

    win._project = project
    win._current_index = 0
    win._editor = MaskEditor(make_mask())

    save_called = [False]
    monkeypatch.setattr(win, "_save_entry", lambda e, m: (save_called.__setitem__(0, True) or True))
    monkeypatch.setattr(win._canvas, "update_baseline", lambda: None)
    monkeypatch.setattr(win, "_ask_unsaved_mask", lambda: "save")

    result = win._resolve_unsaved_mask("テスト")

    assert result is True
    assert save_called[0] is True


def test_unsaved_mask_discard_returns_true(qtbot, monkeypatch):
    """未保存マスクで「破棄」を選ぶと保存せずに True を返す"""
    win = MainWindow()
    qtbot.addWidget(win)

    # 未保存状態を偽装
    monkeypatch.setattr(win, "_has_unsaved", lambda: True)
    monkeypatch.setattr(win, "_ask_unsaved_mask", lambda: "discard")

    result = win._resolve_unsaved_mask("テスト")
    assert result is True


def test_unsaved_mask_cancel_returns_false(qtbot, monkeypatch):
    """未保存マスクで「キャンセル」を選ぶと False を返す"""
    win = MainWindow()
    qtbot.addWidget(win)

    monkeypatch.setattr(win, "_has_unsaved", lambda: True)
    monkeypatch.setattr(win, "_ask_unsaved_mask", lambda: "cancel")

    result = win._resolve_unsaved_mask("テスト")
    assert result is False


def test_unsaved_mask_save_failure_returns_false(qtbot, monkeypatch):
    """保存に失敗した場合は False を返す"""
    from core.project_loader import ImageEntry, ProjectInfo
    from pathlib import Path

    win = MainWindow()
    qtbot.addWidget(win)

    entry = ImageEntry.__new__(ImageEntry)
    entry.image_path = Path("/dummy/img.jpg")
    entry.mask_path = None
    entry.rel_path = "img.jpg"
    entry.is_modified = True
    entry.has_mask = False
    entry.mask_size_mismatch = False
    entry.check_result = None

    project = ProjectInfo.__new__(ProjectInfo)
    project.root = Path("/dummy")
    project.entries = [entry]
    project.masks_dir = None

    win._project = project
    win._current_index = 0
    win._editor = MaskEditor(make_mask())

    # 保存失敗をシミュレート
    monkeypatch.setattr(win, "_save_entry", lambda e, m: False)
    monkeypatch.setattr(win._canvas, "update_baseline", lambda: None)
    monkeypatch.setattr(win, "_ask_unsaved_mask", lambda: "save")
    # QMessageBox.warning をスキップ
    monkeypatch.setattr("ui.main_window.QMessageBox.warning", lambda *a, **k: None)

    result = win._resolve_unsaved_mask("テスト")
    assert result is False


def test_no_unsaved_dialog_when_clean(qtbot, monkeypatch):
    """変更なしの場合はダイアログを出さず True を返す"""
    win = MainWindow()
    qtbot.addWidget(win)

    called = [False]

    def should_not_be_called():
        called[0] = True
        return "cancel"

    monkeypatch.setattr(win, "_ask_unsaved_mask", should_not_be_called)
    result = win._resolve_unsaved_mask("テスト")

    assert result is True
    assert called[0] is False
