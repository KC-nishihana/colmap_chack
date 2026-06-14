"""
バージョン整合テスト (v0.5.1)

APP_VERSION が正しく定義され、MainWindowタイトルと一致することを確認する。
"""

from pathlib import Path

import pytest

from core.version import APP_DISPLAY_NAME, APP_NAME, APP_VERSION


def test_app_version_is_0_7():
    """APP_VERSION が 0.7 である"""
    assert APP_VERSION == "0.7"


def test_app_display_name_contains_version():
    """APP_DISPLAY_NAME にバージョンが含まれる"""
    assert APP_VERSION in APP_DISPLAY_NAME
    assert APP_NAME in APP_DISPLAY_NAME


def test_main_window_title_contains_app_display_name(qtbot):
    """MainWindow のタイトルに APP_DISPLAY_NAME が含まれる"""
    from ui.main_window import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    assert APP_DISPLAY_NAME in win.windowTitle()


def test_readme_starts_with_v06():
    """README.md の先頭行が v0.6 を含む"""
    readme = Path(__file__).parent.parent / "README.md"
    if not readme.exists():
        pytest.skip("README.md が見つかりません")
    first_line = readme.read_text(encoding="utf-8").splitlines()[0]
    assert APP_VERSION in first_line, f"README 先頭行: {first_line!r}"
