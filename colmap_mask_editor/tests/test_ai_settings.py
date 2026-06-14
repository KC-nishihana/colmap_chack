"""
v0.6 AI設定とスキーマ移行 (v1 -> v2) のテスト。
"""

import sys

import pytest

from core.app_settings import AppSettings
from core.version import SETTINGS_SCHEMA_VERSION


def test_ai_defaults(tmp_path):
    s = AppSettings(filepath=str(tmp_path / "s.ini"))
    assert s.get("ai/enabled") is True
    assert s.get("ai/default_model") == "sam2.1_hiera_small"
    assert s.get("ai/precision") == "bf16"
    assert s.get("ai/device") == "cuda:0"
    assert s.get("ai/auto_start_worker") is False
    assert s.get("ai/auto_load_model") is False
    assert s.get("ai/auto_predict") is False


def test_python_executable_defaults_to_sys_executable(tmp_path):
    s = AppSettings(filepath=str(tmp_path / "s.ini"))
    assert s.get_ai_python_executable() == sys.executable


def test_python_executable_override(tmp_path):
    s = AppSettings(filepath=str(tmp_path / "s.ini"))
    s.save({"ai/python_executable": "C:/custom/python.exe"})
    assert s.get_ai_python_executable() == "C:/custom/python.exe"


def test_timeouts_clamped(tmp_path):
    s = AppSettings(filepath=str(tmp_path / "s.ini"))
    s.save({"ai/predict_timeout": 1})
    assert s.get("ai/predict_timeout") == 5  # 下限
    s.save({"ai/model_load_timeout": 99999})
    assert s.get("ai/model_load_timeout") == 1200  # 上限


def test_tab_index_clamp_allows_4_tabs(tmp_path):
    s = AppSettings(filepath=str(tmp_path / "s.ini"))
    s.save({"window/right_tab_index": 3})
    assert s.get("window/right_tab_index") == 3  # AIタブまで許可
    s.save({"window/right_tab_index": 99})
    assert s.get("window/right_tab_index") == 3  # 上限クランプ


def test_schema_version_is_current_for_new(tmp_path):
    s = AppSettings(filepath=str(tmp_path / "s.ini"))
    assert s.schema_version == SETTINGS_SCHEMA_VERSION == 3


def test_migration_preserves_v1_settings(tmp_path):
    """v1 設定 (AIキーなし) を作り、再読込で移行されても既存値が残る。"""
    from PySide6.QtCore import QSettings

    ini = str(tmp_path / "s.ini")
    # v1 を手動で再現: schema_version を書かず既存キーのみ保存
    qs = QSettings(ini, QSettings.Format.IniFormat)
    qs.setValue("edit/brush_size", 123)
    qs.setValue("grabcut/iter_count", 9)
    qs.sync()

    # AppSettings 生成で migrate が走る
    s = AppSettings(filepath=ini)
    assert s.schema_version == SETTINGS_SCHEMA_VERSION
    # 既存値が保持されている
    assert s.get("edit/brush_size") == 123
    assert s.get("grabcut/iter_count") == 9
    # 新規 AI キーはデフォルトで取得できる
    assert s.get("ai/enabled") is True
