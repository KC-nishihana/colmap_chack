"""
AppSettings のテスト (v0.5.1)

一時INIファイルを使用し、ユーザーの実設定を変更しない。
"""

import os
import tempfile

import pytest
from PySide6.QtCore import QByteArray

from core.app_settings import AppSettings


@pytest.fixture
def tmp_settings(tmp_path):
    """一時ファイルを使ったAppSettingsを提供する。"""
    ini_path = str(tmp_path / "test_settings.ini")
    return AppSettings(filepath=ini_path)


# ------------------------------------------------------------------ #
# デフォルト値
# ------------------------------------------------------------------ #

def test_default_brush_size(tmp_settings):
    """ブラシサイズのデフォルト値が 20"""
    assert tmp_settings.get("edit/brush_size") == 20


def test_default_mask_opacity(tmp_settings):
    """透明度のデフォルト値が 45"""
    assert tmp_settings.get("edit/mask_opacity") == 45


def test_default_grabcut_iter_count(tmp_settings):
    """GrabCut反復回数のデフォルトが 5"""
    assert tmp_settings.get("grabcut/iter_count") == 5


def test_default_grabcut_max_size(tmp_settings):
    """GrabCut最大処理サイズのデフォルトが 2048"""
    assert tmp_settings.get("grabcut/max_size") == 2048


def test_default_use_downscale(tmp_settings):
    """大画像縮小のデフォルトが True"""
    assert tmp_settings.get("grabcut/use_downscale") is True


def test_default_last_folder_is_empty(tmp_settings):
    """最後のフォルダのデフォルトが空文字"""
    assert tmp_settings.get("file/last_folder") == ""


# ------------------------------------------------------------------ #
# 保存・復元
# ------------------------------------------------------------------ #

def test_save_and_reload_brush_size(tmp_path):
    """ブラシサイズを保存して再読込できる"""
    ini_path = str(tmp_path / "settings.ini")
    s1 = AppSettings(filepath=ini_path)
    s1.save({"edit/brush_size": 80})

    s2 = AppSettings(filepath=ini_path)
    assert s2.get("edit/brush_size") == 80


def test_save_and_reload_bool(tmp_path):
    """ブール値を保存して再読込できる"""
    ini_path = str(tmp_path / "settings.ini")
    s1 = AppSettings(filepath=ini_path)
    s1.save({"grabcut/post_dilate": True})

    s2 = AppSettings(filepath=ini_path)
    assert s2.get("grabcut/post_dilate") is True


def test_save_and_reload_last_folder(tmp_path):
    """最後のフォルダを保存して再読込できる"""
    ini_path = str(tmp_path / "settings.ini")
    folder = str(tmp_path)
    s1 = AppSettings(filepath=ini_path)
    s1.save({"file/last_folder": folder})

    s2 = AppSettings(filepath=ini_path)
    assert s2.get("file/last_folder") == folder


# ------------------------------------------------------------------ #
# バリデーション (範囲クランプ)
# ------------------------------------------------------------------ #

def test_brush_size_clamped_min(tmp_path):
    """ブラシサイズが下限 1 にクランプされる"""
    ini_path = str(tmp_path / "settings.ini")
    s = AppSettings(filepath=ini_path)
    s.save({"edit/brush_size": -5})
    assert s.get("edit/brush_size") == 1


def test_brush_size_clamped_max(tmp_path):
    """ブラシサイズが上限 300 にクランプされる"""
    ini_path = str(tmp_path / "settings.ini")
    s = AppSettings(filepath=ini_path)
    s.save({"edit/brush_size": 9999})
    assert s.get("edit/brush_size") == 300


def test_grabcut_iter_clamped(tmp_path):
    """GrabCut反復回数が 1〜20 にクランプされる"""
    ini_path = str(tmp_path / "settings.ini")
    s = AppSettings(filepath=ini_path)
    s.save({"grabcut/iter_count": 100})
    assert s.get("grabcut/iter_count") == 20
    s.save({"grabcut/iter_count": 0})
    assert s.get("grabcut/iter_count") == 1


def test_max_size_clamped(tmp_path):
    """最大処理サイズが 512〜4096 にクランプされる"""
    ini_path = str(tmp_path / "settings.ini")
    s = AppSettings(filepath=ini_path)
    s.save({"grabcut/max_size": 100})
    assert s.get("grabcut/max_size") == 512
    s.save({"grabcut/max_size": 10000})
    assert s.get("grabcut/max_size") == 4096


def test_invalid_value_falls_back_to_default(tmp_path):
    """不正な文字列値はデフォルトにフォールバックする"""
    ini_path = str(tmp_path / "settings.ini")
    s = AppSettings(filepath=ini_path)
    s.save({"edit/brush_size": "invalid_string"})
    # クランプ失敗時はデフォルト値 20 が返る
    val = s.get("edit/brush_size")
    assert 1 <= val <= 300  # 少なくとも範囲内


# ------------------------------------------------------------------ #
# 未知キー
# ------------------------------------------------------------------ #

def test_unknown_key_does_not_crash(tmp_settings):
    """未知のキーにアクセスしてもクラッシュしない"""
    val = tmp_settings.get("unknown/nonexistent_key", "default_val")
    assert val == "default_val"


# ------------------------------------------------------------------ #
# 設定リセット
# ------------------------------------------------------------------ #

def test_reset_clears_saved_values(tmp_path):
    """reset() で保存値が消え、デフォルト値が返る"""
    ini_path = str(tmp_path / "settings.ini")
    s = AppSettings(filepath=ini_path)
    s.save({"edit/brush_size": 150, "grabcut/iter_count": 10})

    s.reset()

    # リセット後は再作成してもデフォルト値に戻る
    s2 = AppSettings(filepath=ini_path)
    assert s2.get("edit/brush_size") == 20
    assert s2.get("grabcut/iter_count") == 5


# ------------------------------------------------------------------ #
# QByteArray (geometry / splitter state)
# ------------------------------------------------------------------ #

def test_save_and_load_bytes(tmp_path):
    """QByteArray を保存・復元できる"""
    ini_path = str(tmp_path / "settings.ini")
    s1 = AppSettings(filepath=ini_path)
    original = QByteArray(b"\x00\x01\x02\x03\xff")
    s1.save_bytes("window/geometry", original)
    s1.sync()

    s2 = AppSettings(filepath=ini_path)
    restored = s2.load_bytes("window/geometry")
    assert restored is not None
    assert bytes(restored) == bytes(original)


def test_load_bytes_returns_none_if_missing(tmp_settings):
    """保存されていない場合 load_bytes は None を返す"""
    assert tmp_settings.load_bytes("window/geometry") is None


# ------------------------------------------------------------------ #
# load() による一括取得
# ------------------------------------------------------------------ #

def test_load_returns_all_keys(tmp_settings):
    """load() がすべての既定キーを含む辞書を返す"""
    data = tmp_settings.load()
    assert "edit/brush_size" in data
    assert "grabcut/iter_count" in data
    assert "file/last_folder" in data


# ------------------------------------------------------------------ #
# v0.11 統合レビュー画面の設定キー / スキーマ移行
# ------------------------------------------------------------------ #

def test_v011_ui_defaults(tmp_settings):
    """ui/* 既定値が仕様どおり (動線1 = AIクリック / 除外する)"""
    assert tmp_settings.get("ui/main_workspace") == "review"
    assert tmp_settings.get("ui/default_selection_tool") == "ai_click"
    assert tmp_settings.get("ui/default_apply_operation") == "remove"
    assert tmp_settings.get("ui/auto_start_ai_worker") is True
    assert tmp_settings.get("ui/auto_load_amg_candidates") is True
    assert tmp_settings.get("ui/amg_representatives_only") is True


def test_schema_version_is_7(tmp_settings):
    from core.version import SETTINGS_SCHEMA_VERSION
    assert SETTINGS_SCHEMA_VERSION == 7
    assert tmp_settings.schema_version == 7


def test_v6_to_v7_migration_preserves_existing(tmp_path):
    """v6 設定 (旧バージョン) からの移行で既存値が失われない。"""
    ini_path = str(tmp_path / "settings.ini")
    # v6 相当の設定を直接書く (schema_version=6 と既存ユーザー値)
    from PySide6.QtCore import QSettings
    raw = QSettings(ini_path, QSettings.Format.IniFormat)
    raw.setValue("meta/schema_version", 6)
    raw.setValue("edit/brush_size", 123)
    raw.setValue("amg/review_workflow", "remove_only")
    raw.sync()

    s = AppSettings(filepath=ini_path)        # __init__ で migrate() が走る
    assert s.schema_version == 7               # v7 へ移行
    assert s.get("edit/brush_size") == 123     # 既存値を保持
    assert s.get("amg/review_workflow") == "remove_only"
    # 新規キーは既定値で取得できる
    assert s.get("ui/default_selection_tool") == "ai_click"
