"""v0.7 伝播設定とスキーマ v2->v3 移行のテスト (torch不要)。"""

from PySide6.QtCore import QSettings

from core.app_settings import AppSettings
from core.version import SETTINGS_SCHEMA_VERSION


def test_schema_version_is_current():
    assert SETTINGS_SCHEMA_VERSION == 5  # v0.9


def test_propagation_defaults(tmp_path):
    s = AppSettings(str(tmp_path / "s.ini"))
    assert s.get("propagation/default_direction") == "both"
    assert s.get("propagation/default_count") == 10
    assert s.get("propagation/order_mode") == 0
    assert s.get("propagation/offload_video_to_cpu") is True
    assert s.get("propagation/offload_state_to_cpu") is False
    assert s.get("propagation/jpeg_quality") == 95
    assert s.get("propagation/last_apply_mode") == "add"


def test_propagation_clamps(tmp_path):
    s = AppSettings(str(tmp_path / "s.ini"))
    s.set("propagation/default_count", 99999)
    assert s.get("propagation/default_count") == 500   # 上限クランプ
    s.set("propagation/jpeg_quality", 0)
    assert s.get("propagation/jpeg_quality") == 1       # 下限クランプ
    s.set("propagation/order_mode", 9)
    assert s.get("propagation/order_mode") == 3


def test_migrate_v2_to_v3_preserves_existing(tmp_path):
    f = str(tmp_path / "s.ini")
    raw = QSettings(f, QSettings.Format.IniFormat)
    raw.setValue("meta/schema_version", 2)
    raw.setValue("edit/brush_size", 77)
    raw.setValue("ai/precision", "fp16")
    raw.setValue("grabcut/iter_count", 9)
    raw.sync()

    app = AppSettings(f)
    assert app.schema_version == SETTINGS_SCHEMA_VERSION
    # 既存 v2 設定は保持される
    assert app.get("edit/brush_size") == 77
    assert app.get("ai/precision") == "fp16"
    assert app.get("grabcut/iter_count") == 9
    # 新規 v0.7 キーはデフォルトで利用可能
    assert app.get("propagation/default_direction") == "both"
    assert app.get("propagation/max_frames") == 100
