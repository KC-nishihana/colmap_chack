"""
QSettingsを使ったアプリ設定の保存・復元。
MainWindowへQSettings処理を直接大量に書かず、設定読み書きを分離する。
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QByteArray, QSettings

from core.version import APP_NAME, APP_VERSION, SETTINGS_SCHEMA_VERSION

_log = logging.getLogger(__name__)

# 設定キーとデフォルト値の対応表
_DEFAULTS: dict[str, Any] = {
    "edit/brush_size": 20,
    "edit/mask_opacity": 45,
    "edit/mask_visible": True,
    "edit/diff_visible": False,
    "edit/morph_kernel_size": 5,
    "edit/min_area": 100,
    "grabcut/iter_count": 5,
    "grabcut/post_dilate": False,
    "grabcut/post_erode": False,
    "grabcut/post_kernel_size": 3,
    "grabcut/use_downscale": True,
    "grabcut/max_size": 2048,
    "grabcut/use_existing_mask": False,
    "grabcut/hint_radius": 20,
    "grabcut/last_hint_type": 0,   # 0=fg, 1=bg, 2=erase
    "window/right_tab_index": 0,
    "file/last_folder": "",
}

# 各設定の有効範囲 (下限, 上限) - 数値型のみ
_CLAMPS: dict[str, tuple[int, int]] = {
    "edit/brush_size":        (1, 300),
    "edit/mask_opacity":      (0, 100),
    "edit/morph_kernel_size": (1, 99),
    "edit/min_area":          (1, 100000),
    "grabcut/iter_count":     (1, 20),
    "grabcut/post_kernel_size": (1, 15),
    "grabcut/max_size":       (512, 4096),
    "grabcut/hint_radius":    (1, 300),
    "grabcut/last_hint_type": (0, 2),
    "window/right_tab_index": (0, 2),
}


class AppSettings:
    """
    QSettingsのラッパー。設定の保存・読込・バリデーション・リセットを提供する。

    テスト用に filepath を指定するとINIファイルを直接使用する (実ユーザー設定を汚染しない)。
    filepath を省略すると QSettings() のデフォルト (レジストリまたは ~/.config/) を使用する。
    """

    def __init__(self, filepath: str | None = None) -> None:
        if filepath is not None:
            self._settings = QSettings(filepath, QSettings.Format.IniFormat)
        else:
            self._settings = QSettings()

    # ------------------------------------------------------------------ #
    # 基本 API
    # ------------------------------------------------------------------ #

    def get(self, key: str, default: Any = None) -> Any:
        """キーの値を取得する。存在しない場合は default (または組み込みデフォルト) を返す。"""
        if default is None:
            default = _DEFAULTS.get(key)
        raw = self._settings.value(key, default)
        return self._coerce_and_clamp(key, raw, default)

    def set(self, key: str, value: Any) -> None:
        """キーに値をセットする (即時保存ではない)。"""
        self._settings.setValue(key, value)

    def sync(self) -> None:
        """設定をディスクに書き込む。"""
        self._settings.sync()

    # ------------------------------------------------------------------ #
    # まとめて読み書き
    # ------------------------------------------------------------------ #

    def load(self) -> dict[str, Any]:
        """全設定を辞書で返す。存在しないキーはデフォルト値を使用する。"""
        result: dict[str, Any] = {}
        for key, default in _DEFAULTS.items():
            result[key] = self.get(key, default)
        return result

    def save(self, values: dict[str, Any]) -> None:
        """辞書から一括保存する。sync() も呼ぶ。"""
        for key, value in values.items():
            self._settings.setValue(key, value)
        self._settings.sync()
        _log.debug("設定を保存しました (%d件)", len(values))

    # ------------------------------------------------------------------ #
    # ウィンドウジオメトリ・スプリッター
    # ------------------------------------------------------------------ #

    def save_bytes(self, key: str, data: QByteArray | bytes | None) -> None:
        """QByteArray (geometry, splitter state) を保存する。"""
        if data is not None:
            self._settings.setValue(key, data)

    def load_bytes(self, key: str) -> QByteArray | None:
        """QByteArray を読み込む。存在しない場合は None を返す。"""
        val = self._settings.value(key)
        if val is None:
            return None
        if isinstance(val, QByteArray):
            return val
        if isinstance(val, (bytes, bytearray)):
            return QByteArray(val)
        return None

    # ------------------------------------------------------------------ #
    # リセット
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """全設定を消去する。次回起動時にデフォルト値が使われる。"""
        self._settings.clear()
        self._settings.sync()
        _log.info("設定を初期化しました")

    # ------------------------------------------------------------------ #
    # 内部ヘルパー
    # ------------------------------------------------------------------ #

    def _coerce_and_clamp(self, key: str, value: Any, default: Any) -> Any:
        """型変換と範囲クランプを行う。変換失敗時はデフォルト値を返す。"""
        if value is None:
            return default

        # bool型のデフォルトがあるキーは bool へ変換
        if isinstance(default, bool):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            try:
                return bool(int(value))
            except (ValueError, TypeError):
                return default

        # str型はそのまま
        if isinstance(default, str):
            return str(value) if value is not None else default

        # 数値型
        if key in _CLAMPS:
            lo, hi = _CLAMPS[key]
            try:
                v = int(value)
                return max(lo, min(hi, v))
            except (ValueError, TypeError):
                return default

        # その他はデフォルトと同じ型へ変換
        try:
            return type(default)(value)
        except (ValueError, TypeError):
            return default
