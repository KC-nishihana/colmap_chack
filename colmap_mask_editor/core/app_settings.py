"""
QSettingsを使ったアプリ設定の保存・復元。
MainWindowへQSettings処理を直接大量に書かず、設定読み書きを分離する。
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from PySide6.QtCore import QByteArray, QSettings

from core.version import APP_NAME, APP_VERSION, SETTINGS_SCHEMA_VERSION

_log = logging.getLogger(__name__)

_SCHEMA_KEY = "meta/schema_version"

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

    # ----- v0.6 AIセグメンテーション (SAM 2.1) -----
    "ai/enabled": True,
    "ai/python_executable": "",       # 空文字 = sys.executable を使用
    "ai/default_model": "sam2.1_hiera_small",
    "ai/checkpoint_dir": "",          # 空文字 = models/sam2 を使用
    "ai/precision": "bf16",
    "ai/device": "cuda:0",
    "ai/auto_start_worker": False,
    "ai/auto_load_model": False,
    "ai/auto_predict": False,
    "ai/worker_start_timeout": 30,    # 秒
    "ai/model_load_timeout": 180,     # 秒
    "ai/image_encode_timeout": 120,   # 秒
    "ai/predict_timeout": 30,         # 秒
    "ai/last_prompt_type": 0,         # 0=正クリック, 1=負クリック, 2=矩形

    # ----- v0.7 画像伝播 (SAM 2.1 Video Predictor) -----
    "propagation/default_direction": "both",   # forward / backward / both
    "propagation/default_count": 10,           # 前後N枚
    "propagation/order_mode": 0,               # 0=現在の一覧順,1=COLMAP,2=ファイル名,3=撮影日時
    "propagation/max_frames": 100,
    "propagation/offload_video_to_cpu": True,
    "propagation/offload_state_to_cpu": False,
    "propagation/jpeg_quality": 95,
    "propagation/warn_empty": True,
    "propagation/warn_too_large_ratio": 80,    # %
    "propagation/warn_area_drop_ratio": 25,    # %
    "propagation/warn_area_growth_ratio": 400, # %
    "propagation/warn_component_count": 10,
    "propagation/warn_low_iou": 5,             # % (IoU*100)
    "propagation/last_apply_mode": "add",      # add / exclude / replace

    # ----- v0.8 全画像自動分割 (SAM 2.1 Automatic Mask Generator) -----
    "amg/default_scope": "unprocessed",        # all/selected/unprocessed/stale/failed/current
    "amg/default_preset": "fast",              # fast/standard/detailed/custom
    "amg/default_model": "sam2.1_hiera_small",
    "amg/points_per_side": 16,
    "amg/points_per_batch": 64,
    "amg/pred_iou_thresh": 0.85,
    "amg/stability_score_thresh": 0.95,
    "amg/box_nms_thresh": 0.7,
    "amg/crop_n_layers": 0,
    "amg/crop_n_points_downscale_factor": 1,
    "amg/min_mask_region_area": 100,
    "amg/use_m2m": False,
    "amg/multimask_output": True,
    "amg/oom_retry": True,
    "amg/review_overlay_opacity": 50,          # %
    "amg/review_min_area": 0,
    "amg/review_max_area_ratio": 100,          # %
    "amg/review_min_iou": 0,                   # % (IoU*100)
    "amg/review_min_stability": 0,             # % (stability*100)
    "amg/final_mask_mode": "exclude_remove",   # exclude_remove/keep_only/add_remove
    "amg/rle_decode_cache_size": 12,

    # ----- v0.9 完全被覆・階層型リージョン分割 -----
    "partition/default_preset": "coarse",      # coarse/standard/detailed/custom
    "partition/backend": "auto",               # auto/slic/grid_watershed
    "partition/working_max_side": 2048,        # 1024/2048/3072/4096/0(=原寸)
    "partition/base_region_count": 800,
    "partition/default_visible_count": 30,
    "partition/min_region_area_ratio": 10,     # 0.01% 単位 (10 = 0.10%)
    "partition/slic_region_size": 40,
    "partition/slic_ruler": 10,                # ruler*10 を実値とする (整数保存)
    "partition/watershed_seed_spacing": 32,
    "partition/weight_color": 30,              # 重みは 0..100 整数 (実値=/100)
    "partition/weight_texture": 10,
    "partition/weight_boundary": 30,
    "partition/weight_sam": 25,
    "partition/weight_size": 5,
    "partition/sam_sample_count": 64,
    "partition/sam_top_k": 4,
    "partition/overlay_opacity": 50,           # %
    "partition/show_boundaries": True,
    "partition/show_unreviewed": True,
    "partition/final_unreviewed_action": "ask",  # ask/keep/remove
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
    "window/right_tab_index": (0, 3),   # v0.6: 4タブ (編集/GrabCut/AI/保存)
    "ai/worker_start_timeout":  (5, 600),
    "ai/model_load_timeout":    (10, 1200),
    "ai/image_encode_timeout":  (5, 600),
    "ai/predict_timeout":       (5, 600),
    "ai/last_prompt_type":      (0, 2),
    "propagation/default_count":         (1, 500),
    "propagation/order_mode":            (0, 3),
    "propagation/max_frames":            (2, 1000),
    "propagation/jpeg_quality":          (1, 100),
    "propagation/warn_too_large_ratio":  (1, 100),
    "propagation/warn_area_drop_ratio":  (0, 100),
    "propagation/warn_area_growth_ratio": (100, 5000),
    "propagation/warn_component_count":  (1, 1000),
    "propagation/warn_low_iou":          (0, 100),
    "amg/points_per_side":               (4, 64),
    "amg/points_per_batch":              (1, 256),
    "amg/crop_n_layers":                 (0, 4),
    "amg/crop_n_points_downscale_factor": (1, 4),
    "amg/min_mask_region_area":          (0, 100000),
    "amg/review_overlay_opacity":        (0, 100),
    "amg/review_min_area":               (0, 100000000),
    "amg/review_max_area_ratio":         (0, 100),
    "amg/review_min_iou":                (0, 100),
    "amg/review_min_stability":          (0, 100),
    "amg/rle_decode_cache_size":         (1, 64),
    "partition/working_max_side":        (0, 8192),
    "partition/base_region_count":       (50, 20000),
    "partition/default_visible_count":   (2, 2000),
    "partition/min_region_area_ratio":   (0, 10000),
    "partition/slic_region_size":        (4, 400),
    "partition/slic_ruler":              (1, 1000),
    "partition/watershed_seed_spacing":  (4, 512),
    "partition/weight_color":            (0, 100),
    "partition/weight_texture":          (0, 100),
    "partition/weight_boundary":         (0, 100),
    "partition/weight_sam":              (0, 100),
    "partition/weight_size":             (0, 100),
    "partition/sam_sample_count":        (4, 256),
    "partition/sam_top_k":               (1, 32),
    "partition/overlay_opacity":         (0, 100),
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
        self.migrate()

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
    # スキーマ移行
    # ------------------------------------------------------------------ #

    def migrate(self) -> None:
        """
        設定スキーマを現行バージョンへ移行する。

        v1 -> v2 (v0.6): AIセグメンテーション設定キーを追加。
        v2 -> v3 (v0.7): 画像伝播 (propagation/*) 設定キーを追加。
        v3 -> v4 (v0.8): 全画像自動分割 (amg/*) 設定キーを追加。
        v4 -> v5 (v0.9): 完全被覆・階層型リージョン分割 (partition/*) 設定キーを追加。
        いずれも追加のみ (破壊的変更なし)。既存キーはキー名変更がないため保持され、
        欠けているキーは get() がデフォルトを返すので明示書き込みは不要。
        既存ユーザー設定を失わないことが目的。schema_version のみ更新する。
        """
        raw = self._settings.value(_SCHEMA_KEY)
        if raw is None:
            # 既存キーが1つでもあれば v1 とみなす。完全に空なら現行版として記録。
            has_any = len(self._settings.allKeys()) > 0
            stored = 1 if has_any else SETTINGS_SCHEMA_VERSION
        else:
            try:
                stored = int(raw)
            except (ValueError, TypeError):
                stored = 1

        if stored >= SETTINGS_SCHEMA_VERSION:
            if raw is None:
                self._settings.setValue(_SCHEMA_KEY, SETTINGS_SCHEMA_VERSION)
                self._settings.sync()
            return

        _log.info("設定スキーマを移行: v%d -> v%d", stored, SETTINGS_SCHEMA_VERSION)
        # 追加のみ (破壊的変更なし)。schema_version を更新するだけで既存値は保持される。
        self._settings.setValue(_SCHEMA_KEY, SETTINGS_SCHEMA_VERSION)
        self._settings.sync()

    @property
    def schema_version(self) -> int:
        try:
            return int(self._settings.value(_SCHEMA_KEY, SETTINGS_SCHEMA_VERSION))
        except (ValueError, TypeError):
            return SETTINGS_SCHEMA_VERSION

    # ------------------------------------------------------------------ #
    # AI 設定のヘルパー
    # ------------------------------------------------------------------ #

    def get_ai_python_executable(self) -> str:
        """AI Worker 用 Python。空設定なら sys.executable を返す。"""
        val = self.get("ai/python_executable", "")
        return val if val else sys.executable

    def get_ai_checkpoint_dir(self) -> str:
        """チェックポイント配置ディレクトリ。空設定なら models/sam2。"""
        val = self.get("ai/checkpoint_dir", "")
        if val:
            return val
        from pathlib import Path
        return str(Path(__file__).resolve().parent.parent.parent / "models" / "sam2")

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
