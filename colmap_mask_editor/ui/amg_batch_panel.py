"""
V0.8 GUI: 「全画像自動分割」サブタブのパネル。

操作はシグナルで MainWindow へ通知し、パネル自身は Worker と直接通信しない
(torch / sam2 は import しない)。プリセット変更後に個別値を変更したら「カスタム」表示。
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ai.amg_manifest import DEFAULT_PRESET, GENERATOR_KEYS, PRESETS, match_preset
from ai.model_registry import MODEL_ORDER, get_model

# 解析対象スコープ
SCOPE_ITEMS = [
    ("すべての画像", "all"),
    ("選択した画像", "selected"),
    ("未処理画像", "unprocessed"),
    ("古い解析結果", "stale"),
    ("失敗画像", "failed"),
    ("現在画像のみ", "current"),
]
DEFAULT_SCOPE = "unprocessed"

PRESET_ITEMS = [
    ("高速", "fast"),
    ("標準", "standard"),
    ("詳細", "detailed"),
    ("カスタム", "custom"),
]

# AMG が使うモデル (SAM 2.1 Small / Base Plus)
_AMG_MODELS = [m for m in MODEL_ORDER]


class AmgBatchPanel(QWidget):
    preflight_requested = Signal()
    start_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    cancel_requested = Signal()
    retry_failed_requested = Signal()
    review_requested = Signal()
    validate_cache_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._loading = False
        self._build_ui()
        self._apply_preset(DEFAULT_PRESET)
        self.set_running(False, paused=False, has_results=False)

    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # 解析対象・モデル・プリセット
        cfg = QGroupBox("解析対象")
        form = QFormLayout(cfg)
        form.setContentsMargins(8, 4, 8, 4)
        form.setVerticalSpacing(3)
        self._scope = QComboBox()
        for label, val in SCOPE_ITEMS:
            self._scope.addItem(label, val)
        self._scope.setCurrentIndex(self._index_of(self._scope, DEFAULT_SCOPE))
        form.addRow("対象", self._scope)

        self._model = QComboBox()
        for mid in _AMG_MODELS:
            self._model.addItem(get_model(mid).display_name, mid)
        form.addRow("モデル", self._model)

        self._preset = QComboBox()
        for label, val in PRESET_ITEMS:
            self._preset.addItem(label, val)
        self._preset.currentIndexChanged.connect(self._on_preset_changed)
        form.addRow("プリセット", self._preset)
        root.addWidget(cfg)

        # 詳細設定 (折りたたみ)
        self._adv = QGroupBox("詳細設定")
        self._adv.setCheckable(True)
        self._adv.setChecked(False)
        adv_outer = QVBoxLayout(self._adv)
        adv_outer.setContentsMargins(8, 4, 8, 4)
        adv_outer.setSpacing(0)
        self._adv_body = QWidget()
        af = QFormLayout(self._adv_body)
        af.setContentsMargins(0, 0, 0, 0)
        af.setVerticalSpacing(2)

        self._pps = self._spin(4, 64, 1)
        self._ppb = self._spin(1, 256, 1)
        self._iou = self._dspin(0.0, 1.0, 0.01)
        self._stab = self._dspin(0.0, 1.0, 0.01)
        self._nms = self._dspin(0.0, 1.0, 0.01)
        self._crop_layers = self._spin(0, 4, 1)
        self._crop_ds = self._spin(1, 4, 1)
        self._min_area = self._spin(0, 100000, 10)
        self._use_m2m = QCheckBox("use_m2m")
        self._multimask = QCheckBox("multimask_output")
        af.addRow("points_per_side", self._pps)
        af.addRow("points_per_batch", self._ppb)
        af.addRow("pred_iou_thresh", self._iou)
        af.addRow("stability_score_thresh", self._stab)
        af.addRow("box_nms_thresh", self._nms)
        af.addRow("crop_n_layers", self._crop_layers)
        af.addRow("crop_n_points_downscale", self._crop_ds)
        af.addRow("min_mask_region_area", self._min_area)
        af.addRow(self._use_m2m)
        af.addRow(self._multimask)
        self._oom_retry = QCheckBox("OOM時に points_per_batch を縮小して再試行")
        self._oom_retry.setChecked(True)
        af.addRow(self._oom_retry)
        self._force = QCheckBox("強制再解析 (既存結果を置き換え)")
        af.addRow(self._force)

        # 個別値変更で「カスタム」へ
        for w in (self._pps, self._ppb, self._crop_layers, self._crop_ds, self._min_area):
            w.valueChanged.connect(self._on_detail_changed)
        for w in (self._iou, self._stab, self._nms):
            w.valueChanged.connect(self._on_detail_changed)
        for w in (self._use_m2m, self._multimask):
            w.toggled.connect(self._on_detail_changed)

        adv_outer.addWidget(self._adv_body)
        self._adv.toggled.connect(self._adv_body.setVisible)
        self._adv_body.setVisible(False)
        root.addWidget(self._adv)

        # 事前確認サマリ
        self._summary = QLabel("対象を選び「事前確認」を押してください")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color:#9cf;")
        root.addWidget(self._summary)

        # 操作
        ops = QGroupBox("操作")
        opl = QVBoxLayout(ops)
        opl.setContentsMargins(8, 4, 8, 4)
        opl.setSpacing(3)
        row1 = QHBoxLayout()
        self._btn_preflight = QPushButton("事前確認")
        self._btn_start = QPushButton("解析開始")
        self._btn_pause = QPushButton("一時停止")
        self._btn_resume = QPushButton("再開")
        self._btn_cancel = QPushButton("キャンセル")
        self._btn_preflight.clicked.connect(self.preflight_requested)
        self._btn_start.clicked.connect(self.start_requested)
        self._btn_pause.clicked.connect(self.pause_requested)
        self._btn_resume.clicked.connect(self.resume_requested)
        self._btn_cancel.clicked.connect(self.cancel_requested)
        for b in (self._btn_preflight, self._btn_start, self._btn_pause,
                  self._btn_resume, self._btn_cancel):
            row1.addWidget(b)
        opl.addLayout(row1)
        row2 = QHBoxLayout()
        self._btn_retry = QPushButton("失敗画像を再処理")
        self._btn_review = QPushButton("レビュー")
        self._btn_validate = QPushButton("キャッシュ検証")
        self._btn_retry.clicked.connect(self.retry_failed_requested)
        self._btn_review.clicked.connect(self.review_requested)
        self._btn_validate.clicked.connect(self.validate_cache_requested)
        for b in (self._btn_retry, self._btn_review, self._btn_validate):
            row2.addWidget(b)
        opl.addLayout(row2)
        root.addWidget(ops)

        # 進捗
        self._status = QLabel("状態: 待機")
        self._progress = QLabel("処理: 0 / 0")
        self._counts = QLabel("成功: 0  再利用: 0  失敗: 0")
        self._current = QLabel("現在: -")
        self._mem = QLabel("VRAM: - MB  RAM: - MB")
        for w in (self._status, self._progress, self._counts, self._current, self._mem):
            root.addWidget(w)
        root.addStretch(1)

    # ------------------------------------------------------------------ #
    # ウィジェット生成補助
    # ------------------------------------------------------------------ #

    @staticmethod
    def _spin(lo, hi, step):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        return s

    @staticmethod
    def _dspin(lo, hi, step):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(2)
        return s

    @staticmethod
    def _index_of(combo: QComboBox, data) -> int:
        for i in range(combo.count()):
            if combo.itemData(i) == data:
                return i
        return 0

    # ------------------------------------------------------------------ #
    # プリセット
    # ------------------------------------------------------------------ #

    def _on_preset_changed(self) -> None:
        name = self._preset.currentData()
        if name in PRESETS:
            self._apply_preset(name)

    def _apply_preset(self, name: str) -> None:
        if name not in PRESETS:
            return
        self._loading = True
        p = PRESETS[name]
        self._pps.setValue(p["points_per_side"])
        self._ppb.setValue(p["points_per_batch"])
        self._iou.setValue(p["pred_iou_thresh"])
        self._stab.setValue(p["stability_score_thresh"])
        self._nms.setValue(p["box_nms_thresh"])
        self._crop_layers.setValue(p["crop_n_layers"])
        self._crop_ds.setValue(p["crop_n_points_downscale_factor"])
        self._min_area.setValue(p["min_mask_region_area"])
        self._use_m2m.setChecked(p["use_m2m"])
        self._multimask.setChecked(p["multimask_output"])
        self._preset.setCurrentIndex(self._index_of(self._preset, name))
        self._loading = False

    def _on_detail_changed(self) -> None:
        if self._loading:
            return
        # 現在値がいずれかのプリセットと一致すればその名、なければカスタム
        name = match_preset(self.generator_settings())
        self._loading = True
        self._preset.setCurrentIndex(self._index_of(self._preset, name))
        self._loading = False

    # ------------------------------------------------------------------ #
    # MainWindow から呼ぶ API
    # ------------------------------------------------------------------ #

    def generator_settings(self) -> dict:
        return {
            "points_per_side": self._pps.value(),
            "points_per_batch": self._ppb.value(),
            "pred_iou_thresh": round(self._iou.value(), 4),
            "stability_score_thresh": round(self._stab.value(), 4),
            "box_nms_thresh": round(self._nms.value(), 4),
            "crop_n_layers": self._crop_layers.value(),
            "crop_n_points_downscale_factor": self._crop_ds.value(),
            "min_mask_region_area": self._min_area.value(),
            "use_m2m": self._use_m2m.isChecked(),
            "multimask_output": self._multimask.isChecked(),
        }

    def options(self) -> dict:
        return {
            "scope": self._scope.currentData(),
            "model_id": self._model.currentData(),
            "preset": self._preset.currentData(),
            "settings": self.generator_settings(),
            "oom_retry": self._oom_retry.isChecked(),
            "force": self._force.isChecked(),
        }

    def load_defaults(self, settings) -> None:
        """QSettings (AppSettings) から amg/* の既定を読み込む。"""
        self._loading = True
        self._scope.setCurrentIndex(self._index_of(self._scope, settings.get("amg/default_scope")))
        self._model.setCurrentIndex(self._index_of(self._model, settings.get("amg/default_model")))
        self._oom_retry.setChecked(bool(settings.get("amg/oom_retry")))
        self._loading = False
        preset = settings.get("amg/default_preset")
        if preset in PRESETS:
            self._apply_preset(preset)
        else:
            self._apply_preset(DEFAULT_PRESET)

    def set_summary(self, text: str) -> None:
        self._summary.setText(text)

    def set_progress(self, processed: int, total: int) -> None:
        self._progress.setText(f"処理: {processed} / {total}")

    def set_counts(self, succeeded: int, reused: int, failed: int) -> None:
        self._counts.setText(f"成功: {succeeded}  再利用: {reused}  失敗: {failed}")

    def set_current(self, image_key: str, segment_count=None) -> None:
        extra = f"  候補数: {segment_count}" if segment_count is not None else ""
        self._current.setText(f"現在: {image_key}{extra}")

    def set_memory(self, vram_mb: int, ram_mb: int = 0) -> None:
        self._mem.setText(f"VRAM: {vram_mb} MB  RAM: {ram_mb} MB")

    def set_status_text(self, text: str) -> None:
        self._status.setText(f"状態: {text}")

    def set_running(self, running: bool, *, paused: bool, has_results: bool) -> None:
        idle = not running
        self._btn_start.setEnabled(idle)
        self._btn_preflight.setEnabled(idle)
        self._btn_retry.setEnabled(idle)
        self._btn_validate.setEnabled(idle)
        self._btn_review.setEnabled(has_results)
        self._btn_pause.setEnabled(running and not paused)
        self._btn_resume.setEnabled(running and paused)
        self._btn_cancel.setEnabled(running)
        for w in (self._scope, self._model, self._preset, self._adv):
            w.setEnabled(idle)
