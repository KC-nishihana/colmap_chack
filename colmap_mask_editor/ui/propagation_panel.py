"""
V0.7 GUI: 「画像伝播」サブタブのパネル。

操作はシグナルで MainWindow へ通知し、パネル自身は Worker と直接通信しない
(torch / sam2 は import しない)。状態に応じてボタンの有効/無効を切り替える。
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ai.model_registry import MODEL_ORDER, PRECISIONS, get_model
from ai.propagation_order import PropagationOrder
from ai.propagation_protocol import PropagationDirection
from ai.propagation_session import PropagationUiState

_ORDER_ITEMS = [
    ("現在の一覧順", PropagationOrder.CURRENT_LIST),
    ("COLMAP images.txt優先", PropagationOrder.COLMAP_PRIORITY),
    ("ファイル名順", PropagationOrder.FILE_NAME),
    ("撮影日時順", PropagationOrder.CAPTURE_TIME),
]
_DIR_ITEMS = [
    ("前後", PropagationDirection.BOTH),
    ("前へ", PropagationDirection.FORWARD),
    ("後ろへ", PropagationDirection.BACKWARD),
]


class PropagationPanel(QWidget):
    preflight_requested = Signal()
    start_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    cancel_requested = Signal()
    review_requested = Signal()
    discard_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()
        self.set_state(PropagationUiState.IDLE)

    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # 基準マスク
        basis = QGroupBox("基準マスク")
        bl = QVBoxLayout(basis)
        bl.setContentsMargins(8, 4, 8, 4)
        bl.setSpacing(2)
        self._rb_ai = QRadioButton("現在のAI候補を使用")
        self._rb_mask = QRadioButton("現在の通常マスクを使用")
        self._rb_ai.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self._rb_ai)
        grp.addButton(self._rb_mask)
        bl.addWidget(self._rb_ai)
        bl.addWidget(self._rb_mask)
        root.addWidget(basis)

        # 伝播範囲・順序 (基本)
        cfg = QGroupBox("伝播範囲・順序")
        form = QFormLayout(cfg)
        form.setContentsMargins(8, 4, 8, 4)
        form.setVerticalSpacing(3)
        self._dir = QComboBox()
        for label, val in _DIR_ITEMS:
            self._dir.addItem(label, val)
        form.addRow("方向", self._dir)

        self._count = QSpinBox()
        self._count.setRange(1, 500)
        self._count.setValue(10)
        form.addRow("前後N枚", self._count)

        self._order = QComboBox()
        for label, val in _ORDER_ITEMS:
            self._order.addItem(label, val)
        form.addRow("画像順序", self._order)
        root.addWidget(cfg)

        # 詳細設定 (モデル/精度/オフロード) — 既定は折りたたみ
        self._adv = QGroupBox("詳細設定")
        self._adv.setCheckable(True)
        self._adv.setChecked(False)
        adv_outer = QVBoxLayout(self._adv)
        adv_outer.setContentsMargins(8, 4, 8, 4)
        adv_outer.setSpacing(0)
        self._adv_body = QWidget()
        adv_form = QFormLayout(self._adv_body)
        adv_form.setContentsMargins(0, 0, 0, 0)
        adv_form.setVerticalSpacing(3)
        self._model = QComboBox()
        for mid in MODEL_ORDER:
            self._model.addItem(get_model(mid).display_name, mid)
        adv_form.addRow("モデル", self._model)
        self._precision = QComboBox()
        for p in PRECISIONS:
            self._precision.addItem(p, p)
        adv_form.addRow("精度", self._precision)
        self._max_frames = QSpinBox()
        self._max_frames.setRange(2, 1000)
        self._max_frames.setValue(100)
        adv_form.addRow("最大フレーム数", self._max_frames)
        self._offload_video = QCheckBox("フレームをCPUへオフロード")
        self._offload_video.setChecked(True)
        self._offload_state = QCheckBox("推論状態をCPUへオフロード")
        adv_form.addRow(self._offload_video)
        adv_form.addRow(self._offload_state)
        adv_outer.addWidget(self._adv_body)
        # 折りたたみ: チェックで本体ごと表示/非表示し縦領域を節約 (ラベルも隠れる)
        self._adv.toggled.connect(self._adv_body.setVisible)
        self._adv_body.setVisible(False)
        root.addWidget(self._adv)

        # 順序プレビュー
        order_box = QGroupBox("対象画像の順序 (開始前に確認)")
        ol = QVBoxLayout(order_box)
        ol.setContentsMargins(8, 4, 8, 4)
        ol.setSpacing(3)
        self._order_list = QListWidget()
        self._order_list.setMaximumHeight(96)
        ol.addWidget(self._order_list)
        self._btn_preflight = QPushButton("事前確認 (順序・検証)")
        self._btn_preflight.clicked.connect(self.preflight_requested)
        ol.addWidget(self._btn_preflight)
        root.addWidget(order_box)

        # 操作
        ops = QGroupBox("操作")
        opl = QVBoxLayout(ops)
        opl.setContentsMargins(8, 4, 8, 4)
        opl.setSpacing(3)
        row1 = QHBoxLayout()
        self._btn_start = QPushButton("伝播開始")
        self._btn_pause = QPushButton("一時停止")
        self._btn_resume = QPushButton("再開")
        self._btn_cancel = QPushButton("キャンセル")
        self._btn_start.clicked.connect(self.start_requested)
        self._btn_pause.clicked.connect(self.pause_requested)
        self._btn_resume.clicked.connect(self.resume_requested)
        self._btn_cancel.clicked.connect(self.cancel_requested)
        for b in (self._btn_start, self._btn_pause, self._btn_resume, self._btn_cancel):
            row1.addWidget(b)
        opl.addLayout(row1)
        row2 = QHBoxLayout()
        self._btn_review = QPushButton("結果レビュー")
        self._btn_discard = QPushButton("セッション破棄")
        self._btn_review.clicked.connect(self.review_requested)
        self._btn_discard.clicked.connect(self.discard_requested)
        row2.addWidget(self._btn_review)
        row2.addWidget(self._btn_discard)
        opl.addLayout(row2)
        root.addWidget(ops)

        # 進捗
        self._status = QLabel("状態: 待機")
        self._progress = QLabel("処理: 0 / 0")
        self._counts = QLabel("成功: 0  警告: 0  失敗: 0")
        self._vram = QLabel("VRAM: - MB")
        for w in (self._status, self._progress, self._counts, self._vram):
            root.addWidget(w)
        root.addStretch(1)

    # ------------------------------------------------------------------ #
    # MainWindow から呼ぶ API
    # ------------------------------------------------------------------ #

    def options(self) -> dict:
        return {
            "use_ai_candidate": self._rb_ai.isChecked(),
            "direction": self._dir.currentData(),
            "count": self._count.value(),
            "order_mode": self._order.currentData(),
            "model_id": self._model.currentData(),
            "precision": self._precision.currentData(),
            "max_frames": self._max_frames.value(),
            "offload_video_to_cpu": self._offload_video.isChecked(),
            "offload_state_to_cpu": self._offload_state.isChecked(),
        }

    def set_order_preview(self, lines: list[str]) -> None:
        self._order_list.clear()
        self._order_list.addItems(lines)

    def set_progress(self, processed: int, total: int) -> None:
        self._progress.setText(f"処理: {processed} / {total}")

    def set_counts(self, ok: int, warn: int, failed: int) -> None:
        self._counts.setText(f"成功: {ok}  警告: {warn}  失敗: {failed}")

    def set_vram(self, mb: int) -> None:
        self._vram.setText(f"VRAM: {mb} MB")

    def set_status_text(self, text: str) -> None:
        self._status.setText(f"状態: {text}")

    def set_state(self, state: PropagationUiState) -> None:
        running = state in (PropagationUiState.STAGING, PropagationUiState.INITIALIZING,
                            PropagationUiState.RUNNING)
        paused = state == PropagationUiState.PAUSED
        review = state == PropagationUiState.REVIEW
        idle = state in (PropagationUiState.IDLE, PropagationUiState.COMPLETED,
                         PropagationUiState.CANCELLED, PropagationUiState.ERROR)
        active = running or paused or state == PropagationUiState.CANCELLING

        self._btn_start.setEnabled(idle)
        self._btn_preflight.setEnabled(idle)
        self._btn_pause.setEnabled(running)
        self._btn_resume.setEnabled(paused)
        self._btn_cancel.setEnabled(active)
        self._btn_review.setEnabled(review)
        self._btn_discard.setEnabled(review or state == PropagationUiState.ERROR)
        # 設定変更は実行中不可
        for w in (self._dir, self._count, self._order, self._adv,
                  self._rb_ai, self._rb_mask):
            w.setEnabled(idle)
        self.set_status_text({
            PropagationUiState.IDLE: "待機",
            PropagationUiState.STAGING: "ステージング中",
            PropagationUiState.INITIALIZING: "初期化中",
            PropagationUiState.RUNNING: "伝播中",
            PropagationUiState.PAUSED: "一時停止中",
            PropagationUiState.CANCELLING: "キャンセル中 (現フレーム完了後に停止)",
            PropagationUiState.CANCELLED: "キャンセル済み",
            PropagationUiState.REVIEW: "レビュー可能",
            PropagationUiState.APPLYING: "適用中",
            PropagationUiState.COMPLETED: "完了",
            PropagationUiState.ERROR: "エラー",
        }.get(state, str(state)))
