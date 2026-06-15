"""
V0.9: 完全被覆リージョンの設定・生成パネル (PySide6)。

粒度プリセット (粗い/標準/詳細/カスタム)、バックエンド、作業解像度、目標表示
リージョン数を設定し「生成」を要求する。重い分割処理は CPU 専用 QProcess Worker
で実行するため、このパネルは設定 dict を signal で渡すだけ (torch/sam2 を import
しない)。
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# 粒度プリセット: (base_region_count, default_visible_count, min_region_area_ratio[0.01%])
GRANULARITY_PRESETS = {
    "coarse": (800, 30, 10),
    "standard": (1500, 70, 5),
    "detailed": (3000, 150, 2),
}
GRANULARITY_LABELS = [("粗い", "coarse"), ("標準", "standard"),
                      ("詳細", "detailed"), ("カスタム", "custom")]
BACKEND_LABELS = [("自動 (AUTO)", "auto"), ("SLICO", "slic"),
                  ("Grid Watershed", "grid_watershed")]
WORKING_SIDES = [("1024", 1024), ("2048", 2048), ("3072", 3072),
                 ("4096", 4096), ("原寸", 0)]


class PartitionPanel(QWidget):
    generate_requested = Signal(dict)
    cancel_requested = Signal()
    open_review_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        intro = QLabel(
            "全画素を重複なくリージョンへ割り当て、粗い階層から判断します。\n"
            "SAM 候補は統合ヒントとしてのみ使用します (最終領域にはしません)。")
        intro.setWordWrap(True)
        root.addWidget(intro)

        box = QGroupBox("粒度・バックエンド")
        form = QFormLayout(box)

        self.granularity = QComboBox()
        for label, key in GRANULARITY_LABELS:
            self.granularity.addItem(label, key)
        self.granularity.currentIndexChanged.connect(self._on_granularity)
        form.addRow("粒度", self.granularity)

        self.backend = QComboBox()
        for label, key in BACKEND_LABELS:
            self.backend.addItem(label, key)
        form.addRow("バックエンド", self.backend)

        self.working_side = QComboBox()
        for label, val in WORKING_SIDES:
            self.working_side.addItem(label, val)
        self.working_side.setCurrentIndex(1)  # 2048
        form.addRow("作業解像度(長辺)", self.working_side)

        self.visible_count = QSpinBox()
        self.visible_count.setRange(2, 2000)
        self.visible_count.setValue(30)
        form.addRow("目標表示リージョン数", self.visible_count)

        self.base_count = QSpinBox()
        self.base_count.setRange(50, 20000)
        self.base_count.setValue(800)
        form.addRow("基礎リージョン数", self.base_count)

        root.addWidget(box)

        self.generate_btn = QPushButton("完全被覆リージョンを生成")
        self.generate_btn.clicked.connect(self._emit_generate)
        root.addWidget(self.generate_btn)

        self.cancel_btn = QPushButton("キャンセル")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_requested)
        root.addWidget(self.cancel_btn)

        self.review_btn = QPushButton("レビューを開く")
        self.review_btn.clicked.connect(self.open_review_requested)
        root.addWidget(self.review_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        root.addWidget(self.progress)

        self.status = QLabel("未生成")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        root.addStretch(1)

        self._apply_preset("coarse")

    # ---- 内部 ---- #
    def _on_granularity(self) -> None:
        key = self.granularity.currentData()
        if key != "custom":
            self._apply_preset(key)

    def _apply_preset(self, key: str) -> None:
        base, visible, _ratio = GRANULARITY_PRESETS[key]
        self.base_count.setValue(base)
        self.visible_count.setValue(visible)

    def _emit_generate(self) -> None:
        self.generate_requested.emit(self.options())

    # ---- 公開 API ---- #
    def options(self) -> dict:
        key = self.granularity.currentData()
        ratio = GRANULARITY_PRESETS.get(key, (0, 0, 5))[2] if key != "custom" else 5
        return {
            "preset": key,
            "backend": self.backend.currentData(),
            "working_max_side": int(self.working_side.currentData()),
            "default_visible_count": int(self.visible_count.value()),
            "base_region_count": int(self.base_count.value()),
            "min_region_area_ratio": int(ratio),
        }

    def apply_settings(self, settings) -> None:
        """AppSettings から初期値を反映する。"""
        preset = settings.get("partition/default_preset", "coarse")
        for i in range(self.granularity.count()):
            if self.granularity.itemData(i) == preset:
                self.granularity.setCurrentIndex(i)
                break

    def set_busy(self, busy: bool) -> None:
        self.generate_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)

    def set_progress(self, value: int) -> None:
        self.progress.setValue(max(0, min(100, int(value))))

    def set_status(self, text: str) -> None:
        self.status.setText(text)
