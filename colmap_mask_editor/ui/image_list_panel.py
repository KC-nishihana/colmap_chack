"""
左ペイン: 画像一覧と状態表示
"""

from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.project_loader import ImageEntry

# 状態ラベルの色
_STATUS_COLORS = {
    "no_mask":     QColor(180, 180, 180),   # グレー: マスクなし
    "has_mask":    QColor(100, 200, 100),   # 緑: マスクあり
    "modified":    QColor(255, 180,  60),   # オレンジ: 未保存
    "size_error":  QColor(220,  80,  80),   # 赤: サイズ不一致
}


def _status_text(entry: ImageEntry) -> str:
    if entry.mask_size_mismatch:
        return "⚠ サイズ不一致"
    if entry.is_modified:
        return "● 未保存"
    if entry.has_mask:
        return "✓ マスクあり"
    return "  マスクなし"


def _status_color(entry: ImageEntry) -> QColor:
    if entry.mask_size_mismatch:
        return _STATUS_COLORS["size_error"]
    if entry.is_modified:
        return _STATUS_COLORS["modified"]
    if entry.has_mask:
        return _STATUS_COLORS["has_mask"]
    return _STATUS_COLORS["no_mask"]


class ImageListPanel(QWidget):
    """画像一覧パネル。行をクリックすると image_selected シグナルを送出する。"""

    image_selected = Signal(int)  # 選択された画像のインデックス

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._entries: list[ImageEntry] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._title = QLabel("画像一覧")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._title)

        self._count_label = QLabel("0 枚")
        self._count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._count_label)

        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self._list)

    def set_entries(self, entries: list[ImageEntry]) -> None:
        """画像エントリ一覧をセットしてリストを再構築"""
        self._entries = entries
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        self._list.blockSignals(True)
        current = self._list.currentRow()
        self._list.clear()

        for entry in self._entries:
            status = _status_text(entry)
            text = f"{entry.rel_path}\n{status}"
            item = QListWidgetItem(text)
            item.setForeground(_status_color(entry))
            self._list.addItem(item)

        self._count_label.setText(f"{len(self._entries)} 枚")
        self._list.blockSignals(False)

        # 選択を復元
        if 0 <= current < len(self._entries):
            self._list.setCurrentRow(current)

    def update_entry(self, index: int) -> None:
        """特定エントリの表示を更新(未保存フラグ変化時など)"""
        if 0 <= index < len(self._entries):
            entry = self._entries[index]
            status = _status_text(entry)
            text = f"{entry.rel_path}\n{status}"
            item = self._list.item(index)
            if item:
                item.setText(text)
                item.setForeground(_status_color(entry))

    def select_row(self, index: int) -> None:
        """プログラムから行を選択(シグナルは発火させる)"""
        self._list.setCurrentRow(index)

    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            self.image_selected.emit(row)
