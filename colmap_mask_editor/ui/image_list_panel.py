"""
左ペイン: 画像一覧・ステータス別フィルタ・状態表示
"""

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.project_loader import ImageEntry

# ステータスの日本語ラベル
_STATUS_LABELS: dict[str, str] = {
    "ok":                  "[OK]",
    "no_mask":             "[マスクなし]",
    "size_mismatch":       "[サイズ不一致]",
    "empty_mask":          "[空マスク]",
    "full_mask":           "[全面マスク]",
    "intermediate_values": "[中間値あり]",
    "unreadable_image":    "[画像エラー]",
    "unreadable_mask":     "[マスクエラー]",
    "not_saved":           "[未保存]",
    "needs_check":         "[要確認]",
}

# ステータス別の文字色
_STATUS_COLORS: dict[str, QColor] = {
    "ok":                  QColor(100, 210, 100),
    "no_mask":             QColor(160, 160, 160),
    "size_mismatch":       QColor(220,  80,  80),
    "empty_mask":          QColor(210, 140,  40),
    "full_mask":           QColor(200, 200,  40),
    "intermediate_values": QColor(100, 170, 230),
    "unreadable_image":    QColor(255,  60,  60),
    "unreadable_mask":     QColor(255,  60,  60),
    "not_saved":           QColor(255, 180,  60),
    "needs_check":         QColor(140, 210, 210),
}

# フィルタ選択肢 (表示名, 内部キー)
_FILTER_OPTIONS: list[tuple[str, str]] = [
    ("すべて",        "all"),
    ("正常",          "ok"),
    ("要確認",        "warn"),
    ("未保存",        "not_saved"),
    ("マスクなし",    "no_mask"),
    ("サイズ不一致",  "size_mismatch"),
    ("空マスク",      "empty_mask"),
    ("全面マスク",    "full_mask"),
    ("中間値あり",    "intermediate_values"),
    ("読み込みエラー", "error"),
]


def get_entry_status(entry: ImageEntry) -> str:
    """エントリの現在ステータスを返す（check_result優先、なければフォールバック）"""
    cr = entry.check_result
    if cr is not None:
        # 未保存フラグが立っていてok判定なら not_saved に格上げ
        if entry.is_modified and cr.status == "ok":
            return "not_saved"
        return cr.status
    # check_result なし: 既存フィールドから推定
    if entry.mask_size_mismatch:
        return "size_mismatch"
    if entry.is_modified:
        return "not_saved"
    if entry.has_mask:
        return "ok"
    return "no_mask"


def _matches_filter(entry: ImageEntry, filter_val: str) -> bool:
    if filter_val == "all":
        return True
    status = get_entry_status(entry)
    if filter_val == "ok":
        return status == "ok"
    if filter_val == "warn":
        return status in {"needs_check", "intermediate_values", "full_mask"}
    if filter_val == "not_saved":
        return status == "not_saved" or entry.is_modified
    if filter_val == "no_mask":
        return status == "no_mask"
    if filter_val == "size_mismatch":
        return status == "size_mismatch"
    if filter_val == "empty_mask":
        return status == "empty_mask"
    if filter_val == "full_mask":
        return status == "full_mask"
    if filter_val == "intermediate_values":
        return status == "intermediate_values"
    if filter_val == "error":
        return status in {"unreadable_image", "unreadable_mask"}
    return True


def _make_item(entry: ImageEntry) -> QListWidgetItem:
    status = get_entry_status(entry)
    label = _STATUS_LABELS.get(status, f"[{status}]")
    text = f"{label} {entry.rel_path}"
    item = QListWidgetItem(text)
    item.setForeground(_STATUS_COLORS.get(status, QColor(200, 200, 200)))
    return item


def _update_item(item: QListWidgetItem, entry: ImageEntry) -> None:
    status = get_entry_status(entry)
    label = _STATUS_LABELS.get(status, f"[{status}]")
    item.setText(f"{label} {entry.rel_path}")
    item.setForeground(_STATUS_COLORS.get(status, QColor(200, 200, 200)))


class ImageListPanel(QWidget):
    """
    画像一覧パネル。
    行クリック時に image_selected(global_index) シグナルを送出する。
    global_index は self._entries 内のインデックス（フィルタに関係なく一定）。
    """

    image_selected = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._entries: list[ImageEntry] = []
        self._filtered_indices: list[int] = []   # list行 → entries グローバルindex
        self._current_filter: str = "all"
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        title = QLabel("画像一覧")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # フィルタ
        self._filter_combo = QComboBox()
        for label, _ in _FILTER_OPTIONS:
            self._filter_combo.addItem(label)
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        layout.addWidget(self._filter_combo)

        self._count_label = QLabel("0 / 0 枚")
        self._count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._count_label)

        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self._list)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def set_entries(self, entries: list[ImageEntry]) -> None:
        self._entries = entries
        self._apply_filter()

    def update_entry(self, global_index: int) -> None:
        """1エントリの表示を更新（保存・編集時）"""
        if global_index not in self._filtered_indices:
            return
        row = self._filtered_indices.index(global_index)
        item = self._list.item(row)
        if item:
            _update_item(item, self._entries[global_index])

    def refresh_all(self) -> None:
        """全エントリの表示を再構築（一括チェック後など）"""
        self._apply_filter()

    def select_row(self, global_index: int) -> None:
        """プログラムからグローバルインデックスを選択状態にする"""
        if global_index in self._filtered_indices:
            row = self._filtered_indices.index(global_index)
            self._list.setCurrentRow(row)

    def current_global_index(self) -> int:
        """現在選択中のグローバルインデックスを返す（未選択は -1）"""
        row = self._list.currentRow()
        if 0 <= row < len(self._filtered_indices):
            return self._filtered_indices[row]
        return -1

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #

    def _on_filter_changed(self, idx: int) -> None:
        _, val = _FILTER_OPTIONS[idx]
        self._current_filter = val
        self._apply_filter()

    def _apply_filter(self) -> None:
        # 現在の選択グローバルインデックスを保存
        saved_global = self.current_global_index()

        self._filtered_indices = [
            i for i, e in enumerate(self._entries)
            if _matches_filter(e, self._current_filter)
        ]
        self._rebuild_list()

        # 選択を復元（フィルタ後に見えていれば）
        if saved_global >= 0 and saved_global in self._filtered_indices:
            row = self._filtered_indices.index(saved_global)
            self._list.blockSignals(True)
            self._list.setCurrentRow(row)
            self._list.blockSignals(False)

    def _rebuild_list(self) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        for i in self._filtered_indices:
            self._list.addItem(_make_item(self._entries[i]))
        total = len(self._entries)
        shown = len(self._filtered_indices)
        self._count_label.setText(f"{shown} / {total} 枚")
        self._list.blockSignals(False)

    def _on_row_changed(self, row: int) -> None:
        if 0 <= row < len(self._filtered_indices):
            self.image_selected.emit(self._filtered_indices[row])
