"""
V0.7 GUI: 伝播結果レビュー画面。

採用/除外をフレーム単位で行い、採用済みへ適用モード (追加/除外/置換) で一括適用する。
サムネイルは遅延生成 (選択時に読む)。全画像をフル解像度で保持しない。
torch / sam2 は import しない。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ai.ai_mask_ops import APPLY_ADD, APPLY_EXCLUDE, APPLY_REPLACE
from ai.propagation_session import FrameState, PropagationSession
from ai.propagation_staging import read_mask_png

_STATE_ICON = {
    FrameState.DONE: "✓",
    FrameState.WARNING: "△",
    FrameState.FAILED: "!",
    FrameState.PENDING: "…",
    FrameState.SKIPPED: "×",
}


class PropagationReviewDialog(QDialog):
    """伝播結果のレビューと採否・適用モード選択。

    accepted() で採用フレーム、apply_mode() で適用モードを返す。
    実際の一括適用は呼び出し側 (MainWindow) が PropagationApplyWorker で行う。
    """

    def __init__(self, session: PropagationSession, parent=None) -> None:
        super().__init__(parent)
        self._session = session
        self.setWindowTitle("伝播結果レビュー")
        self.resize(900, 600)
        self._apply_requested = False
        self._build_ui()
        self._populate()

    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        # 左: フレーム一覧
        left = QVBoxLayout()
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_row_changed)
        left.addWidget(QLabel("フレーム"))
        left.addWidget(self._list, 1)

        btn_row = QHBoxLayout()
        self._btn_accept = QPushButton("採用")
        self._btn_exclude = QPushButton("除外")
        self._btn_accept.clicked.connect(lambda: self._set_current_accept(True))
        self._btn_exclude.clicked.connect(lambda: self._set_current_accept(False))
        btn_row.addWidget(self._btn_accept)
        btn_row.addWidget(self._btn_exclude)
        left.addLayout(btn_row)

        bulk_row = QHBoxLayout()
        b_all = QPushButton("全採用")
        b_none = QPushButton("全除外")
        b_warn = QPushButton("警告なしを採用")
        b_all.clicked.connect(lambda: self._bulk_accept("all"))
        b_none.clicked.connect(lambda: self._bulk_accept("none"))
        b_warn.clicked.connect(lambda: self._bulk_accept("no_warn"))
        bulk_row.addWidget(b_all)
        bulk_row.addWidget(b_none)
        bulk_row.addWidget(b_warn)
        left.addLayout(bulk_row)
        root.addLayout(left, 0)

        # 中央: プレビュー
        center = QVBoxLayout()
        self._preview = QLabel("プレビュー")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(420, 360)
        self._preview.setStyleSheet("background:#202020;color:#aaa;")
        center.addWidget(self._preview, 1)
        self._info = QLabel("")
        self._info.setWordWrap(True)
        center.addWidget(self._info)
        root.addLayout(center, 1)

        # 右: 適用
        right = QVBoxLayout()
        right.addWidget(QLabel("適用モード"))
        self._mode = QComboBox()
        self._mode.addItem("追加", APPLY_ADD)
        self._mode.addItem("除外", APPLY_EXCLUDE)
        self._mode.addItem("置換", APPLY_REPLACE)
        right.addWidget(self._mode)
        self._summary = QLabel("")
        right.addWidget(self._summary)
        right.addStretch(1)

        bb = QDialogButtonBox()
        self._apply_btn = bb.addButton("適用", QDialogButtonBox.ButtonRole.AcceptRole)
        bb.addButton("キャンセル", QDialogButtonBox.ButtonRole.RejectRole)
        bb.accepted.connect(self._on_apply)
        bb.rejected.connect(self.reject)
        right.addWidget(bb)
        root.addLayout(right, 0)

    def _populate(self) -> None:
        self._list.clear()
        for f in self._session.frames:
            self._list.addItem(QListWidgetItem(self._row_label(f)))
        self._update_summary()
        if self._session.frames:
            self._list.setCurrentRow(0)

    def _row_label(self, f) -> str:
        icon = _STATE_ICON.get(f.state, "…")
        if f.is_reviewable and not f.accepted:
            icon = "×"
        ref = " [基準]" if f.frame_index == self._session.reference_frame_index else ""
        warn = f" ⚠{len(f.warning_codes)}" if f.warning_codes else ""
        return f"{icon} {f.frame_index:>4}  {f.entry_key}{ref}{warn}"

    # ------------------------------------------------------------------ #

    def _current_frame(self):
        row = self._list.currentRow()
        if 0 <= row < len(self._session.frames):
            return self._session.frames[row]
        return None

    def _on_row_changed(self, row: int) -> None:
        f = self._current_frame()
        if f is None:
            return
        self._load_preview(f)
        warns = ", ".join(f.warning_codes) if f.warning_codes else "なし"
        self._info.setText(
            f"frame {f.frame_index} / {f.entry_key}\n"
            f"状態: {f.state} / 採用: {'はい' if f.accepted else 'いいえ'}\n"
            f"前景率: {f.foreground_ratio:.3f} / 警告: {warns}"
        )

    def _load_preview(self, f) -> None:
        """遅延サムネイル生成 (選択時に1枚だけ読む)。"""
        if not f.result_mask_path:
            self._preview.setText("結果なし")
            return
        try:
            mask = read_mask_png(f.result_mask_path)
        except Exception:
            self._preview.setText("読み込み失敗")
            return
        h, w = mask.shape
        img = QImage(np.ascontiguousarray(mask).data, w, h, w, QImage.Format.Format_Grayscale8)
        pm = QPixmap.fromImage(img).scaled(
            self._preview.width(), self._preview.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        )
        self._preview.setPixmap(pm)

    def _set_current_accept(self, accepted: bool) -> None:
        f = self._current_frame()
        if f is None or not f.is_reviewable:
            return
        f.accepted = accepted
        self._refresh_row(self._list.currentRow())
        self._on_row_changed(self._list.currentRow())
        self._update_summary()

    def _bulk_accept(self, kind: str) -> None:
        for f in self._session.frames:
            if not f.is_reviewable:
                continue
            if kind == "all":
                f.accepted = True
            elif kind == "none":
                f.accepted = False
            elif kind == "no_warn":
                f.accepted = not f.warning_codes
        self._populate_refresh()
        self._update_summary()

    def _populate_refresh(self) -> None:
        for i in range(self._list.count()):
            self._refresh_row(i)

    def _refresh_row(self, row: int) -> None:
        if 0 <= row < len(self._session.frames):
            self._list.item(row).setText(self._row_label(self._session.frames[row]))

    def _update_summary(self) -> None:
        acc = len(self._session.accepted_frames())
        self._summary.setText(f"適用対象: {acc} フレーム (基準・失敗を除く)")

    def _on_apply(self) -> None:
        self._apply_requested = True
        self.accept()

    # ------------------------------------------------------------------ #

    def apply_requested(self) -> bool:
        return self._apply_requested

    def apply_mode(self) -> str:
        return self._mode.currentData()

    def accepted_frames(self):
        return self._session.accepted_frames()
