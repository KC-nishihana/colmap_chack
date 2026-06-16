"""
V0.11: 統合レビュー画面の上部ツールバー。

選択方法 (SelectionTool) と適用方法 (ApplyOperation) を 2 軸で常時表示し、
日常的に使う機能だけを並べる。中央キャンバスはこのツールバーの状態に従う。

    選択方法: [AIクリック] [画像全体を自動分割] [ブラシ] [ポリゴン] [矩形] [パン]
    適用方法: [有効にする] [除外する] [置き換える]

初期値: 選択=AIクリック / 適用=除外する (設定 ui/default_selection_tool,
ui/default_apply_operation で上書き可能)。

このウィジェットは選択状態とシグナルだけを提供し、実際の編集は MainWindow 側で
SelectionTool / ApplyOperation を見て中央キャンバスへ反映する。torch / sam2 非依存。
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QWidget,
)

# 統合ツールバーの見た目 (選択中ボタンをはっきり強調し、十分なパディングを取る)
_TOOLBAR_STYLE = """
QToolButton {
    padding: 5px 12px;
    margin: 0 1px;
    border: 1px solid #555;
    border-radius: 4px;
    background: #3a3a3a;
    color: #ddd;
}
QToolButton:hover { background: #474747; }
QToolButton:checked {
    background: #2d6cdf;
    color: white;
    border: 1px solid #2d6cdf;
    font-weight: bold;
}
"""

from core.selection_tools import (
    DEFAULT_APPLY_OPERATION,
    DEFAULT_SELECTION_TOOL,
    ApplyOperation,
    SelectionTool,
)

__all__ = ["UnifiedToolBar"]


# (SelectionTool, ラベル, ツールチップ)。表示順は仕様どおり。
_TOOL_DEFS: list[tuple[SelectionTool, str, str]] = [
    (SelectionTool.AI_CLICK,     "AIクリック",       "対象をクリックして AI で領域選択 (SAM 正/負ポイント・矩形)"),
    (SelectionTool.AI_AUTOMATIC, "画像全体を自動分割", "現在画像を自動分割し候補から選ぶ (AMG)"),
    (SelectionTool.BRUSH,        "ブラシ",            "ブラシで塗って修正 [B]"),
    (SelectionTool.POLYGON,      "ポリゴン",          "ポリゴンで囲って修正 [P]"),
    (SelectionTool.RECTANGLE,    "矩形",              "矩形で囲って修正 [R]"),
    (SelectionTool.PAN,          "パン",              "画像を移動 (パン操作)"),
]

# (ApplyOperation, ラベル, ツールチップ)
_OP_DEFS: list[tuple[ApplyOperation, str, str]] = [
    (ApplyOperation.ADD,     "有効にする", "選択領域を有効 (255) にする"),
    (ApplyOperation.REMOVE,  "除外する",   "選択領域を除外 (0) にする"),
    (ApplyOperation.REPLACE, "置き換える", "選択領域だけ有効・それ以外を除外する"),
]

# 設定キー (AppSettings)。settings を渡せば初期値とボタン表示を反映する。
_KEY_DEFAULT_TOOL = "ui/default_selection_tool"
_KEY_DEFAULT_OP = "ui/default_apply_operation"
_KEY_SHOW_AI_CLICK = "ui/show_ai_click_button"
_KEY_SHOW_AI_AUTOMATIC = "ui/show_ai_automatic_button"


class UnifiedToolBar(QWidget):
    """選択方法と適用方法を分離して常時表示するツールバー。

    Signals:
        selection_tool_changed(object): SelectionTool が変わったとき
        apply_operation_changed(object): ApplyOperation が変わったとき
    """

    selection_tool_changed = Signal(object)
    apply_operation_changed = Signal(object)

    def __init__(self, settings=None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tool_btns: dict[SelectionTool, QToolButton] = {}
        self._op_btns: dict[ApplyOperation, QToolButton] = {}
        self.setStyleSheet(_TOOLBAR_STYLE)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(3)

        # ----- 選択方法 -----
        layout.addWidget(self._group_label("選択方法"))
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        for i, (tool, label, tip) in enumerate(_TOOL_DEFS):
            btn = self._make_button(label, tip)
            self._tool_group.addButton(btn, i)
            self._tool_btns[tool] = btn
            layout.addWidget(btn)

        layout.addSpacing(10)
        layout.addWidget(self._vline())
        layout.addSpacing(10)

        # ----- 適用方法 -----
        layout.addWidget(self._group_label("適用方法"))
        self._op_group = QButtonGroup(self)
        self._op_group.setExclusive(True)
        for i, (op, label, tip) in enumerate(_OP_DEFS):
            btn = self._make_button(label, tip)
            self._op_group.addButton(btn, i)
            self._op_btns[op] = btn
            layout.addWidget(btn)

        layout.addStretch(1)

        # 初期値とボタン表示の反映 (設定があれば優先)
        init_tool = DEFAULT_SELECTION_TOOL
        init_op = DEFAULT_APPLY_OPERATION
        if settings is not None:
            init_tool = self._read_tool(settings, _KEY_DEFAULT_TOOL, init_tool)
            init_op = self._read_op(settings, _KEY_DEFAULT_OP, init_op)
            self.set_button_visible(
                SelectionTool.AI_CLICK,
                self._read_bool(settings, _KEY_SHOW_AI_CLICK, True))
            self.set_button_visible(
                SelectionTool.AI_AUTOMATIC,
                self._read_bool(settings, _KEY_SHOW_AI_AUTOMATIC, True))

        # 初期選択 (シグナルは出さない)
        self._tool_btns[init_tool].setChecked(True)
        self._op_btns[init_op].setChecked(True)

        self._tool_group.idToggled.connect(self._on_tool_toggled)
        self._op_group.idToggled.connect(self._on_op_toggled)

    # ------------------------------------------------------------------ #
    # 構築ヘルパー
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_button(label: str, tip: str) -> QToolButton:
        btn = QToolButton()
        btn.setText(label)
        btn.setToolTip(tip)
        btn.setCheckable(True)
        btn.setAutoExclusive(False)  # 排他は QButtonGroup が担当
        btn.setMinimumHeight(28)
        return btn

    @staticmethod
    def _group_label(text: str) -> QLabel:
        lbl = QLabel(text + ":")
        lbl.setStyleSheet("color: #9aa0a6; font-weight: bold; padding: 0 2px;")
        return lbl

    @staticmethod
    def _vline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("color: #555;")
        return line

    @staticmethod
    def _read_bool(settings, key: str, default: bool) -> bool:
        try:
            return bool(settings.get(key, default))
        except Exception:  # noqa: BLE001
            return default

    @staticmethod
    def _read_tool(settings, key: str, default: SelectionTool) -> SelectionTool:
        try:
            return SelectionTool.from_str(str(settings.get(key, default.value)))
        except Exception:  # noqa: BLE001
            return default

    @staticmethod
    def _read_op(settings, key: str, default: ApplyOperation) -> ApplyOperation:
        try:
            return ApplyOperation.from_str(str(settings.get(key, default.value)))
        except Exception:  # noqa: BLE001
            return default

    # ------------------------------------------------------------------ #
    # 公開 API
    # ------------------------------------------------------------------ #

    def selection_tool(self) -> SelectionTool:
        bid = self._tool_group.checkedId()
        return _TOOL_DEFS[bid][0] if bid >= 0 else DEFAULT_SELECTION_TOOL

    def apply_operation(self) -> ApplyOperation:
        bid = self._op_group.checkedId()
        return _OP_DEFS[bid][0] if bid >= 0 else DEFAULT_APPLY_OPERATION

    def set_selection_tool(self, tool: SelectionTool, *, emit: bool = True) -> None:
        """選択ツールを変更する。emit=False ならシグナルを抑制する。"""
        btn = self._tool_btns[tool]
        if btn.isChecked():
            return
        if not emit:
            self._tool_group.blockSignals(True)
        btn.setChecked(True)
        if not emit:
            self._tool_group.blockSignals(False)

    def set_apply_operation(self, op: ApplyOperation, *, emit: bool = True) -> None:
        """適用操作を変更する。emit=False ならシグナルを抑制する。"""
        btn = self._op_btns[op]
        if btn.isChecked():
            return
        if not emit:
            self._op_group.blockSignals(True)
        btn.setChecked(True)
        if not emit:
            self._op_group.blockSignals(False)

    def set_button_visible(self, tool: SelectionTool, visible: bool) -> None:
        """選択ツールボタンの表示/非表示を切り替える (AIボタンの設定反映用)。"""
        self._tool_btns[tool].setVisible(bool(visible))

    # ------------------------------------------------------------------ #
    # シグナル
    # ------------------------------------------------------------------ #

    def _on_tool_toggled(self, bid: int, checked: bool) -> None:
        if checked and bid >= 0:
            self.selection_tool_changed.emit(_TOOL_DEFS[bid][0])

    def _on_op_toggled(self, bid: int, checked: bool) -> None:
        if checked and bid >= 0:
            self.apply_operation_changed.emit(_OP_DEFS[bid][0])
