"""
V0.11: 選択ツール (SelectionTool) と適用操作 (ApplyOperation) の分離。

従来の EditMode は「矩形追加 / 矩形削除 / ポリゴン追加 / ...」のように選択ツールと
適用方法 (追加 / 削除) を 1 つの値へ混ぜていた。V0.11 ではこれを 2 軸へ分離する:

  SelectionTool   何で選択するか (AIクリック / 画像全体自動分割 / ブラシ / ポリゴン / 矩形 / パン)
  ApplyOperation  選択をどう適用するか (有効にする ADD / 除外する REMOVE / 置き換える REPLACE)

EditMode は後方互換アダプタとして残す。新 UI はツールと適用方法を別々に選ぶ。
このモジュールは純粋 (PySide6 / numpy 非依存)。EditMode への変換だけ遅延 import する。
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "SelectionTool",
    "ApplyOperation",
    "DEFAULT_SELECTION_TOOL",
    "DEFAULT_APPLY_OPERATION",
    "to_edit_mode",
    "from_edit_mode",
]


class SelectionTool(Enum):
    AI_CLICK = "ai_click"          # AIクリックで領域選択 (SAM 正/負ポイント・矩形プロンプト)
    AI_AUTOMATIC = "ai_automatic"  # 画像全体を自動分割 (AMG 候補)
    BRUSH = "brush"
    POLYGON = "polygon"
    RECTANGLE = "rectangle"
    PAN = "pan"

    @classmethod
    def from_str(cls, value: str) -> "SelectionTool":
        try:
            return cls(value)
        except ValueError as e:
            raise ValueError(f"不明な選択ツール: {value!r}") from e


class ApplyOperation(Enum):
    ADD = "add"          # 有効にする (候補領域を 255)
    REMOVE = "remove"    # 除外する (候補領域を 0)
    REPLACE = "replace"  # 置き換える (候補領域だけ 255・それ以外 0)

    @classmethod
    def from_str(cls, value: str) -> "ApplyOperation":
        try:
            return cls(value)
        except ValueError as e:
            raise ValueError(f"不明な適用操作: {value!r}") from e


# V0.11 既定値 (設定 ui/default_selection_tool, ui/default_apply_operation と整合)
DEFAULT_SELECTION_TOOL = SelectionTool.AI_CLICK
DEFAULT_APPLY_OPERATION = ApplyOperation.REMOVE


# ------------------------------------------------------------------ #
# EditMode 後方互換アダプタ
# ------------------------------------------------------------------ #
#
# 旧 EditMode へ 1:1 で対応するのは矩形 / ポリゴン (追加・削除)・ブラシ・パン・
# AIプロンプト (= AIクリック) のみ。画像全体自動分割 (AI_AUTOMATIC) と REPLACE は
# 旧 EditMode に対応する値が無いため変換不可 (ValueError)。GrabCut 系 EditMode は
# 新 SelectionTool に存在しないため from_edit_mode では None を返す。


def to_edit_mode(tool: SelectionTool, operation: ApplyOperation):
    """(SelectionTool, ApplyOperation) を旧 EditMode へ変換する。

    対応値が無い組み合わせ (AI_AUTOMATIC, REPLACE 等) は ValueError。
    EditMode は ui 層にあるため遅延 import する (core 層が ui に依存しないように)。
    """
    from ui.image_canvas import EditMode

    if tool is SelectionTool.PAN:
        return EditMode.PAN
    if tool is SelectionTool.BRUSH:
        return EditMode.BRUSH   # ブラシの ADD/REMOVE は描画時に決まる (Shift 反転等)
    if tool is SelectionTool.AI_CLICK:
        return EditMode.AI_PROMPT
    if tool is SelectionTool.RECTANGLE:
        if operation is ApplyOperation.ADD:
            return EditMode.RECT_ADD
        if operation is ApplyOperation.REMOVE:
            return EditMode.RECT_DEL
    if tool is SelectionTool.POLYGON:
        if operation is ApplyOperation.ADD:
            return EditMode.POLY_ADD
        if operation is ApplyOperation.REMOVE:
            return EditMode.POLY_DEL
    raise ValueError(f"EditMode へ変換できない組み合わせ: {tool}, {operation}")


def from_edit_mode(mode) -> tuple["SelectionTool", "ApplyOperation"] | None:
    """旧 EditMode を (SelectionTool, ApplyOperation) へ変換する。

    GrabCut 系など新ツールに対応が無い EditMode は None を返す。
    """
    from ui.image_canvas import EditMode

    table = {
        EditMode.PAN: (SelectionTool.PAN, ApplyOperation.REMOVE),
        EditMode.BRUSH: (SelectionTool.BRUSH, ApplyOperation.ADD),
        EditMode.AI_PROMPT: (SelectionTool.AI_CLICK, ApplyOperation.REMOVE),
        EditMode.RECT_ADD: (SelectionTool.RECTANGLE, ApplyOperation.ADD),
        EditMode.RECT_DEL: (SelectionTool.RECTANGLE, ApplyOperation.REMOVE),
        EditMode.POLY_ADD: (SelectionTool.POLYGON, ApplyOperation.ADD),
        EditMode.POLY_DEL: (SelectionTool.POLYGON, ApplyOperation.REMOVE),
    }
    return table.get(mode)
