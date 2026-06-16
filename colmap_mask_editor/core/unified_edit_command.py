"""
V0.11: 統一 Undo/Redo。AI候補の適用も手動編集と同じ履歴へ入れる。

通常マスクの before/after は既存 MaskEditor の履歴をそのまま使う。本モジュールは
AI 判断メタデータ (候補の適用状態 = decisions) だけを MaskEditor の履歴と同じ歩調で
同期する。AI候補REMOVE / ADD / AIクリック / ブラシ / ポリゴン / 矩形 をすべて 1 本の
履歴で Undo/Redo できる。

使い方 (controller 側):
  1. マスクを変更する前に before = history.current_decisions を控える
  2. MaskEditor を 1 操作だけ変更する
        - 対話編集 (ブラシ等):   editor.begin_stroke() を 1 回呼んでから paint
        - 一括差し替え (AI適用):  editor.replace(new_mask)
     どちらも MaskEditor の Undo スタックへ「1 エントリ」積む
  3. history.push(UnifiedEditCommand(... before, after ...)) を 1 回呼ぶ
  → MaskEditor の履歴エントリ数と本履歴のコマンド数が常に 1:1 で揃う

undo()/redo() は MaskEditor の undo/redo を呼び、AI 判断を before/after へ同期する。

stdlib のみ (numpy/Qt/torch 非依存)。MaskEditor は duck-typing で受ける。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = ["UnifiedEditCommand", "UnifiedEditHistory"]

# source の語彙 (任意の文字列でよいが代表値を定義)
SOURCE_AI_AUTOMATIC = "ai_automatic"
SOURCE_AI_CLICK = "ai_click"
SOURCE_BRUSH = "brush"
SOURCE_POLYGON = "polygon"
SOURCE_RECTANGLE = "rectangle"


@dataclass
class UnifiedEditCommand:
    """1 回の編集を表す。マスク本体は MaskEditor 履歴側、ここは AI 判断の差分。"""
    source: str
    operation: str
    affected_segment_ids: list[int] = field(default_factory=list)
    before_decisions: dict[str, str] = field(default_factory=dict)
    after_decisions: dict[str, str] = field(default_factory=dict)


class UnifiedEditHistory:
    """MaskEditor の Undo/Redo に AI 判断メタデータを同期させる薄いラッパー。"""

    def __init__(self, editor, initial_decisions: Optional[dict[str, str]] = None) -> None:
        self._editor = editor
        self._undo: list[UnifiedEditCommand] = []
        self._redo: list[UnifiedEditCommand] = []
        self.current_decisions: dict[str, str] = dict(initial_decisions or {})
        # MaskEditor の履歴上限に合わせて古いコマンドを落とし、件数を揃える。
        self._max = int(getattr(editor, "MAX_HISTORY", 50))

    # ------------------------------------------------------------------ #
    # コマンド登録
    # ------------------------------------------------------------------ #

    def push(self, command: UnifiedEditCommand) -> None:
        """
        MaskEditor を 1 操作変更した「後」に呼ぶ。判断を after へ進め、redo を捨てる。

        呼び出し側が MaskEditor へ Undo エントリを 1 つだけ積んでいることが前提
        (begin_stroke 1 回、または replace 1 回)。
        """
        self._undo.append(command)
        if len(self._undo) > self._max:
            self._undo.pop(0)        # MaskEditor の上限切り捨てと歩調を合わせる
        self._redo.clear()
        self.current_decisions = dict(command.after_decisions)

    def record(
        self,
        *,
        source: str,
        operation: str,
        after_decisions: Optional[dict[str, str]] = None,
        affected_segment_ids: Optional[list[int]] = None,
    ) -> UnifiedEditCommand:
        """before=現在の判断, after=指定 (省略時は変化なし) でコマンドを作って push する。"""
        before = dict(self.current_decisions)
        after = dict(after_decisions) if after_decisions is not None else dict(before)
        cmd = UnifiedEditCommand(
            source=source, operation=operation,
            affected_segment_ids=list(affected_segment_ids or []),
            before_decisions=before, after_decisions=after)
        self.push(cmd)
        return cmd

    # ------------------------------------------------------------------ #
    # Undo / Redo
    # ------------------------------------------------------------------ #

    def can_undo(self) -> bool:
        return bool(self._undo) and self._editor.can_undo()

    def can_redo(self) -> bool:
        return bool(self._redo) and self._editor.can_redo()

    def undo(self) -> Optional[UnifiedEditCommand]:
        """マスクを 1 つ戻し、AI 判断を before へ同期する。"""
        if not self._undo or not self._editor.undo():
            return None
        cmd = self._undo.pop()
        self._redo.append(cmd)
        self.current_decisions = dict(cmd.before_decisions)
        return cmd

    def redo(self) -> Optional[UnifiedEditCommand]:
        """マスクを 1 つ進め、AI 判断を after へ同期する。"""
        if not self._redo or not self._editor.redo():
            return None
        cmd = self._redo.pop()
        self._undo.append(cmd)
        self.current_decisions = dict(cmd.after_decisions)
        return cmd

    # ------------------------------------------------------------------ #
    # その他
    # ------------------------------------------------------------------ #

    def reset(self, decisions: Optional[dict[str, str]] = None) -> None:
        """画像切替時など履歴を破棄して判断を初期化する。"""
        self._undo.clear()
        self._redo.clear()
        self.current_decisions = dict(decisions or {})

    @property
    def undo_depth(self) -> int:
        return len(self._undo)
