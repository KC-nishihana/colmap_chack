"""
AIセグメンテーションのプロンプト (正クリック・負クリック・矩形) と
プロンプト専用の Undo/Redo を管理する。

座標はすべて「元画像座標」で保持する (キャンバスの表示スケールに依存しない)。
通常マスクや GrabCut ヒントの Undo/Redo とは完全に独立した履歴を持つ。

torch / sam2 / PySide6 に依存しない純粋ロジック。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class AiPromptType(Enum):
    POSITIVE_POINT = auto()
    NEGATIVE_POINT = auto()
    BOX = auto()


# SAM のラベル規約: 1 = 前景(対象), 0 = 背景
LABEL_POSITIVE = 1
LABEL_NEGATIVE = 0


@dataclass(frozen=True)
class AiPointPrompt:
    x: float
    y: float
    label: int  # 1=正クリック, 0=負クリック

    @property
    def is_positive(self) -> bool:
        return self.label == LABEL_POSITIVE


@dataclass(frozen=True)
class AiBoxPrompt:
    x1: float
    y1: float
    x2: float
    y2: float

    def normalized(self) -> "AiBoxPrompt":
        """左上 <= 右下 になるよう並べ替えた矩形を返す。"""
        return AiBoxPrompt(
            x1=min(self.x1, self.x2),
            y1=min(self.y1, self.y2),
            x2=max(self.x1, self.x2),
            y2=max(self.y1, self.y2),
        )

    @property
    def width(self) -> float:
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return abs(self.y2 - self.y1)


@dataclass
class AiPromptState:
    """1枚の画像に対する現在のプロンプト集合 (スナップショット可能)。"""
    points: list[AiPointPrompt] = field(default_factory=list)
    box: Optional[AiBoxPrompt] = None

    def is_empty(self) -> bool:
        return not self.points and self.box is None

    def clone(self) -> "AiPromptState":
        return AiPromptState(points=list(self.points), box=self.box)


class AiPromptSession:
    """
    プロンプトの編集と Undo/Redo を提供する。

    1操作 = 1スナップショット (points と box のセット)。点の追加・削除・矩形設定の
    たびに履歴へ積む。通常マスク/GrabCut とは別系統。
    """

    MAX_HISTORY = 100

    def __init__(self) -> None:
        self._state = AiPromptState()
        self._undo: list[AiPromptState] = []
        self._redo: list[AiPromptState] = []

    # ----- 参照 -----

    @property
    def state(self) -> AiPromptState:
        return self._state

    @property
    def points(self) -> list[AiPointPrompt]:
        return list(self._state.points)

    @property
    def box(self) -> Optional[AiBoxPrompt]:
        return self._state.box

    def is_empty(self) -> bool:
        return self._state.is_empty()

    def has_any(self) -> bool:
        return not self._state.is_empty()

    # ----- 履歴管理 -----

    def _push_history(self) -> None:
        self._undo.append(self._state.clone())
        if len(self._undo) > self.MAX_HISTORY:
            self._undo.pop(0)
        self._redo.clear()

    # ----- 編集操作 -----

    def add_point(self, x: float, y: float, positive: bool) -> None:
        self._push_history()
        label = LABEL_POSITIVE if positive else LABEL_NEGATIVE
        self._state = self._state.clone()
        self._state.points.append(AiPointPrompt(x=float(x), y=float(y), label=label))

    def set_box(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self._push_history()
        self._state = self._state.clone()
        self._state.box = AiBoxPrompt(x1, y1, x2, y2).normalized()

    def clear_box(self) -> None:
        if self._state.box is None:
            return
        self._push_history()
        self._state = self._state.clone()
        self._state.box = None

    def remove_last_point(self) -> bool:
        if not self._state.points:
            return False
        self._push_history()
        self._state = self._state.clone()
        self._state.points.pop()
        return True

    def clear(self) -> None:
        """全プロンプトを消去する (履歴へ積むのでUndoで戻せる)。"""
        if self._state.is_empty():
            return
        self._push_history()
        self._state = AiPromptState()

    def reset(self) -> None:
        """履歴ごと完全リセット (画像切替・適用後など)。Undoでは戻せない。"""
        self._state = AiPromptState()
        self._undo.clear()
        self._redo.clear()

    # ----- Undo / Redo -----

    def can_undo(self) -> bool:
        return len(self._undo) > 0

    def can_redo(self) -> bool:
        return len(self._redo) > 0

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._state.clone())
        self._state = self._undo.pop()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._state.clone())
        self._state = self._redo.pop()
        return True

    # ----- プロトコル変換 -----

    def to_predict_fields(self) -> dict:
        """
        predict コマンド用の points / box フィールドを生成する。
        矩形は [x1, y1, x2, y2]、点は {"x","y","label"}。
        """
        points = [
            {"x": p.x, "y": p.y, "label": p.label}
            for p in self._state.points
        ]
        fields: dict = {"points": points}
        if self._state.box is not None:
            b = self._state.box.normalized()
            fields["box"] = [b.x1, b.y1, b.x2, b.y2]
        return fields
