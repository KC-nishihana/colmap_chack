"""
マスク編集操作(ブラシ、Undo/Redo、リサイズ)を担うモジュール
"""

from copy import deepcopy

import cv2
import numpy as np


class MaskEditor:
    """
    マスク画像に対するブラシ編集と Undo/Redo を管理するクラス。
    """

    MAX_HISTORY = 50  # 最大 Undo 履歴数

    def __init__(self, mask: np.ndarray) -> None:
        self._current: np.ndarray = mask.copy()
        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []

    @property
    def mask(self) -> np.ndarray:
        return self._current

    def set_mask(self, mask: np.ndarray) -> None:
        """外部からマスクを差し替える(画像切替時など)"""
        self._current = mask.copy()
        self._undo_stack.clear()
        self._redo_stack.clear()

    def _push_undo(self) -> None:
        """現在の状態を Undo スタックに積む"""
        self._undo_stack.append(self._current.copy())
        if len(self._undo_stack) > self.MAX_HISTORY:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def paint(self, cx: int, cy: int, radius: int, add: bool) -> None:
        """
        ブラシ描画。add=True でマスク追加(白)、False で削除(黒)。
        Undo スタックには push しない(ドラッグ開始時にpush済みを想定)。
        """
        color = 255 if add else 0
        cv2.circle(self._current, (cx, cy), radius, color, -1)

    def begin_stroke(self) -> None:
        """ドラッグ開始時に現在状態を Undo スタックに積む"""
        self._push_undo()

    def replace(self, new_mask: np.ndarray) -> None:
        """
        マスク全体を 1 操作で差し替える (Undo 可能)。

        現在状態を Undo スタックへ積んでから差し替える。AI候補の適用 (REMOVE 和集合
        を 0 にする等) のように、1 コマンドでマスク全体が変わる編集に使う。
        set_mask (履歴クリア) とは異なり履歴を保持する。
        """
        self._push_undo()
        self._current = new_mask.copy()

    def undo(self) -> bool:
        """Undo: 1つ前の状態に戻す。成功したら True を返す"""
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._current.copy())
        self._current = self._undo_stack.pop()
        return True

    def redo(self) -> bool:
        """Redo: 1つ先の状態に進む。成功したら True を返す"""
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._current.copy())
        self._current = self._redo_stack.pop()
        return True

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def resize_to(self, width: int, height: int) -> None:
        """
        マスクを指定サイズにリサイズ。最近傍補間を使用して2値を維持。
        リサイズ後も Undo 可能にするため履歴に積む。
        """
        self._push_undo()
        resized = cv2.resize(self._current, (width, height), interpolation=cv2.INTER_NEAREST)
        _, self._current = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
