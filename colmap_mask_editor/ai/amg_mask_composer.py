"""
V0.8: レビュー判断 (keep/remove) から最終マスク PNG 用の 2 値配列を合成する。

最終マスク生成時だけ RLE を復号する。必要な segment_id のだけ復号して
KEEP / REMOVE 和集合を作り、3 方式で最終マスクを生成する。

REMOVE は KEEP より優先する (重複領域は REMOVE が勝つ)。
未確認 (unreviewed) は自動反映しない。

numpy のみに依存する。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ai import amg_rle
from ai.amg_review_state import SegmentDecision

# 最終マスク生成方式
MODE_EXCLUDE_REMOVE = "exclude_remove"      # 不要領域を除外 (既定)
MODE_KEEP_ONLY = "keep_only"                # 必要領域のみ
MODE_ADD_REMOVE = "add_remove"              # 現在マスクへ追加・除外

FINAL_MASK_MODES = frozenset({MODE_EXCLUDE_REMOVE, MODE_KEEP_ONLY, MODE_ADD_REMOVE})

__all__ = [
    "MODE_EXCLUDE_REMOVE",
    "MODE_KEEP_ONLY",
    "MODE_ADD_REMOVE",
    "FINAL_MASK_MODES",
    "build_decision_unions",
    "compose_final_mask",
]


def _decode_union(npz_data, decisions: dict[str, str], target: str, height: int, width: int) -> np.ndarray:
    """target ('keep'|'remove') の segment の和集合を bool (h,w) で返す。"""
    segment_ids = np.asarray(npz_data["segment_ids"]).tolist()
    union = np.zeros((height, width), dtype=bool)
    for index, sid in enumerate(segment_ids):
        if decisions.get(str(int(sid))) != target:
            continue
        counts = amg_rle.unpack_counts(npz_data, index)
        mask = amg_rle.decode_rle(counts, height, width) > 0
        union |= mask
    return union


def build_decision_unions(npz_data, decisions: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """KEEP 和集合と REMOVE 和集合 (bool, h,w) を返す。"""
    image_shape = np.asarray(npz_data["image_shape"])
    h, w = int(image_shape[0]), int(image_shape[1])
    keep_union = _decode_union(npz_data, decisions, SegmentDecision.KEEP.value, h, w)
    remove_union = _decode_union(npz_data, decisions, SegmentDecision.REMOVE.value, h, w)
    return keep_union, remove_union


def compose_final_mask(
    npz_data,
    decisions: dict[str, str],
    mode: str,
    existing_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    最終マスクを (h, w) uint8 (0/255) で合成する。REMOVE を最後に適用。

    mode:
      exclude_remove : base = 既存 or 全面255、REMOVE 和集合を 0 にする
      keep_only      : base = 全面0、KEEP 和集合を 255 にする
      add_remove     : base = 既存 or 全面0、KEEP を 255、最後に REMOVE を 0
    """
    if mode not in FINAL_MASK_MODES:
        raise ValueError(f"不明な最終マスク方式: {mode!r}")

    image_shape = np.asarray(npz_data["image_shape"])
    h, w = int(image_shape[0]), int(image_shape[1])
    keep_union, remove_union = build_decision_unions(npz_data, decisions)

    base_existing = None
    if existing_mask is not None:
        base_existing = (np.asarray(existing_mask) > 0)
        if base_existing.shape != (h, w):
            raise ValueError(
                f"既存マスク shape {base_existing.shape} が画像 {(h, w)} と一致しません"
            )

    if mode == MODE_EXCLUDE_REMOVE:
        if base_existing is not None:
            result = base_existing.copy()
        else:
            result = np.ones((h, w), dtype=bool)
        result[remove_union] = False
    elif mode == MODE_KEEP_ONLY:
        result = np.zeros((h, w), dtype=bool)
        result[keep_union] = True
    else:  # MODE_ADD_REMOVE
        if base_existing is not None:
            result = base_existing.copy()
        else:
            result = np.zeros((h, w), dtype=bool)
        result[keep_union] = True
        result[remove_union] = False  # REMOVE を最後に適用 -> 優先

    return result.astype(np.uint8) * 255
