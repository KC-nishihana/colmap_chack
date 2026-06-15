"""
V0.10: REMOVE_ONLY (不要領域だけ選択) レビュー方式の中核ロジック (numpy のみ)。

考え方:
  - 全画素は暗黙的に KEEP (基準マスクで初期化)。ユーザーは不要候補だけ REMOVE する。
  - UNREVIEWED は暗黙 KEEP として扱い、最終マスクへ反映しない。
  - KEEP は従来方式との互換用。REMOVE_ONLY 出力には影響しない。
  - 最終マスク = 基準マスク のうち REMOVE 和集合だけ 0 にしたもの (既存
    amg_mask_composer.compose_final_mask の MODE_EXCLUDE_REMOVE を再利用)。

基準マスク:
  - 現在の通常マスク (存在すれば既定)
  - 画像全体を有効 (全面 255)
  既存マスクのサイズが画像と一致しない場合は中止する (黙って全面 255 にしない)。

torch / sam2 / PySide6 に依存しない。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai import amg_mask_composer
from ai.amg_review_state import SegmentDecision

# レビュー方式
WORKFLOW_REMOVE_ONLY = "remove_only"
WORKFLOW_STANDARD = "standard"
VALID_WORKFLOWS = frozenset({WORKFLOW_REMOVE_ONLY, WORKFLOW_STANDARD})

# 基準マスク
BASE_EXISTING_OR_FULL = "existing_or_full"   # 通常マスク優先、無ければ全面
BASE_FULL = "full"                           # 常に全面 255
VALID_BASE_MODES = frozenset({BASE_EXISTING_OR_FULL, BASE_FULL})

# covered (除外済み領域に包含) 抑制のしきい値
DEFAULT_COVERED_THRESHOLD = 0.98

__all__ = [
    "WORKFLOW_REMOVE_ONLY",
    "WORKFLOW_STANDARD",
    "VALID_WORKFLOWS",
    "BASE_EXISTING_OR_FULL",
    "BASE_FULL",
    "VALID_BASE_MODES",
    "DEFAULT_COVERED_THRESHOLD",
    "BaseMaskSizeMismatch",
    "PixelStats",
    "remove_segment_ids",
    "count_remove",
    "prune_remove_only_decisions",
    "resolve_base_existing",
    "resolve_base_mask",
    "compose_remove_only_final",
    "pixel_stats",
    "covered_ratio",
    "is_covered",
]


class BaseMaskSizeMismatch(ValueError):
    """既存マスクのサイズが解析画像と一致しないときに送出する。"""


# ------------------------------------------------------------------ #
# 判断状態
# ------------------------------------------------------------------ #


def remove_segment_ids(decisions: dict[str, str]) -> list[int]:
    """REMOVE 指定された segment_id を昇順で返す。"""
    out = [int(k) for k, v in (decisions or {}).items()
           if v == SegmentDecision.REMOVE.value]
    out.sort()
    return out


def count_remove(decisions: dict[str, str]) -> int:
    """REMOVE 指定された候補数。"""
    return sum(1 for v in (decisions or {}).values()
               if v == SegmentDecision.REMOVE.value)


def prune_remove_only_decisions(decisions: dict[str, str]) -> dict[str, str]:
    """
    REMOVE_ONLY で保存する decisions を最小化する。

    REMOVE だけを残し、UNREVIEWED / KEEP は保存しない (大量の keep を書かない)。
    """
    return {str(int(k)): SegmentDecision.REMOVE.value
            for k, v in (decisions or {}).items()
            if v == SegmentDecision.REMOVE.value}


# ------------------------------------------------------------------ #
# 基準マスク
# ------------------------------------------------------------------ #


def _binarize_existing(existing_mask, h: int, w: int) -> np.ndarray:
    em = np.asarray(existing_mask)
    if em.ndim >= 3:
        em = em[:, :, 0]
    if em.shape != (h, w):
        raise BaseMaskSizeMismatch(
            f"既存マスク shape {em.shape} が画像 {(h, w)} と一致しません")
    return (em > 127).astype(np.uint8) * 255


def resolve_base_existing(h: int, w: int, existing_mask, base_mode: str):
    """
    compose_final_mask へ渡す existing_mask (0/255) または None を返す。

    None は「全面 255 を基準にする」を意味する。サイズ不一致なら例外。
    """
    if base_mode not in VALID_BASE_MODES:
        raise ValueError(f"不明な基準マスク方式: {base_mode!r}")
    if base_mode == BASE_FULL:
        return None
    # BASE_EXISTING_OR_FULL
    if existing_mask is None:
        return None
    return _binarize_existing(existing_mask, h, w)


def resolve_base_mask(h: int, w: int, existing_mask, base_mode: str) -> np.ndarray:
    """基準マスクを bool (h,w) で返す (画素率計算用)。全面なら全 True。"""
    resolved = resolve_base_existing(h, w, existing_mask, base_mode)
    if resolved is None:
        return np.ones((h, w), dtype=bool)
    return resolved > 0


# ------------------------------------------------------------------ #
# 最終マスク合成 (既存経路を再利用)
# ------------------------------------------------------------------ #


def compose_remove_only_final(
    npz_data,
    decisions: dict[str, str],
    *,
    existing_mask=None,
    base_mode: str = BASE_EXISTING_OR_FULL,
) -> np.ndarray:
    """
    REMOVE_ONLY 最終マスクを (h,w) uint8(0/255) で生成する。

    既存の compose_final_mask(MODE_EXCLUDE_REMOVE) を再利用する。専用の重複処理は
    作らない。REMOVE 和集合だけ 0 になり、未確認/KEEP は反映されない。
    """
    image_shape = np.asarray(npz_data["image_shape"])
    h, w = int(image_shape[0]), int(image_shape[1])
    base_existing = resolve_base_existing(h, w, existing_mask, base_mode)
    return amg_mask_composer.compose_final_mask(
        npz_data, decisions, amg_mask_composer.MODE_EXCLUDE_REMOVE,
        existing_mask=base_existing,
    )


# ------------------------------------------------------------------ #
# 画素率 (最終マスク生成と同じ優先順位で計算)
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class PixelStats:
    total_px: int
    effective_px: int        # 最終マスクで 255 になる画素 (= 有効)
    excluded_px: int         # 最終マスクで 0 になる画素 (= 除外)
    effective_ratio: float   # 0..1
    excluded_ratio: float    # 0..1


def pixel_stats(base_mask: np.ndarray, remove_union: np.ndarray) -> PixelStats:
    """
    基準マスク (bool) と REMOVE 和集合 (bool) から最終マスクの画素率を計算する。

    最終マスク = base & ~remove。除外画素は画像全体に対する 0 画素数。
    """
    base = np.asarray(base_mask, dtype=bool)
    remove = np.asarray(remove_union, dtype=bool)
    total = int(base.size)
    final = base & ~remove
    effective = int(final.sum())
    excluded = total - effective
    eff_ratio = (effective / total) if total else 0.0
    exc_ratio = (excluded / total) if total else 0.0
    return PixelStats(
        total_px=total, effective_px=effective, excluded_px=excluded,
        effective_ratio=eff_ratio, excluded_ratio=exc_ratio,
    )


# ------------------------------------------------------------------ #
# covered (除外済み領域に包含) 抑制
# ------------------------------------------------------------------ #


def covered_ratio(seg_mask: np.ndarray, remove_union: np.ndarray) -> float:
    """候補が REMOVE 和集合にどれだけ含まれるか = (seg ∩ remove) / area(seg)。"""
    seg = np.asarray(seg_mask, dtype=bool)
    area = int(seg.sum())
    if area == 0:
        return 0.0
    inter = int((seg & np.asarray(remove_union, dtype=bool)).sum())
    return inter / area


def is_covered(seg_mask: np.ndarray, remove_union: np.ndarray,
               threshold: float = DEFAULT_COVERED_THRESHOLD) -> bool:
    """候補の covered_ratio がしきい値以上なら True (表示抑制対象)。"""
    return covered_ratio(seg_mask, remove_union) >= threshold
