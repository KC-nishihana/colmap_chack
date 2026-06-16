"""
V0.11: 保存 / 確定時の自動品質チェック (numpy + stdlib のみ・Qt/torch 非依存)。

保存または「確定して次へ」の前にマスクを検証する。致命的問題 (errors) は保存不可、
警告 (warnings) は確認を促すだけで停止しない (ユーザーが [このまま確定] を選べる)。

検証 (errors — 保存を止める):
  - 画像とマスクのサイズ一致 (image_shape 指定時)
  - dtype が uint8
  - 画素値が 0 / 255 のみ

警告 (warnings — 確認のみ):
  - 除外率 0%
  - 除外率 95% 以上
  - マスクが全面 0
  - マスクが全面 255
  - 前回マスクとの差分率 50% 以上
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "QualityResult",
    "DEFAULT_WARN_EXCLUDE_HIGH",
    "DEFAULT_WARN_DIFF",
    "check_mask_quality",
]

DEFAULT_WARN_EXCLUDE_HIGH = 0.95   # 除外率 95% 以上で警告
DEFAULT_WARN_DIFF = 0.50           # 前回との差分 50% 以上で警告


@dataclass(frozen=True)
class QualityResult:
    ok: bool                       # 致命的エラーが無い (= 保存可能)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_px: int = 0
    effective_px: int = 0          # 255 (有効) 画素数
    excluded_px: int = 0           # 0 (除外) 画素数
    effective_ratio: float = 0.0
    excluded_ratio: float = 0.0
    diff_ratio: Optional[float] = None  # 前回マスクとの差分率 (前回無し/不一致は None)
    all_zero: bool = False
    all_full: bool = False

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)


def _to_2d(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim >= 3:
        arr = arr[:, :, 0]
    return arr


def check_mask_quality(
    mask: np.ndarray,
    image_shape: Optional[tuple[int, int]] = None,
    *,
    previous_mask: Optional[np.ndarray] = None,
    warn_exclude_high: float = DEFAULT_WARN_EXCLUDE_HIGH,
    warn_diff: float = DEFAULT_WARN_DIFF,
) -> QualityResult:
    """
    マスクを検証して QualityResult を返す。

    image_shape は (H, W)。previous_mask を渡すと差分率を計算する (形が一致する場合)。
    """
    errors: list[str] = []
    warnings: list[str] = []

    arr = _to_2d(mask)
    h, w = (int(arr.shape[0]), int(arr.shape[1])) if arr.ndim == 2 else (0, 0)

    # ----- 致命的検証 -----
    if arr.ndim != 2:
        errors.append(f"マスクが2次元ではありません (shape={np.asarray(mask).shape})")
    if image_shape is not None:
        ih, iw = int(image_shape[0]), int(image_shape[1])
        if (h, w) != (ih, iw):
            errors.append(
                f"画像とマスクのサイズが一致しません: mask=({h},{w}), image=({ih},{iw})")
    if arr.dtype != np.uint8:
        errors.append(f"マスクの dtype が uint8 ではありません ({arr.dtype})")

    # 画素値 0/255 のみ (空配列は検査しない)
    values_ok = True
    if arr.size:
        uniq = np.unique(arr)
        if not np.all(np.isin(uniq, (0, 255))):
            values_ok = False
            bad = [int(v) for v in uniq if v not in (0, 255)][:5]
            errors.append(f"マスクに 0/255 以外の画素値があります: {bad}")

    # ----- 画素統計 (値検証が通っていれば 255 を有効とみなす) -----
    total = int(arr.size)
    effective = int(np.count_nonzero(arr == 255)) if (arr.size and values_ok) else 0
    excluded = total - effective if total else 0
    eff_ratio = (effective / total) if total else 0.0
    exc_ratio = (excluded / total) if total else 0.0
    all_zero = bool(total and effective == 0)
    all_full = bool(total and excluded == 0)

    # ----- 前回マスクとの差分率 -----
    diff_ratio: Optional[float] = None
    if previous_mask is not None:
        prev = _to_2d(previous_mask)
        if prev.shape == arr.shape and arr.size:
            changed = int(np.count_nonzero((arr > 127) != (prev > 127)))
            diff_ratio = changed / total

    # ----- 警告 -----
    if total:
        if all_zero:
            warnings.append("マスクが全面0です (有効領域がありません)")
        if all_full:
            warnings.append("マスクが全面255です (除外領域がありません)")
        if exc_ratio == 0.0 and not all_full:
            warnings.append("除外率が0%です")
        if exc_ratio >= warn_exclude_high and not all_zero:
            warnings.append(f"除外率が高すぎます ({exc_ratio * 100:.1f}%)")
        if diff_ratio is not None and diff_ratio >= warn_diff:
            warnings.append(f"前回マスクとの差分が大きいです ({diff_ratio * 100:.1f}%)")

    return QualityResult(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        total_px=total,
        effective_px=effective,
        excluded_px=excluded,
        effective_ratio=eff_ratio,
        excluded_ratio=exc_ratio,
        diff_ratio=diff_ratio,
        all_zero=all_zero,
        all_full=all_full,
    )
