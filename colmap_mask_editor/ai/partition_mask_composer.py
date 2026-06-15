"""
V0.9: partition の葉判断から最終マスク (uint8, KEEP=255 / REMOVE=0) を生成する。

SAM RLE ではなく partition の葉 region_id と実効判断から作る。run-length を
判断値へ変換して C-order で展開するため、全葉マスクを個別復号しない。

  flat_index = y * width + x
  mask = flat.reshape(height, width)
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from ai import partition_review_state as prs

__all__ = ["leaf_decision_values", "compose_mask", "save_mask_png"]

KEEP_VALUE = np.uint8(255)
REMOVE_VALUE = np.uint8(0)


def leaf_decision_values(parent_array, leaf_count: int, decisions: dict,
                         *, unreviewed_as: str | None = None) -> np.ndarray:
    """
    葉 region_id (1..K) -> 0/255 の LUT を返す (index 0 はダミー)。

    unreviewed_as=None なら未確認が残っていれば ValueError。'keep'/'remove' を
    指定すると未確認をその値で確定する。
    """
    eff = prs.effective_leaf_decisions(parent_array, leaf_count, decisions)
    lut = np.zeros(int(leaf_count) + 1, dtype=np.uint8)
    unreviewed = eff[1:int(leaf_count) + 1] == 0
    if np.any(unreviewed):
        if unreviewed_as is None:
            raise ValueError("未確認の葉が残っています (最終確定前に解消してください)")
        fill = KEEP_VALUE if unreviewed_as == "keep" else REMOVE_VALUE
        lut[1:][unreviewed] = fill
    lut[1:][eff[1:int(leaf_count) + 1] == 1] = KEEP_VALUE
    lut[1:][eff[1:int(leaf_count) + 1] == 2] = REMOVE_VALUE
    return lut


def compose_mask(run_region_ids, run_lengths, height: int, width: int,
                 leaf_value_lut: np.ndarray) -> np.ndarray:
    """
    run-length region map と葉->値 LUT から (height, width) uint8 マスクを作る。

    run ごとに LUT を引いて np.repeat で展開し C-order で reshape する。
    """
    ids = np.asarray(run_region_ids).astype(np.int64)
    lengths = np.asarray(run_lengths).astype(np.int64)
    values = np.asarray(leaf_value_lut, dtype=np.uint8)[ids]
    flat = np.repeat(values, lengths)
    if flat.size != int(height) * int(width):
        raise ValueError(
            f"展開画素数 {flat.size} が {int(height)*int(width)} と一致しません"
        )
    return flat.reshape(int(height), int(width))


def save_mask_png(final_path, mask: np.ndarray) -> None:
    """マスク PNG を原子的に保存する (tmp -> flush/fsync -> os.replace)。"""
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_suffix(final.suffix + ".tmp.png")
    ok, buf = cv2.imencode(".png", np.asarray(mask, dtype=np.uint8))
    if not ok:
        raise RuntimeError("PNG エンコードに失敗しました")
    with open(tmp, "wb") as f:
        f.write(buf.tobytes())
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, final)
