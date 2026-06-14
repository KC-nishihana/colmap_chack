"""
推論結果マスクの NPZ 書き出し。

仕様 (ai/ai_mask_ops.py と対応):
  masks      : (N, H, W) uint8 0/255
  scores     : (N,)       float32
  request_id : int64
  image_key  : str

一時ファイルは別名に書いてから os.replace() でアトミックに確定する
(GUI が書き込み途中のファイルを読まないように)。

numpy のみに依存。torch を import しない。
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np

from ai import runtime_paths


def write_result_npz(
    masks: np.ndarray,
    scores: np.ndarray,
    request_id: int,
    image_key: str,
    runtime_dir: Path | None = None,
) -> str:
    """
    結果を NPZ へ書き出し、確定後の絶対パスを返す。

    masks は (N,H,W)。0/1 でも 0/255 でも受け付け、0/255 へ正規化して保存する。
    """
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.dtype != np.uint8:
        masks = masks.astype(np.uint8)
    # 0/255 へ正規化
    if masks.size and masks.max() == 1:
        masks = (masks * 255).astype(np.uint8)
    masks = np.ascontiguousarray(masks)

    scores = np.asarray(scores, dtype=np.float32).reshape(-1)

    d = runtime_dir if runtime_dir is not None else runtime_paths.get_runtime_dir(create=True)
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)

    final_path = d / f"sam_result_{request_id}.npz"
    tmp_path = d / f".tmp_sam_result_{request_id}_{uuid.uuid4().hex}.npz"

    with open(tmp_path, "wb") as f:
        np.savez(
            f,
            masks=masks,
            scores=scores,
            request_id=np.int64(request_id),
            image_key=np.array(str(image_key)),
        )
        f.flush()
        os.fsync(f.fileno())

    os.replace(str(tmp_path), str(final_path))
    return str(final_path)
