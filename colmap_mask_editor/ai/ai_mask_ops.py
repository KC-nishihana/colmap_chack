"""
AI推論結果 (NPZ) の読み込み・候補統計・通常マスクへの適用。

NPZ仕様 (sam_backend/result_writer.py と対応):
  masks      : shape (N, H, W), dtype uint8, 値は 0 または 255
  scores     : shape (N,),       dtype float32
  request_id : int64
  image_key  : str

GUI 側はこのモジュールを使い、torch / sam2 に触れずに結果を扱う。
numpy のみに依存 (PySide6 にも依存しない)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AiCandidate:
    """候補マスク1件。"""
    index: int                 # 候補番号 (0始まり)
    mask: np.ndarray           # (H, W) uint8 0/255
    score: float               # 予測スコア
    fg_pixels: int             # 前景 (255) ピクセル数
    fg_ratio: float            # 前景率 (0.0〜1.0)

    @property
    def size(self) -> tuple[int, int]:
        """(width, height)"""
        h, w = self.mask.shape[:2]
        return (w, h)


@dataclass(frozen=True)
class AiPredictionResult:
    """predict 結果一式 (最大3候補)。"""
    request_id: int
    image_key: str
    width: int
    height: int
    candidates: list[AiCandidate]

    @property
    def mask_count(self) -> int:
        return len(self.candidates)

    def best_index(self) -> int:
        """最大スコアの候補インデックス。候補が無ければ -1。"""
        if not self.candidates:
            return -1
        return max(range(len(self.candidates)), key=lambda i: self.candidates[i].score)


class NpzCorruptError(Exception):
    """NPZ が壊れている / 期待するキーが無い場合。"""


def load_prediction_npz(
    npz_path,
    expected_request_id: Optional[int] = None,
    expected_image_key: Optional[str] = None,
    max_candidates: int = 3,
) -> AiPredictionResult:
    """
    予測結果NPZを読み込み AiPredictionResult を返す。

    expected_request_id / expected_image_key を指定すると、NPZ内のメタと
    一致しない場合 NpzCorruptError を送出する (古い結果の取り違え防止)。
    壊れている場合も NpzCorruptError。
    """
    path = Path(npz_path)
    if not path.exists():
        raise NpzCorruptError(f"NPZが存在しません: {path}")

    try:
        with np.load(str(path), allow_pickle=False) as data:
            if "masks" not in data or "scores" not in data:
                raise NpzCorruptError("NPZに masks / scores がありません")
            masks = np.asarray(data["masks"])
            scores = np.asarray(data["scores"], dtype=np.float32)
            req_id = int(data["request_id"]) if "request_id" in data else -1
            image_key = str(data["image_key"]) if "image_key" in data else ""
    except NpzCorruptError:
        raise
    except Exception as e:  # zip破損・pickle拒否など
        raise NpzCorruptError(f"NPZ読み込み失敗: {e}") from e

    if masks.ndim != 3:
        raise NpzCorruptError(f"masks の次元が不正です: shape={masks.shape}")
    if masks.dtype != np.uint8:
        masks = masks.astype(np.uint8)

    n, h, w = masks.shape
    if scores.shape[0] != n:
        raise NpzCorruptError(
            f"masks ({n}) と scores ({scores.shape[0]}) の件数が一致しません"
        )

    if expected_request_id is not None and req_id != expected_request_id:
        raise NpzCorruptError(
            f"request_id不一致 (NPZ={req_id}, 期待={expected_request_id})"
        )
    if expected_image_key is not None and image_key != expected_image_key:
        raise NpzCorruptError(
            f"image_key不一致 (NPZ={image_key!r}, 期待={expected_image_key!r})"
        )

    total = float(h * w) if h * w > 0 else 1.0
    candidates: list[AiCandidate] = []
    for i in range(min(n, max_candidates)):
        m = masks[i]
        # 念のため 0/255 へ正規化
        if m.max() == 1:
            m = (m * 255).astype(np.uint8)
        fg = int(np.count_nonzero(m >= 128))
        candidates.append(AiCandidate(
            index=i,
            mask=m,
            score=float(scores[i]),
            fg_pixels=fg,
            fg_ratio=fg / total,
        ))

    return AiPredictionResult(
        request_id=req_id,
        image_key=image_key,
        width=w,
        height=h,
        candidates=candidates,
    )


# ------------------------------------------------------------------ #
# 通常マスクへの適用 (GrabCut と同じ 追加/除外/置換 の3方式)
# ------------------------------------------------------------------ #

APPLY_ADD = "add"        # AI抽出領域を255にする
APPLY_EXCLUDE = "exclude"  # AI抽出領域を0にする
APPLY_REPLACE = "replace"  # AI結果でマスク全体を置換する


def apply_ai_mask(
    current_mask: np.ndarray,
    ai_mask: np.ndarray,
    mode: str,
) -> np.ndarray:
    """
    AI候補マスクを現在マスクへ合成する。current_mask は変更せず新配列を返す。

    mode:
      add      -> ai_mask が255の領域を255にする
      exclude  -> ai_mask が255の領域を0にする
      replace  -> ai_mask そのもので全体を置換する
    """
    if current_mask.shape[:2] != ai_mask.shape[:2]:
        raise ValueError(
            f"マスクサイズ不一致: current={current_mask.shape[:2]}, ai={ai_mask.shape[:2]}"
        )

    result = current_mask.copy()
    region = ai_mask >= 128

    if mode == APPLY_ADD:
        result[region] = 255
    elif mode == APPLY_EXCLUDE:
        result[region] = 0
    elif mode == APPLY_REPLACE:
        binary = np.where(region, 255, 0).astype(np.uint8)
        result[:] = binary
    else:
        raise ValueError(f"不明なmode: {mode!r}")

    return result
