"""
Worker 側の SAM 2.1 推論ラッパー。

SAM2ImagePredictor を使い、画像Embedding生成と点/矩形プロンプト推論を行う。
クリック追加のたびにモデル再ロードや Embedding 再生成はしない
(set_image で1度だけ Embedding を生成し、predict では使い回す)。

torch / sam2 はこのモジュール内でのみ使用。
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

_log = logging.getLogger("sam_worker.predict")


class Sam2Predictor:
    """1つのモデルに対する set_image / predict を管理する。"""

    def __init__(self, model_manager) -> None:
        self._mm = model_manager
        self._image_key: Optional[str] = None
        self._image_size: Optional[tuple[int, int]] = None  # (w, h)
        self._precision: Optional[str] = None

    @property
    def image_key(self) -> Optional[str]:
        return self._image_key

    def _autocast(self):
        """ロード精度に合わせた autocast コンテキストを返す。"""
        import torch
        precision = getattr(self._mm, "_precision", "bf16")
        if precision == "bf16":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        if precision == "fp16":
            return torch.autocast("cuda", dtype=torch.float16)
        # fp32 は autocast なし
        from contextlib import nullcontext
        return nullcontext()

    def set_image(self, rgb: np.ndarray, image_key: str) -> float:
        """画像Embeddingを生成する。所要秒を返す。"""
        import torch
        predictor = self._mm.predictor
        if predictor is None:
            raise RuntimeError("モデルがロードされていません")

        t0 = time.perf_counter()
        with torch.inference_mode(), self._autocast():
            predictor.set_image(rgb)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        self._image_key = image_key
        h, w = rgb.shape[:2]
        self._image_size = (w, h)
        _log.info("Embedding生成: key=%s, %dx%d, %.3fs", image_key, w, h, elapsed)
        return elapsed

    def predict(
        self,
        points: list[dict],
        box: Optional[list] = None,
        multimask_output: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        推論を実行する。
        戻り値: (masks (N,H,W) uint8 0/255, scores (N,) float32, 所要秒)
        """
        import torch
        predictor = self._mm.predictor
        if predictor is None:
            raise RuntimeError("モデルがロードされていません")

        point_coords = None
        point_labels = None
        if points:
            point_coords = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
            point_labels = np.array([int(p["label"]) for p in points], dtype=np.int32)

        box_arr = None
        if box is not None:
            box_arr = np.array(box, dtype=np.float32).reshape(-1)

        t0 = time.perf_counter()
        with torch.inference_mode(), self._autocast():
            masks, scores, _logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_arr,
                multimask_output=multimask_output,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        # masks: (N, H, W) bool/float -> uint8 0/255。スコア降順に並べ替え。
        masks = np.asarray(masks)
        if masks.ndim == 2:
            masks = masks[None, ...]
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        order = np.argsort(-scores)
        masks = masks[order]
        scores = scores[order]

        masks_u8 = (masks > 0.5).astype(np.uint8) * 255
        return masks_u8, scores, elapsed

    def release(self) -> None:
        self._image_key = None
        self._image_size = None
