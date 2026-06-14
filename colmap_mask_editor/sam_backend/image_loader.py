"""
Worker 側の画像読み込み。

GUI 側 (core/mask_io.imread_jp + MainWindow._select_image) と同じピクセル配置に
なるよう統一する:
  - 日本語/全角スペースを含むパス対応 (np.fromfile + cv2.imdecode)
  - グレースケール -> BGR
  - BGRA -> BGR (アルファ破棄)
  - SAM へは RGB で渡す

EXIF 方向: cv2.imdecode は EXIF 回転を適用しない。GUI 側 (imread_jp) も適用しない
ため、ここでも適用しない (両者で同一のピクセル配置・幅高さになる)。

このモジュールは numpy / cv2 のみに依存し torch を import しない。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageLoadError(Exception):
    pass


def load_image_bgr(path: str) -> np.ndarray:
    """GUI と同一規約で BGR uint8 (H,W,3) を返す。"""
    p = Path(path)
    if not p.exists():
        raise ImageLoadError(f"画像が存在しません: {path}")

    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        raise ImageLoadError(f"画像の読み込みに失敗しました: {path}: {e}") from e

    if img is None:
        raise ImageLoadError(f"画像をデコードできませんでした: {path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        pass
    else:
        raise ImageLoadError(f"対応していない画像形式です: shape={img.shape}")

    return np.ascontiguousarray(img, dtype=np.uint8)


def load_image_rgb(path: str) -> tuple[np.ndarray, int, int]:
    """SAM 用に RGB (H,W,3) uint8 と (width, height) を返す。"""
    bgr = load_image_bgr(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return np.ascontiguousarray(rgb), w, h
