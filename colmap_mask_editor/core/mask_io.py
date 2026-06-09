"""
マスク画像の読み込み・保存・パス解決を担うモジュール
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def imread_jp(path: Path) -> Optional[np.ndarray]:
    """
    日本語パス対応の画像読み込み。
    np.fromfile + cv2.imdecode を使用。
    """
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        return img
    except Exception as e:
        print(f"[ERROR] imread_jp: {path}: {e}")
        return None


def imwrite_jp(path: Path, img: np.ndarray) -> bool:
    """
    日本語パス対応の画像保存。
    cv2.imencode + tofile を使用。
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lower()
        success, buf = cv2.imencode(ext, img)
        if not success:
            return False
        buf.tofile(str(path))
        return True
    except Exception as e:
        print(f"[ERROR] imwrite_jp: {path}: {e}")
        return False


def load_mask(mask_path: Path, image_size: tuple[int, int]) -> tuple[np.ndarray, bool]:
    """
    マスク画像を読み込み、uint8 2値画像(0 or 255)として返す。
    image_size は (width, height)。
    戻り値: (mask_array, size_mismatch_flag)
    マスクが読み込めない場合は空マスクを返す。
    """
    img = imread_jp(mask_path)
    if img is None:
        print(f"[WARN] マスク読み込み失敗: {mask_path} -> 空マスクを使用")
        return _empty_mask(image_size), False

    # グレースケールに変換
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    # 2値化: 128以上を255、未満を0
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # サイズ確認
    h, w = binary.shape
    if (w, h) != image_size:
        return binary, True  # size_mismatch=True

    return binary, False


def load_mask_or_empty(
    mask_path: Optional[Path],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, bool]:
    """
    マスクパスが None の場合は空マスクを返す。
    戻り値: (mask_array, size_mismatch_flag)
    """
    if mask_path is None:
        return _empty_mask(image_size), False
    return load_mask(mask_path, image_size)


def _empty_mask(image_size: tuple[int, int]) -> np.ndarray:
    """画像サイズに合わせた空(全0)マスクを生成"""
    w, h = image_size
    return np.zeros((h, w), dtype=np.uint8)


def get_edited_mask_path(project_root: Path, rel_path: Path) -> Path:
    """
    masks_edited/ 以下の保存先パスを生成。
    常に .png 拡張子。
    """
    # 拡張子を .png に変換
    rel_png = rel_path.with_suffix(".png")
    return project_root / "masks_edited" / rel_png


def get_colmap_mask_path(project_root: Path, rel_path: Path) -> Path:
    """
    masks_colmap/ 以下の保存先パスを生成。
    ファイル名 = 元ファイル名 + ".png"
    """
    return project_root / "masks_colmap" / rel_path.parent / (rel_path.name + ".png")


def save_mask(
    mask: np.ndarray,
    save_path: Path,
) -> bool:
    """
    マスクをPNGとして保存する。
    """
    return imwrite_jp(save_path, mask)
