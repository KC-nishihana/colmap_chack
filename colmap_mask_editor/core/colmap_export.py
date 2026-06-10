"""
COLMAP互換マスク出力: masks_edited/ -> masks_colmap/
"""

from pathlib import Path

import cv2
import numpy as np

from core.mask_io import get_colmap_mask_path, save_mask


def export_colmap_mask(entry, project_root: Path, mask: np.ndarray) -> bool:
    """1枚分のCOLMAP互換マスクを masks_colmap/ に出力する"""
    colmap_path = get_colmap_mask_path(project_root, entry.rel_path)
    return save_mask(mask, colmap_path)


def export_all_colmap_masks(project) -> tuple[int, int]:
    """
    全画像の編集済みマスク (masks_edited/) を masks_colmap/ に一括出力する。
    戻り値: (成功数, 失敗数)
    """
    ok = 0
    ng = 0
    for entry in project.entries:
        mask_path = entry.mask_path
        if mask_path is None or not mask_path.exists():
            ng += 1
            continue
        try:
            buf = np.fromfile(str(mask_path), dtype=np.uint8)
            mask = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        except Exception:
            mask = None
        if mask is None:
            ng += 1
            continue
        colmap_path = get_colmap_mask_path(project.root, entry.rel_path)
        if save_mask(mask, colmap_path):
            ok += 1
        else:
            ng += 1
    return ok, ng
