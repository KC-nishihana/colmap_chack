"""
GrabCutによる半自動マスク生成ツール (v0.4A)
"""

import cv2
import numpy as np


def run_grabcut_with_rect(
    image_bgr: np.ndarray,
    rect: tuple[int, int, int, int],
    iter_count: int = 5,
) -> np.ndarray:
    """
    矩形指定でGrabCutを実行し、抽出領域を255、それ以外を0とした2値マスクを返す。
    rect は (x, y, w, h)。
    戻り値は uint8 の 0/255 マスク。
    """
    img = _to_bgr(image_bgr)
    h, w = img.shape[:2]

    x, y, rw, rh = rect
    x = max(0, x)
    y = max(0, y)
    rw = min(rw, w - x)
    rh = min(rh, h - y)

    if rw < 2 or rh < 2:
        raise ValueError(f"矩形が小さすぎます: w={rw}, h={rh}")

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    gc_mask = np.zeros((h, w), np.uint8)

    cv2.grabCut(
        img, gc_mask, (x, y, rw, rh),
        bgd_model, fgd_model,
        iter_count, cv2.GC_INIT_WITH_RECT,
    )

    result = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        np.uint8(255),
        np.uint8(0),
    )
    return result.astype(np.uint8)


def run_grabcut_with_existing_mask(
    image_bgr: np.ndarray,
    current_mask: np.ndarray,  # noqa: ARG001 (将来拡張用)
    rect: tuple[int, int, int, int],
    iter_count: int = 5,
) -> np.ndarray:
    """
    可能であれば既存マスクも初期ヒントとして使うGrabCut。
    将来拡張用に用意した関数で、現在は run_grabcut_with_rect に委譲する。
    """
    return run_grabcut_with_rect(image_bgr, rect, iter_count)


def apply_grabcut_result(
    current_mask: np.ndarray,
    grabcut_mask: np.ndarray,
    mode: str,
) -> np.ndarray:
    """
    GrabCut結果を現在マスクへ合成する。

    mode:
      add     -> grabcut_mask が255の領域を255にする
      remove  -> grabcut_mask が255の領域を0にする
      replace -> grabcut_mask そのものに置換する
    """
    result = current_mask.copy()
    gc_region = grabcut_mask == 255

    if mode == "add":
        result[gc_region] = 255
    elif mode == "remove":
        result[gc_region] = 0
    elif mode == "replace":
        result[:] = grabcut_mask
    else:
        raise ValueError(f"不明なmode: {mode!r}")

    return result


# ------------------------------------------------------------------ #
# 内部ヘルパー
# ------------------------------------------------------------------ #

def _to_bgr(image: np.ndarray) -> np.ndarray:
    """グレースケール・BGRAをBGRに変換する"""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()
