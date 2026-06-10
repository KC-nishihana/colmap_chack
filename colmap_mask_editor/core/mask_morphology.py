"""
マスクのモルフォロジー処理: 膨張・収縮・穴埋め
"""

import cv2
import numpy as np


def get_ellipse_kernel(size: int) -> np.ndarray:
    s = max(1, size)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))


def dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = get_ellipse_kernel(kernel_size)
    return cv2.dilate(mask, kernel, iterations=1)


def erode_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = get_ellipse_kernel(kernel_size)
    return cv2.erode(mask, kernel, iterations=1)


def close_holes(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = get_ellipse_kernel(kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
