"""
マスクの連結成分処理: 小さい白領域の除去
"""

import cv2
import numpy as np


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """面積が min_area 未満の白(255)連結成分を 0 にして返す"""
    result = mask.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for label in range(1, num_labels):  # 0 は背景
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            result[labels == label] = 0
    return result
