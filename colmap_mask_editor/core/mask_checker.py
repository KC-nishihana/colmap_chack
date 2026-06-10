"""
マスク品質チェック: チェック項目判定・ステータス決定
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.mask_io import get_colmap_mask_path

# 重大度順（小さいほど重大）
_SEVERITY: dict[str, int] = {
    "unreadable_image":    0,
    "unreadable_mask":     1,
    "size_mismatch":       2,
    "no_mask":             3,
    "empty_mask":          4,
    "full_mask":           5,
    "intermediate_values": 6,
    "not_saved":           7,
    "needs_check":         8,
    "ok":                  9,
}


@dataclass
class CheckResult:
    """マスク品質チェック結果"""
    image_exists: bool = False
    input_mask_exists: bool = False
    edited_mask_exists: bool = False
    colmap_mask_exists: bool = False
    image_readable: bool = False
    mask_readable: bool = False
    size_match: bool = False
    has_intermediate_values: bool = False
    is_empty_mask: bool = False
    is_full_mask: bool = False
    mask_ratio: float = 0.0
    image_width: int = 0
    image_height: int = 0
    mask_width: int = 0
    mask_height: int = 0
    status: str = "needs_check"
    note: str = ""


def check_image(entry, project_root: Path) -> CheckResult:
    """画像1枚のマスク品質チェックを実行する"""
    result = CheckResult()
    notes: list[str] = []

    # 画像存在確認
    result.image_exists = entry.image_path.exists()
    if not result.image_exists:
        result.status = "unreadable_image"
        result.note = "画像ファイルが存在しない"
        return result

    # 画像読み込み（日本語パス対応）
    buf = np.fromfile(str(entry.image_path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        result.status = "unreadable_image"
        result.note = "画像ファイルを読み込めない"
        return result

    result.image_readable = True
    h, w = (img.shape[0], img.shape[1]) if img.ndim >= 2 else (0, 0)
    result.image_width = w
    result.image_height = h

    # マスクパス確認
    colmap_path = get_colmap_mask_path(project_root, entry.rel_path)

    result.input_mask_exists = (entry.mask_path is not None and entry.mask_path.exists())
    result.edited_mask_exists = False
    result.colmap_mask_exists = colmap_path.exists()

    # チェック対象マスク（edited優先、次にinput）
    mask_to_check: Optional[Path] = None
    if result.input_mask_exists:
        mask_to_check = entry.mask_path

    if mask_to_check is None:
        if entry.is_modified:
            result.status = "not_saved"
            result.note = "未保存の編集がある"
        else:
            result.status = "no_mask"
            result.note = "マスクファイルが存在しない"
        return result

    # マスク読み込み（日本語パス対応）
    try:
        mbuf = np.fromfile(str(mask_to_check), dtype=np.uint8)
        mask_raw = cv2.imdecode(mbuf, cv2.IMREAD_UNCHANGED)
    except Exception:
        mask_raw = None

    if mask_raw is None:
        result.status = "unreadable_mask"
        result.note = "マスクファイルを読み込めない"
        return result

    result.mask_readable = True

    # グレースケール変換
    if mask_raw.ndim == 3:
        mask_gray = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)
    elif mask_raw.ndim == 4:
        mask_gray = cv2.cvtColor(mask_raw, cv2.COLOR_BGRA2GRAY)
    else:
        mask_gray = mask_raw.copy()

    mh, mw = mask_gray.shape
    result.mask_width = mw
    result.mask_height = mh

    # サイズ確認
    result.size_match = (mw == w and mh == h)

    # 中間値検出（0/255以外の値）
    unique_vals = np.unique(mask_gray)
    result.has_intermediate_values = bool(any(int(v) != 0 and int(v) != 255 for v in unique_vals))
    if result.has_intermediate_values:
        notes.append("入力マスクに中間値があったため2値化")

    # 2値化して統計計算
    _, binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    total = mw * mh
    black = int(np.sum(binary == 0))
    result.mask_ratio = black / total if total > 0 else 0.0
    result.is_empty_mask = (black == 0)   # 除外ピクセルなし = 全域有効
    result.is_full_mask = (result.mask_ratio >= 0.95)

    # 問題リスト収集（重大度順にソート）
    issues: list[str] = []
    if not result.size_match:
        issues.append("size_mismatch")
    if result.is_empty_mask:
        issues.append("empty_mask")
    if result.is_full_mask:
        issues.append("full_mask")
    if result.has_intermediate_values:
        issues.append("intermediate_values")
    if entry.is_modified:
        issues.append("not_saved")

    if issues:
        issues.sort(key=lambda s: _SEVERITY.get(s, 99))
        result.status = issues[0]
        secondary = issues[1:]
        if secondary:
            notes.append(f"その他の問題: {', '.join(secondary)}")
    else:
        result.status = "ok"

    result.note = "; ".join(notes)
    return result
