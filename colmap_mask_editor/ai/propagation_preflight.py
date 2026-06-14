"""
V0.7: 伝播開始前の事前検証 (純粋ロジック・numpy のみ・torch非依存)。

サイズ不一致を自動リサイズしない。基準マスクの妥当性・サイズ均一・基準位置・
重複・枚数上限を検査し、人間可読のエラー文字列リストを返す。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DimEntry:
    entry_key: str
    file_name: str
    width: int
    height: int


def validate_reference_mask(mask, width: int, height: int) -> list[str]:
    """基準マスクの妥当性を検査しエラー文字列を返す (空なら合格)。"""
    errors: list[str] = []
    arr = np.asarray(mask)
    if arr.ndim != 2:
        return [f"基準マスクは2次元である必要があります (shape={arr.shape})"]
    if arr.shape != (height, width):
        errors.append(
            f"基準マスクのサイズが画像と一致しません: "
            f"マスク {arr.shape[1]}x{arr.shape[0]} / 画像 {width}x{height}"
        )
    if arr.dtype != np.uint8:
        errors.append(f"基準マスクは uint8 である必要があります (dtype={arr.dtype})")
    uniq = set(np.unique(arr).tolist())
    if not uniq.issubset({0, 255}):
        errors.append(f"基準マスクは 0/255 のみである必要があります (値={sorted(uniq)})")
    fg = int((arr > 0).sum())
    total = arr.size
    if fg == 0:
        errors.append("基準マスクの前景が0ピクセルです")
    elif total and fg == total:
        errors.append("基準マスクが画像全体を占めています")
    return errors


def validate_sequence(
    dims: list[DimEntry],
    reference_entry_key: str,
    max_frames: int,
) -> list[str]:
    """対象シーケンスの妥当性を検査する (サイズ均一・基準位置・重複・枚数)。"""
    errors: list[str] = []

    if len(dims) < 2:
        errors.append("伝播対象が2枚未満です。基準を含め2枚以上選択してください。")

    keys = [d.entry_key for d in dims]
    if len(set(keys)) != len(keys):
        errors.append("対象画像に重複が含まれています。")

    if reference_entry_key not in keys:
        errors.append("基準画像が対象範囲内に存在しません。")

    if max_frames is not None and len(dims) > max_frames:
        errors.append(f"対象枚数 {len(dims)} が最大値 {max_frames} を超えています。")

    # サイズ均一 (基準画像のサイズを基準にする)
    ref = next((d for d in dims if d.entry_key == reference_entry_key), None)
    base = ref if ref is not None else (dims[0] if dims else None)
    if base is not None:
        for d in dims:
            if (d.width, d.height) != (base.width, base.height):
                errors.append(
                    "伝播対象に異なる画像サイズが含まれています。\n"
                    f"基準画像: {base.width} x {base.height}\n"
                    f"{d.file_name}: {d.width} x {d.height}\n"
                    "V0.7では同一サイズの画像だけを伝播できます。"
                )
                break

    return errors
