"""
V0.8: SAM 2.1 Automatic Mask Generator の uncompressed RLE を NumPy のみで扱う。

GUI 側では sam2 / torch を import せず、このモジュールだけで RLE の復号・
クリック判定・検証ができる。固定 SAM 2 コミット
(2b90b9f5ceec907a1c18123530e92e794ad901a4) の
sam2.utils.amg.mask_to_rle_pytorch / rle_to_mask / area_from_rle と互換。

RLE 仕様 (SAM 2 uncompressed_rle と一致):
  - segmentation = {"size": [h, w], "counts": [int, ...]}
  - 走査順: Fortran order (列優先)。すなわちピクセル (x, y) の 1 次元位置は
    flat_index = y + x * height
  - counts: 背景長, 前景長, 背景長, 前景長, ... の交互配列
  - 必ず背景 (0) の連続長から開始する (先頭が前景なら counts[0] = 0)
  - area = sum(counts[1::2])  (前景 run の総和)

このモジュールは numpy のみに依存する (torch / sam2 / PySide6 非依存)。
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

__all__ = [
    "pack_rles",
    "unpack_counts",
    "decode_rle",
    "rle_contains_point",
    "rle_area",
    "validate_rle",
    "encode_mask",
]


class RleError(ValueError):
    """RLE が不正なときに送出する。"""


# ------------------------------------------------------------------ #
# 検証
# ------------------------------------------------------------------ #


def validate_rle(counts, height: int, width: int) -> None:
    """
    counts が SAM2 uncompressed RLE として妥当か検証する。不正なら RleError。

    確認内容:
      - counts が 1 次元
      - counts が整数
      - counts が負数を含まない
      - counts の合計が height * width
    """
    arr = np.asarray(counts)
    if arr.ndim != 1:
        raise RleError(f"counts は 1 次元である必要があります (ndim={arr.ndim})")
    if arr.size == 0:
        raise RleError("counts が空です")
    if not np.issubdtype(arr.dtype, np.integer):
        # 整数値の float は許容するが、非整数値は拒否する
        if not np.all(np.equal(np.mod(arr, 1), 0)):
            raise RleError("counts が整数ではありません")
        arr = arr.astype(np.int64)
    if np.any(arr < 0):
        raise RleError("counts に負数が含まれます")
    total = int(arr.sum())
    expected = int(height) * int(width)
    if total != expected:
        raise RleError(
            f"counts 合計 {total} が画像サイズ {expected} (={height}x{width}) と一致しません"
        )


def rle_area(counts) -> int:
    """前景 run (奇数 index) の総和。SAM2 area_from_rle と一致。"""
    arr = np.asarray(counts)
    return int(arr[1::2].sum())


# ------------------------------------------------------------------ #
# 復号
# ------------------------------------------------------------------ #


def decode_rle(counts, height: int, width: int) -> np.ndarray:
    """
    RLE を (height, width) uint8 マスク (値 0 / 255) へ復号する。

    SAM2 rle_to_mask と同じく Fortran order で展開し、reshape(w, h).T で
    C order (h, w) へ戻す。
    """
    arr = np.asarray(counts)
    h, w = int(height), int(width)
    flat = np.zeros(h * w, dtype=bool)
    idx = 0
    parity = False  # 先頭は背景
    for count in arr.tolist():
        c = int(count)
        if c:
            if parity:
                flat[idx:idx + c] = True
            idx += c
        parity = not parity
    if idx != h * w:
        raise RleError(f"counts 合計 {idx} が {h*w} と一致しません")
    mask = flat.reshape(w, h).T  # Fortran 展開 -> C order (h, w)
    return (mask.astype(np.uint8)) * 255


def rle_contains_point(counts, height: int, width: int, x: int, y: int) -> bool:
    """
    全マスクを復号せずに、ピクセル (x, y) が前景かを判定する。

    Fortran order の 1 次元位置 flat_index = y + x * height を求め、
    counts を累積走査して該当 run の前景/背景を返す。範囲外は False。
    """
    h, w = int(height), int(width)
    xi, yi = int(x), int(y)
    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return False
    target = yi + xi * h
    arr = np.asarray(counts)
    # 累積和で run 境界を作り、target が入る run の parity を求める。
    cum = np.cumsum(arr.astype(np.int64))
    # target < cum[i] となる最小 i が、target を含む run の index。
    run_index = int(np.searchsorted(cum, target, side="right"))
    # 偶数 index = 背景, 奇数 index = 前景
    return (run_index % 2) == 1


# ------------------------------------------------------------------ #
# エンコード (主にテスト・round-trip 用。Worker は SAM2 出力をそのまま使う)
# ------------------------------------------------------------------ #


def encode_mask(mask: np.ndarray) -> list[int]:
    """
    (h, w) の 2 値マスクを SAM2 uncompressed RLE の counts へ変換する。

    SAM2 mask_to_rle_pytorch と同じ規則 (Fortran order, 背景開始, 交互)。
    True/非0 を前景とみなす。
    """
    m = np.asarray(mask)
    if m.ndim != 2:
        raise RleError("mask は 2 次元である必要があります")
    h, w = m.shape
    boolmask = m.astype(bool)
    flat = boolmask.T.reshape(-1)  # C order (h,w) -> Fortran flatten
    # run length encode
    if flat.size == 0:
        return [0]
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    lengths = np.diff(bounds).astype(np.int64)
    counts = [] if not bool(flat[0]) else [0]
    counts.extend(int(v) for v in lengths)
    return counts


# ------------------------------------------------------------------ #
# パック (annotations -> 連結 NPZ 配列群)
# ------------------------------------------------------------------ #


def _ann_counts(ann: dict[str, Any]) -> np.ndarray:
    """annotation から uncompressed RLE の counts を取り出す。"""
    seg = ann["segmentation"]
    if not isinstance(seg, dict) or "counts" not in seg or "size" not in seg:
        raise RleError("annotation['segmentation'] が uncompressed RLE ではありません")
    counts = np.asarray(seg["counts"], dtype=np.int64)
    return counts


def pack_rles(annotations: Iterable[dict[str, Any]], height: int, width: int) -> dict:
    """
    SAM2 AMG の annotations (uncompressed_rle) を NPZ 用配列群へ連結する。

    返り値の dict は amg_npz.save_segments_npz へそのまま渡せる。各セグメントの
    counts は rle_counts へ連結し、rle_offsets で境界を表す。

    ここでは検証のみ行い、決定的ソート・segment_id 付与は呼び出し側
    (amg_npz.build_segment_arrays) で行う。本関数は「与えられた順」で連結する。
    """
    h, w = int(height), int(width)
    anns = list(annotations)
    n = len(anns)

    rle_counts_parts: list[np.ndarray] = []
    rle_offsets = np.zeros(n + 1, dtype=np.uint64)
    bbox_xywh = np.zeros((n, 4), dtype=np.int32)
    area = np.zeros(n, dtype=np.uint64)
    predicted_iou = np.zeros(n, dtype=np.float32)
    stability_score = np.zeros(n, dtype=np.float32)
    point_coords = np.zeros((n, 2), dtype=np.float32)
    crop_box_xywh = np.zeros((n, 4), dtype=np.int32)

    offset = 0
    for i, ann in enumerate(anns):
        counts = _ann_counts(ann)
        validate_rle(counts, h, w)
        a = rle_area(counts)
        declared_area = int(ann.get("area", a))
        if declared_area != a:
            raise RleError(
                f"segment {i}: area {declared_area} が RLE 前景長 {a} と一致しません"
            )
        rle_counts_parts.append(counts.astype(np.uint64))
        offset += int(counts.size)
        rle_offsets[i + 1] = offset

        bbox = ann["bbox"]
        bbox_xywh[i] = [int(round(v)) for v in bbox]
        _validate_bbox(bbox_xywh[i], h, w)
        area[i] = a
        predicted_iou[i] = float(ann.get("predicted_iou", 0.0))
        stability_score[i] = float(ann.get("stability_score", 0.0))
        pc = ann.get("point_coords") or [[0.0, 0.0]]
        point_coords[i] = [float(pc[0][0]), float(pc[0][1])]
        cb = ann.get("crop_box", [0, 0, w, h])
        crop_box_xywh[i] = [int(round(v)) for v in cb]

    if rle_counts_parts:
        rle_counts = np.concatenate(rle_counts_parts).astype(np.uint64)
    else:
        rle_counts = np.zeros(0, dtype=np.uint64)

    return {
        "image_shape": np.asarray([h, w], dtype=np.uint32),
        "rle_offsets": rle_offsets,
        "rle_counts": rle_counts,
        "bbox_xywh": bbox_xywh,
        "area": area,
        "predicted_iou": predicted_iou,
        "stability_score": stability_score,
        "point_coords": point_coords,
        "crop_box_xywh": crop_box_xywh,
    }


def _validate_bbox(bbox, height: int, width: int) -> None:
    x, y, bw, bh = (int(v) for v in bbox)
    if bw < 0 or bh < 0:
        raise RleError(f"bbox の幅/高さが負です: {bbox}")
    if x < 0 or y < 0 or x + bw > width or y + bh > height:
        raise RleError(f"bbox {bbox} が画像範囲 {width}x{height} を超えます")


def unpack_counts(npz_data, segment_index: int) -> np.ndarray:
    """
    NPZ データ (np.load の結果 or dict) から segment_index の counts を取り出す。

    rle_offsets[i] から rle_offsets[i+1] までが segment i の counts。
    """
    offsets = np.asarray(npz_data["rle_offsets"])
    counts = np.asarray(npz_data["rle_counts"])
    i = int(segment_index)
    if i < 0 or i + 1 >= offsets.size:
        raise IndexError(f"segment_index {i} が範囲外です")
    start = int(offsets[i])
    end = int(offsets[i + 1])
    return counts[start:end]
