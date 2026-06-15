"""
V0.8: segments.npz の決定的ビルド・原子保存・読込・検証 (numpy のみ)。

segments.npz は SAM 解析結果を保持する不変のバイナリファイル。
判断状態 (keep/remove/unreviewed) は NPZ へ保存しない (manifest.json 側)。

スキーマ (allow_pickle=False で読める形式。object 配列・pickle・dense マスク禁止):
  schema_version   uint16  (1,)
  image_shape      uint32  (2,)   [height, width]
  segment_ids      uint32  (N,)
  rle_offsets      uint64  (N+1,)
  rle_counts       uint64  (総counts数,)
  bbox_xywh        int32   (N,4)
  area             uint64  (N,)
  predicted_iou    float32 (N,)
  stability_score  float32 (N,)
  point_coords     float32 (N,2)
  crop_box_xywh    int32   (N,4)

rle_offsets[i]..rle_offsets[i+1] が segment i の counts。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ai import amg_rle
from ai.amg_rle import RleError

SCHEMA_VERSION = 1

REQUIRED_ARRAYS: dict[str, np.dtype] = {
    "schema_version": np.dtype(np.uint16),
    "image_shape": np.dtype(np.uint32),
    "segment_ids": np.dtype(np.uint32),
    "rle_offsets": np.dtype(np.uint64),
    "rle_counts": np.dtype(np.uint64),
    "bbox_xywh": np.dtype(np.int32),
    "area": np.dtype(np.uint64),
    "predicted_iou": np.dtype(np.float32),
    "stability_score": np.dtype(np.float32),
    "point_coords": np.dtype(np.float32),
    "crop_box_xywh": np.dtype(np.int32),
}

__all__ = [
    "SCHEMA_VERSION",
    "REQUIRED_ARRAYS",
    "build_segment_arrays",
    "save_segments_npz",
    "load_segments_npz",
    "verify_segments_npz",
    "file_sha256",
    "NpzValidationError",
]


class NpzValidationError(ValueError):
    """NPZ の整合検証に失敗したときに送出する。"""


# ------------------------------------------------------------------ #
# 決定的ソートと配列ビルド
# ------------------------------------------------------------------ #


def _sort_key(packed: dict, i: int):
    """area 降順 -> iou 降順 -> stability 降順 -> bbox x,y,w,h 昇順。"""
    bbox = packed["bbox_xywh"][i]
    return (
        -int(packed["area"][i]),
        -float(packed["predicted_iou"][i]),
        -float(packed["stability_score"][i]),
        int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
    )


def build_segment_arrays(annotations: Iterable[dict[str, Any]], height: int, width: int) -> dict:
    """
    SAM2 AMG annotations を決定的に並べ替え、segment_id を付与して NPZ 用
    配列群 (np.savez_compressed へ渡せる dict) を構築する。

    返却順は area 降順を主キーとした決定的順序。segment_id = 1..N。
    """
    packed = amg_rle.pack_rles(annotations, height, width)
    n = packed["area"].shape[0]

    order = sorted(range(n), key=lambda i: _sort_key(packed, i))

    # rle_counts を新しい順序で連結し直す
    old_offsets = packed["rle_offsets"]
    old_counts = packed["rle_counts"]
    new_parts: list[np.ndarray] = []
    new_offsets = np.zeros(n + 1, dtype=np.uint64)
    offset = 0
    for new_i, old_i in enumerate(order):
        s = int(old_offsets[old_i])
        e = int(old_offsets[old_i + 1])
        seg_counts = old_counts[s:e]
        new_parts.append(seg_counts)
        offset += int(seg_counts.size)
        new_offsets[new_i + 1] = offset

    new_counts = (
        np.concatenate(new_parts).astype(np.uint64)
        if new_parts else np.zeros(0, dtype=np.uint64)
    )
    idx = np.asarray(order, dtype=np.intp)

    arrays = {
        "schema_version": np.asarray([SCHEMA_VERSION], dtype=np.uint16),
        "image_shape": np.asarray([int(height), int(width)], dtype=np.uint32),
        "segment_ids": np.arange(1, n + 1, dtype=np.uint32),
        "rle_offsets": new_offsets,
        "rle_counts": new_counts,
        "bbox_xywh": packed["bbox_xywh"][idx].astype(np.int32),
        "area": packed["area"][idx].astype(np.uint64),
        "predicted_iou": packed["predicted_iou"][idx].astype(np.float32),
        "stability_score": packed["stability_score"][idx].astype(np.float32),
        "point_coords": packed["point_coords"][idx].astype(np.float32),
        "crop_box_xywh": packed["crop_box_xywh"][idx].astype(np.int32),
    }
    return arrays


# ------------------------------------------------------------------ #
# 原子保存
# ------------------------------------------------------------------ #


def save_segments_npz(final_path, arrays: dict) -> str:
    """
    arrays を segments.npz として原子的に保存し、SHA-256 を返す。

    一時ファイルへ file object 経由で書く (文字列パスの自動 .npz 付与を回避)。
    flush + fsync 後 os.replace。保存後に allow_pickle=False で再読込検証する。
    """
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final.with_suffix(".npz.tmp")

    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())

    # 確定前に検証 (壊れたファイルを ready にしない)
    try:
        verify_segments_npz(tmp_path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise

    os.replace(tmp_path, final)
    return file_sha256(final)


def load_segments_npz(path) -> dict[str, np.ndarray]:
    """allow_pickle=False で NPZ を読み、配列 dict を返す (メモリへ展開)。"""
    with np.load(path, allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ #
# 検証
# ------------------------------------------------------------------ #


def verify_segments_npz(path) -> dict[str, np.ndarray]:
    """
    NPZ を全面的に検証する。問題があれば NpzValidationError。正常なら配列 dict。

    確認:
      - allow_pickle=False で読込可能
      - 必須配列の存在 / dtype / shape 整合
      - rle_offsets 単調増加・先頭 0・末尾 == rle_counts 長
      - 各 segment の RLE counts 合計が画像サイズと一致
      - area が RLE 前景長と一致
      - segment_id 重複なし
      - dense な (N,H,W) マスク配列が含まれない
    """
    try:
        data = load_segments_npz(path)
    except ValueError as e:
        raise NpzValidationError(f"NPZ を allow_pickle=False で読めません: {e}") from e

    for name, dtype in REQUIRED_ARRAYS.items():
        if name not in data:
            raise NpzValidationError(f"必須配列 {name} がありません")
        if data[name].dtype != dtype:
            raise NpzValidationError(
                f"{name} の dtype {data[name].dtype} が期待値 {dtype} と一致しません"
            )

    # 余計な (巨大 dense) 配列が無いこと
    allowed = set(REQUIRED_ARRAYS.keys())
    for name in data:
        if name not in allowed:
            raise NpzValidationError(f"未知の配列 {name} が含まれます")
        if data[name].ndim >= 3:
            raise NpzValidationError(f"{name} が 3 次元以上です (dense マスク禁止)")

    if int(data["schema_version"][0]) != SCHEMA_VERSION:
        raise NpzValidationError(
            f"schema_version {int(data['schema_version'][0])} 非対応"
        )

    image_shape = data["image_shape"]
    if image_shape.shape != (2,):
        raise NpzValidationError("image_shape の shape が (2,) ではありません")
    h, w = int(image_shape[0]), int(image_shape[1])

    segment_ids = data["segment_ids"]
    n = int(segment_ids.shape[0])

    # shape 整合
    expected_shapes = {
        "segment_ids": (n,),
        "rle_offsets": (n + 1,),
        "bbox_xywh": (n, 4),
        "area": (n,),
        "predicted_iou": (n,),
        "stability_score": (n,),
        "point_coords": (n, 2),
        "crop_box_xywh": (n, 4),
    }
    for name, shape in expected_shapes.items():
        if data[name].shape != shape:
            raise NpzValidationError(
                f"{name} の shape {data[name].shape} が期待値 {shape} と一致しません"
            )

    if len(set(segment_ids.tolist())) != n:
        raise NpzValidationError("segment_id が重複しています")

    offsets = data["rle_offsets"]
    counts = data["rle_counts"]
    if n > 0:
        if int(offsets[0]) != 0:
            raise NpzValidationError("rle_offsets の先頭が 0 ではありません")
        diffs = np.diff(offsets.astype(np.int64))
        if np.any(diffs < 0):
            raise NpzValidationError("rle_offsets が単調増加ではありません")
    if int(offsets[-1]) != int(counts.shape[0]):
        raise NpzValidationError(
            f"rle_offsets 末尾 {int(offsets[-1])} が rle_counts 長 {int(counts.shape[0])} と不一致"
        )

    # 各セグメントの RLE / area / bbox
    for i in range(n):
        seg_counts = amg_rle.unpack_counts(data, i)
        try:
            amg_rle.validate_rle(seg_counts, h, w)
        except RleError as e:
            raise NpzValidationError(f"segment {i} の RLE が不正: {e}") from e
        a = amg_rle.rle_area(seg_counts)
        if int(data["area"][i]) != a:
            raise NpzValidationError(
                f"segment {i}: area {int(data['area'][i])} が RLE 前景長 {a} と不一致"
            )
        bbox = data["bbox_xywh"][i]
        bx, by, bw, bh = (int(v) for v in bbox)
        if bw < 0 or bh < 0 or bx < 0 or by < 0 or bx + bw > w or by + bh > h:
            raise NpzValidationError(f"segment {i}: bbox {bbox.tolist()} が範囲外")

    return data
