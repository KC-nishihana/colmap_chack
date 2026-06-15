"""
V0.9: 葉 region_id マップの C-order run-length 圧縮・復号・点参照 (numpy のみ)。

V0.8 の amg_rle が SAM の Fortran-order 2 値 RLE を扱うのに対し、こちらは
完全被覆 partition の「全画素 -> 葉 region_id」マップ (値域 1..leaf_count) を
C-order (行優先) で run-length 圧縮する。

C-order の 1 次元位置:
  flat_index = y * width + x
  mask.reshape(height, width) で 2 次元へ戻る。

run 仕様:
  run_region_ids[i] が run_lengths[i] 画素だけ連続する。
  同じ region_id が連続する run は保存前に統合する (隣接 run の id は必ず異なる)。
  run_lengths の総和 == height * width。
  region_id はすべて 1..leaf_count (0 や負値を残さない)。

このモジュールは numpy のみに依存する (torch / sam2 / PySide6 非依存)。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "RegionRleError",
    "encode_label_map",
    "decode_to_label_map",
    "region_at_index",
    "region_at_point",
    "validate_region_rle",
]


class RegionRleError(ValueError):
    """region map RLE が不正なときに送出する。"""


def encode_label_map(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    (h, w) の葉 region_id マップを C-order RLE へ圧縮する。

    返り値: (run_region_ids uint32, run_lengths uint64)。
    同一 region_id の連続 run は統合済み。
    """
    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise RegionRleError(f"labels は 2 次元である必要があります (ndim={arr.ndim})")
    flat = np.ascontiguousarray(arr).reshape(-1)
    if flat.size == 0:
        raise RegionRleError("labels が空です")
    # 値が変化する境界を検出
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    run_region_ids = flat[bounds[:-1]].astype(np.uint32)
    run_lengths = np.diff(bounds).astype(np.uint64)
    return run_region_ids, run_lengths


def decode_to_label_map(
    run_region_ids: np.ndarray,
    run_lengths: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """RLE を (height, width) の葉 region_id マップ (uint32) へ復号する。"""
    ids = np.asarray(run_region_ids)
    lengths = np.asarray(run_lengths)
    h, w = int(height), int(width)
    if ids.shape != lengths.shape or ids.ndim != 1:
        raise RegionRleError("run_region_ids と run_lengths の shape が一致しません")
    total = int(lengths.astype(np.int64).sum())
    if total != h * w:
        raise RegionRleError(f"run 長合計 {total} が {h*w} と一致しません")
    flat = np.repeat(ids.astype(np.uint32), lengths.astype(np.int64))
    return flat.reshape(h, w)


def region_at_index(
    run_region_ids: np.ndarray,
    run_lengths: np.ndarray,
    flat_index: int,
    *,
    cum: np.ndarray | None = None,
) -> int:
    """
    全マップを復号せず、C-order flat_index の葉 region_id を返す。

    run_lengths の累積和に二分探索する。`cum` を渡すと毎回の cumsum を省ける
    (連続クリックでの再利用を想定)。
    """
    ids = np.asarray(run_region_ids)
    if cum is None:
        cum = np.cumsum(np.asarray(run_lengths).astype(np.int64))
    idx = int(flat_index)
    if idx < 0 or idx >= int(cum[-1]):
        raise IndexError(f"flat_index {idx} が範囲外です (0..{int(cum[-1]) - 1})")
    run = int(np.searchsorted(cum, idx, side="right"))
    return int(ids[run])


def region_at_point(
    run_region_ids: np.ndarray,
    run_lengths: np.ndarray,
    width: int,
    x: int,
    y: int,
    *,
    cum: np.ndarray | None = None,
) -> int:
    """ピクセル (x, y) の葉 region_id を返す (C-order)。"""
    flat_index = int(y) * int(width) + int(x)
    return region_at_index(run_region_ids, run_lengths, flat_index, cum=cum)


def validate_region_rle(
    run_region_ids: np.ndarray,
    run_lengths: np.ndarray,
    height: int,
    width: int,
    leaf_count: int,
) -> None:
    """
    region map RLE を検証する。不正なら RegionRleError。

    確認内容 (spec の必須検証):
      - 1 次元・同長
      - run 長合計 == height * width
      - region_id がすべて 1..leaf_count (0 や負値なし)
      - run_lengths がすべて正
      - 隣接 run の region_id が異なる (統合漏れなし)
    """
    ids = np.asarray(run_region_ids)
    lengths = np.asarray(run_lengths)
    if ids.ndim != 1 or lengths.ndim != 1:
        raise RegionRleError("run_region_ids / run_lengths は 1 次元である必要があります")
    if ids.shape[0] != lengths.shape[0]:
        raise RegionRleError("run_region_ids と run_lengths の長さが一致しません")
    if ids.size == 0:
        raise RegionRleError("run が空です")
    total = int(lengths.astype(np.int64).sum())
    expected = int(height) * int(width)
    if total != expected:
        raise RegionRleError(f"run 長合計 {total} が {expected} と一致しません")
    if np.any(lengths.astype(np.int64) <= 0):
        raise RegionRleError("run_lengths に 0 以下が含まれます")
    ids64 = ids.astype(np.int64)
    if np.any(ids64 < 1):
        raise RegionRleError("region_id に 0 以下 (未所属/負値) が含まれます")
    if np.any(ids64 > int(leaf_count)):
        raise RegionRleError(
            f"region_id が leaf_count {int(leaf_count)} を超えます (max={int(ids64.max())})"
        )
    if ids.size > 1 and np.any(ids[1:] == ids[:-1]):
        raise RegionRleError("隣接 run に同一 region_id があります (統合漏れ)")
