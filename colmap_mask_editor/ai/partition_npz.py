"""
V0.9: partition.npz の決定的ビルド・原子保存・読込・検証 (numpy のみ)。

partition.npz は完全被覆・階層型リージョンの不変バイナリ。判断状態 (keep/remove/
unreviewed) や UI 状態は一切保存しない (partition_review.json 側)。dense な H×W
ラベルマップも保存しない (run-length のみ)。

すべて allow_pickle=False で読める固定 dtype 配列。配列 index は node_id - 1。
  葉ノード     : node_id 1..leaf_count, left=right=0, merge_cost=0
  親ノード     : node_id leaf_count+1.., 統合で生成
  root         : node_parent=0

スキーマ (spec の partition.npz スキーマと一致):
  schema_version   uint16  (1,)
  image_shape      uint32  (2,)            [height, width]
  scan_order       uint8   (1,)            0 = C order
  leaf_count       uint32  (1,)
  node_count       uint32  (1,)
  root_id          uint32  (1,)
  run_region_ids   uint32  (R,)
  run_lengths      uint64  (R,)
  node_left        uint32  (node_count,)
  node_right       uint32  (node_count,)
  node_parent      uint32  (node_count,)
  node_area        uint64  (node_count,)
  node_bbox        int32   (node_count,4)  [x, y, w, h]
  node_centroid    float32 (node_count,2)  [cx, cy]
  node_mean_lab    float32 (node_count,3)
  node_texture     float32 (node_count,4)
  node_merge_cost  float32 (node_count,)
  node_level       uint16  (node_count,)
  sam_sig_offsets  uint64  (leaf_count+1,)
  sam_segment_ids  uint32  (S,)
  sam_coverages    float32 (S,)
  sam_scores       float32 (S,)

このモジュールは numpy のみに依存する (torch / sam2 / PySide6 非依存)。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from ai import partition_rle
from ai.partition_rle import RegionRleError

SCHEMA_VERSION = 1
SCAN_ORDER_C = 0

# 配列名 -> 期待 dtype
REQUIRED_ARRAYS: dict[str, np.dtype] = {
    "schema_version": np.dtype(np.uint16),
    "image_shape": np.dtype(np.uint32),
    "scan_order": np.dtype(np.uint8),
    "leaf_count": np.dtype(np.uint32),
    "node_count": np.dtype(np.uint32),
    "root_id": np.dtype(np.uint32),
    "run_region_ids": np.dtype(np.uint32),
    "run_lengths": np.dtype(np.uint64),
    "node_left": np.dtype(np.uint32),
    "node_right": np.dtype(np.uint32),
    "node_parent": np.dtype(np.uint32),
    "node_area": np.dtype(np.uint64),
    "node_bbox": np.dtype(np.int32),
    "node_centroid": np.dtype(np.float32),
    "node_mean_lab": np.dtype(np.float32),
    "node_texture": np.dtype(np.float32),
    "node_merge_cost": np.dtype(np.float32),
    "node_level": np.dtype(np.uint16),
    "sam_sig_offsets": np.dtype(np.uint64),
    "sam_segment_ids": np.dtype(np.uint32),
    "sam_coverages": np.dtype(np.float32),
    "sam_scores": np.dtype(np.float32),
}

__all__ = [
    "SCHEMA_VERSION",
    "SCAN_ORDER_C",
    "REQUIRED_ARRAYS",
    "PartitionNpzError",
    "build_partition_arrays",
    "save_partition_npz",
    "load_partition_npz",
    "verify_partition_npz",
    "file_sha256",
]


class PartitionNpzError(ValueError):
    """partition.npz の整合検証に失敗したときに送出する。"""


# ------------------------------------------------------------------ #
# 配列ビルド
# ------------------------------------------------------------------ #


def build_partition_arrays(
    *,
    height: int,
    width: int,
    run_region_ids: np.ndarray,
    run_lengths: np.ndarray,
    leaf_count: int,
    node_left: np.ndarray,
    node_right: np.ndarray,
    node_parent: np.ndarray,
    node_area: np.ndarray,
    node_bbox: np.ndarray,
    node_centroid: np.ndarray,
    node_mean_lab: np.ndarray,
    node_texture: np.ndarray,
    node_merge_cost: np.ndarray,
    node_level: np.ndarray,
    root_id: int,
    sam_sig_offsets: np.ndarray,
    sam_segment_ids: np.ndarray,
    sam_coverages: np.ndarray,
    sam_scores: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    階層ツリー + region map RLE + SAM シグネチャを partition.npz 用の固定 dtype
    配列 dict へまとめる。node 配列は index = node_id - 1 で渡すこと。
    """
    node_count = int(np.asarray(node_left).shape[0])
    arrays = {
        "schema_version": np.asarray([SCHEMA_VERSION], dtype=np.uint16),
        "image_shape": np.asarray([int(height), int(width)], dtype=np.uint32),
        "scan_order": np.asarray([SCAN_ORDER_C], dtype=np.uint8),
        "leaf_count": np.asarray([int(leaf_count)], dtype=np.uint32),
        "node_count": np.asarray([node_count], dtype=np.uint32),
        "root_id": np.asarray([int(root_id)], dtype=np.uint32),
        "run_region_ids": np.asarray(run_region_ids, dtype=np.uint32),
        "run_lengths": np.asarray(run_lengths, dtype=np.uint64),
        "node_left": np.asarray(node_left, dtype=np.uint32),
        "node_right": np.asarray(node_right, dtype=np.uint32),
        "node_parent": np.asarray(node_parent, dtype=np.uint32),
        "node_area": np.asarray(node_area, dtype=np.uint64),
        "node_bbox": np.asarray(node_bbox, dtype=np.int32).reshape(node_count, 4),
        "node_centroid": np.asarray(node_centroid, dtype=np.float32).reshape(node_count, 2),
        "node_mean_lab": np.asarray(node_mean_lab, dtype=np.float32).reshape(node_count, 3),
        "node_texture": np.asarray(node_texture, dtype=np.float32).reshape(node_count, 4),
        "node_merge_cost": np.asarray(node_merge_cost, dtype=np.float32),
        "node_level": np.asarray(node_level, dtype=np.uint16),
        "sam_sig_offsets": np.asarray(sam_sig_offsets, dtype=np.uint64),
        "sam_segment_ids": np.asarray(sam_segment_ids, dtype=np.uint32),
        "sam_coverages": np.asarray(sam_coverages, dtype=np.float32),
        "sam_scores": np.asarray(sam_scores, dtype=np.float32),
    }
    return arrays


# ------------------------------------------------------------------ #
# 原子保存・読込
# ------------------------------------------------------------------ #


def save_partition_npz(final_path, arrays: dict) -> str:
    """
    arrays を partition.npz として原子的に保存し SHA-256 を返す。

    tmp へ file object 経由で書き flush + fsync。確定前に allow_pickle=False で
    再読込検証。検証成功時のみ os.replace。
    """
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final.with_suffix(".npz.tmp")

    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())

    try:
        verify_partition_npz(tmp_path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise

    os.replace(tmp_path, final)
    return file_sha256(final)


def load_partition_npz(path) -> dict[str, np.ndarray]:
    """allow_pickle=False で NPZ を読み配列 dict を返す。"""
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


def verify_partition_npz(path) -> dict[str, np.ndarray]:
    """
    partition.npz を全面検証する。問題があれば PartitionNpzError、正常なら配列 dict。

    確認:
      - allow_pickle=False で読込可能
      - 必須配列の存在 / dtype / shape
      - dense な (>=2 次元かつ H×W 級) ラベルマップが無いこと
      - region map RLE の妥当性 (run 長合計・id 範囲)
      - ツリー整合 (葉/親/root, parent 逆参照, area 整合, bbox 包含, 循環なし)
      - SAM シグネチャ offsets の単調性
    """
    try:
        data = load_partition_npz(path)
    except ValueError as e:
        raise PartitionNpzError(f"NPZ を allow_pickle=False で読めません: {e}") from e

    allowed = set(REQUIRED_ARRAYS.keys())
    for name in data:
        if name not in allowed:
            raise PartitionNpzError(f"未知の配列 {name} が含まれます")
    for name, dtype in REQUIRED_ARRAYS.items():
        if name not in data:
            raise PartitionNpzError(f"必須配列 {name} がありません")
        if data[name].dtype != dtype:
            raise PartitionNpzError(
                f"{name} の dtype {data[name].dtype} が期待値 {dtype} と一致しません"
            )

    if int(data["schema_version"][0]) != SCHEMA_VERSION:
        raise PartitionNpzError(f"schema_version {int(data['schema_version'][0])} 非対応")
    if int(data["scan_order"][0]) != SCAN_ORDER_C:
        raise PartitionNpzError("scan_order が C order (0) ではありません")

    if data["image_shape"].shape != (2,):
        raise PartitionNpzError("image_shape の shape が (2,) ではありません")
    h, w = int(data["image_shape"][0]), int(data["image_shape"][1])
    leaf_count = int(data["leaf_count"][0])
    node_count = int(data["node_count"][0])
    root_id = int(data["root_id"][0])

    if not (1 <= leaf_count <= node_count):
        raise PartitionNpzError(f"leaf_count {leaf_count} / node_count {node_count} が不整合")
    # 完全二分マージツリー: node_count == 2*leaf_count - 1
    if node_count != 2 * leaf_count - 1:
        raise PartitionNpzError(
            f"node_count {node_count} が 2*leaf_count-1 ({2 * leaf_count - 1}) と一致しません"
        )
    if root_id != node_count:
        raise PartitionNpzError(f"root_id {root_id} が node_count {node_count} と一致しません")

    # 1 次元 (node_count,) 配列
    for name in ("node_left", "node_right", "node_parent", "node_area",
                 "node_merge_cost", "node_level"):
        if data[name].shape != (node_count,):
            raise PartitionNpzError(f"{name} の shape {data[name].shape} != ({node_count},)")
    for name, cols in (("node_bbox", 4), ("node_centroid", 2),
                       ("node_mean_lab", 3), ("node_texture", 4)):
        if data[name].shape != (node_count, cols):
            raise PartitionNpzError(
                f"{name} の shape {data[name].shape} != ({node_count},{cols})"
            )
        if data[name].ndim != 2:
            raise PartitionNpzError(f"{name} が 2 次元ではありません")

    # region map RLE
    run_ids = data["run_region_ids"]
    run_len = data["run_lengths"]
    try:
        partition_rle.validate_region_rle(run_ids, run_len, h, w, leaf_count)
    except RegionRleError as e:
        raise PartitionNpzError(f"region map RLE が不正: {e}") from e
    # 全葉が出現すること (葉が孤立せず必ず画素を持つ)
    used = np.unique(run_ids.astype(np.int64))
    if used.size != leaf_count or int(used[0]) != 1 or int(used[-1]) != leaf_count:
        raise PartitionNpzError(
            f"run_region_ids が 1..{leaf_count} を完全被覆していません (出現 {used.size} 種)"
        )

    _verify_tree(data, leaf_count, node_count, root_id)
    _verify_sam_signatures(data, leaf_count)
    return data


def _verify_tree(data, leaf_count: int, node_count: int, root_id: int) -> None:
    left = data["node_left"].astype(np.int64)
    right = data["node_right"].astype(np.int64)
    parent = data["node_parent"].astype(np.int64)
    area = data["node_area"].astype(np.int64)
    bbox = data["node_bbox"].astype(np.int64)

    def idx(node_id: int) -> int:
        return node_id - 1

    # 葉ノード: left=right=merge_cost=0
    for nid in range(1, leaf_count + 1):
        if left[idx(nid)] != 0 or right[idx(nid)] != 0:
            raise PartitionNpzError(f"葉ノード {nid} に子が設定されています")
    # 親ノード: 子が範囲内、双方向 parent 整合
    seen_child: set[int] = set()
    for nid in range(leaf_count + 1, node_count + 1):
        l = int(left[idx(nid)])
        r = int(right[idx(nid)])
        if not (1 <= l <= node_count) or not (1 <= r <= node_count):
            raise PartitionNpzError(f"親ノード {nid} の子 {l},{r} が範囲外")
        if l == r:
            raise PartitionNpzError(f"親ノード {nid} の左右の子が同一 {l}")
        for c in (l, r):
            if c in seen_child:
                raise PartitionNpzError(f"ノード {c} が複数の親を持ちます")
            seen_child.add(c)
            if int(parent[idx(c)]) != nid:
                raise PartitionNpzError(f"ノード {c} の parent が {nid} を指していません")
        # 面積 = 子面積合計
        if int(area[idx(nid)]) != int(area[idx(l)]) + int(area[idx(r)]):
            raise PartitionNpzError(f"親ノード {nid} の面積が子の合計と一致しません")
        # bbox が子を包含
        px, py, pw, ph = bbox[idx(nid)]
        for c in (l, r):
            cx, cy, cw, ch = bbox[idx(c)]
            if cx < px or cy < py or cx + cw > px + pw or cy + ch > py + ph:
                raise PartitionNpzError(f"親ノード {nid} の bbox が子 {c} を包含していません")

    # root の parent は 0、それ以外は親を持つ
    if int(parent[idx(root_id)]) != 0:
        raise PartitionNpzError("root の parent が 0 ではありません")
    for nid in range(1, node_count + 1):
        if nid == root_id:
            continue
        if int(parent[idx(nid)]) == 0:
            raise PartitionNpzError(f"ノード {nid} が root 以外なのに parent=0")
    # 全ノードが root へ到達 (循環なし)
    for nid in range(1, node_count + 1):
        steps = 0
        cur = nid
        while cur != root_id:
            cur = int(parent[idx(cur)])
            steps += 1
            if cur == 0 or steps > node_count:
                raise PartitionNpzError(f"ノード {nid} が root へ到達しません (循環の可能性)")


def _verify_sam_signatures(data, leaf_count: int) -> None:
    offsets = data["sam_sig_offsets"].astype(np.int64)
    if offsets.shape != (leaf_count + 1,):
        raise PartitionNpzError("sam_sig_offsets の shape が (leaf_count+1,) ではありません")
    if int(offsets[0]) != 0:
        raise PartitionNpzError("sam_sig_offsets の先頭が 0 ではありません")
    if np.any(np.diff(offsets) < 0):
        raise PartitionNpzError("sam_sig_offsets が単調増加ではありません")
    s = int(data["sam_segment_ids"].shape[0])
    if int(offsets[-1]) != s:
        raise PartitionNpzError(
            f"sam_sig_offsets 末尾 {int(offsets[-1])} が sam_segment_ids 長 {s} と不一致"
        )
    for name in ("sam_coverages", "sam_scores"):
        if data[name].shape != (s,):
            raise PartitionNpzError(f"{name} の shape {data[name].shape} != ({s},)")
