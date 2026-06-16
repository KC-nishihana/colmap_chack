"""
V0.10: REMOVE_ONLY レビュー用の索引 (review_index.npz) の構築・原子保存・検証・stale 判定。

review_index は segments.npz から派生する「確認順 / グループ / 代表候補」のキャッシュ。
判断状態 (decisions) は持たず、表示制御のための数値だけを保持する。

NPZ スキーマ (allow_pickle=False で読める形式。object 配列・dense マスク・dict 禁止):
  schema_version              uint16  (1,)
  segment_ids                 uint32  (N,)
  group_ids                   uint32  (N,)   IoU 重複グループ
  representative_segment_ids  uint32  (N,)
  parent_segment_ids          int64   (N,)   入れ子の親 segment_id / 親無しは -1 (V0.11)
  quality_scores              float32 (N,)   predicted_iou * stability_score
  priority_scores             float32 (N,)   確認順スコア (REMOVEらしさではない)
  edge_touch_flags            uint8   (N,)

priority_scores (確認順 — 不要領域を早く見つけるための表示順。意味分類ではない):
  priority = 0.60 * normalized_area + 0.25 * quality_score + 0.15 * edge_touch_score

review_index_manifest.json:
  schema_version / segments_npz_sha256 / settings_hash / group_count /
  segment_count / created_at

stale 判定: segments.npz の SHA-256 が変わった or グループしきい値 (settings_hash) が
変わったら stale。stdlib + numpy のみ。
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ai import amg_candidate_grouping as grouping

# V0.11: 重複/親子分離のためグループ算法を変更し parent_segment_ids を追加した。
# 旧キャッシュ (schema 1) は新算法で再計算させるため両 schema を 2 へ上げる。
SCHEMA_VERSION = 2
REVIEW_INDEX_NPZ_NAME = "review_index.npz"
REVIEW_INDEX_MANIFEST_NAME = "review_index_manifest.json"
MANIFEST_SCHEMA_VERSION = 2

# 確認順スコアの重み (REMOVEらしさではなく、不要領域を早く見つけるための確認順)
W_AREA = 0.60
W_QUALITY = 0.25
W_EDGE = 0.15

REQUIRED_ARRAYS: dict[str, np.dtype] = {
    "schema_version": np.dtype(np.uint16),
    "segment_ids": np.dtype(np.uint32),
    "group_ids": np.dtype(np.uint32),
    "representative_segment_ids": np.dtype(np.uint32),
    "parent_segment_ids": np.dtype(np.int64),
    "quality_scores": np.dtype(np.float32),
    "priority_scores": np.dtype(np.float32),
    "edge_touch_flags": np.dtype(np.uint8),
}

__all__ = [
    "SCHEMA_VERSION",
    "REVIEW_INDEX_NPZ_NAME",
    "REVIEW_INDEX_MANIFEST_NAME",
    "ReviewIndexError",
    "grouping_settings_hash",
    "priority_scores",
    "build_review_index_arrays",
    "save_review_index",
    "load_review_index",
    "verify_review_index",
    "build_review_index_manifest",
    "is_review_index_stale",
]


class ReviewIndexError(ValueError):
    """review_index の検証に失敗したときに送出する。"""


def grouping_settings_hash(iou_threshold: float, containment_threshold: float) -> str:
    """グループしきい値から安定した settings_hash を生成する (しきい値変更で stale)。"""
    payload = {
        "iou_threshold": round(float(iou_threshold), 6),
        "containment_threshold": round(float(containment_threshold), 6),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def priority_scores(npz_data, quality: np.ndarray, edge_flags: np.ndarray) -> np.ndarray:
    """
    確認順スコアを float32 (N,) で返す。

    priority = 0.60*normalized_area + 0.25*quality + 0.15*edge_touch
    normalized_area は最大面積で正規化 (0..1)。意味分類ではない。
    """
    area = np.asarray(npz_data["area"], dtype=np.float64)
    max_area = float(area.max()) if area.size and area.max() > 0 else 1.0
    normalized_area = area / max_area
    q = np.asarray(quality, dtype=np.float64)
    e = np.asarray(edge_flags, dtype=np.float64)
    pri = W_AREA * normalized_area + W_QUALITY * q + W_EDGE * e
    return pri.astype(np.float32)


def build_review_index_arrays(
    npz_data,
    *,
    iou_threshold: float = 0.85,
    containment_threshold: float = 0.95,
) -> dict[str, np.ndarray]:
    """segments.npz から review_index 用の配列 dict を構築する (グループ計算込み)。"""
    segment_ids = np.asarray(npz_data["segment_ids"]).astype(np.uint32)
    result = grouping.group_candidates(
        npz_data, iou_threshold=iou_threshold, containment_threshold=containment_threshold)
    quality = grouping.quality_scores(npz_data)
    edge = grouping.edge_touch_flags(npz_data)
    pri = priority_scores(npz_data, quality, edge)
    return {
        "schema_version": np.asarray([SCHEMA_VERSION], dtype=np.uint16),
        "segment_ids": segment_ids,
        "group_ids": result.group_ids.astype(np.uint32),
        "representative_segment_ids": result.representative_segment_ids.astype(np.uint32),
        "parent_segment_ids": result.parent_segment_ids.astype(np.int64),
        "quality_scores": quality.astype(np.float32),
        "priority_scores": pri.astype(np.float32),
        "edge_touch_flags": edge.astype(np.uint8),
    }


# ------------------------------------------------------------------ #
# 原子保存 / 読込 / 検証
# ------------------------------------------------------------------ #


def save_review_index(final_path, arrays: dict[str, np.ndarray]) -> None:
    """arrays を review_index.npz として原子的に保存する (tmp -> fsync -> replace)。"""
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final.with_suffix(".npz.tmp")
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    try:
        verify_review_index(tmp_path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
    os.replace(tmp_path, final)


def load_review_index(path) -> dict[str, np.ndarray]:
    """allow_pickle=False で review_index.npz を読み、配列 dict を返す。"""
    with np.load(path, allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def verify_review_index(path) -> dict[str, np.ndarray]:
    """review_index.npz を検証する。問題があれば ReviewIndexError。"""
    try:
        data = load_review_index(path)
    except ValueError as e:
        raise ReviewIndexError(f"allow_pickle=False で読めません: {e}") from e

    allowed = set(REQUIRED_ARRAYS)
    for name, dtype in REQUIRED_ARRAYS.items():
        if name not in data:
            raise ReviewIndexError(f"必須配列 {name} がありません")
        if data[name].dtype != dtype:
            raise ReviewIndexError(
                f"{name} の dtype {data[name].dtype} が期待値 {dtype} と不一致")
    for name in data:
        if name not in allowed:
            raise ReviewIndexError(f"未知の配列 {name} が含まれます")
        if data[name].ndim >= 2:
            raise ReviewIndexError(f"{name} が 2 次元以上です (dense 禁止)")
    if int(data["schema_version"][0]) != SCHEMA_VERSION:
        raise ReviewIndexError(f"schema_version {int(data['schema_version'][0])} 非対応")
    n = int(data["segment_ids"].shape[0])
    for name in ("group_ids", "representative_segment_ids", "parent_segment_ids",
                 "quality_scores", "priority_scores", "edge_touch_flags"):
        if data[name].shape != (n,):
            raise ReviewIndexError(f"{name} の shape {data[name].shape} が (N,)={n} と不一致")
    return data


# ------------------------------------------------------------------ #
# manifest
# ------------------------------------------------------------------ #


def build_review_index_manifest(
    *,
    segments_npz_sha256: str,
    settings_hash: str,
    group_count: int,
    segment_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "segments_npz_sha256": segments_npz_sha256,
        "settings_hash": settings_hash,
        "group_count": int(group_count),
        "segment_count": int(segment_count),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def is_review_index_stale(
    manifest: dict[str, Any] | None,
    *,
    segments_npz_sha256: str,
    settings_hash: str,
) -> bool:
    """
    review_index が再計算を要するか判定する。

    True (stale): manifest 無し / schema 不一致 / segments.npz の SHA-256 変化 /
                  グループしきい値 (settings_hash) 変化。
    """
    if not manifest:
        return True
    if int(manifest.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        return True
    if manifest.get("segments_npz_sha256") != segments_npz_sha256:
        return True
    if manifest.get("settings_hash") != settings_hash:
        return True
    return False
