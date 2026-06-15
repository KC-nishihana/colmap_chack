"""
V0.9: partition_manifest.json (不変) と partition_review.json (判断/UI 状態) の
構築・原子更新・キャッシュ有効判定 (stdlib のみ)。

- partition.npz と partition_manifest は原則不変。
- 判断変更は partition_review.json だけを原子的に更新し、NPZ を書き換えない。
- segments.npz が再生成された等で stale になった場合、古い review は自動移行せず
  バックアップして新規レビューを開始する (ノード id が変わり得るため)。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from ai.amg_manifest import atomic_write_json, now_iso, read_json, source_fingerprint
from ai.partition_review_state import normalize_node_decisions

PARTITION_MANIFEST_SCHEMA_VERSION = 1
PARTITION_REVIEW_SCHEMA_VERSION = 1

PARTITION_NPZ_NAME = "partition.npz"
PARTITION_MANIFEST_NAME = "partition_manifest.json"
PARTITION_REVIEW_NAME = "partition_review.json"

# settings_hash に含めるキー (これらが変われば partition を作り直す)
SETTINGS_KEYS = (
    "backend", "working_max_side", "base_region_count", "default_visible_count",
    "min_region_area_ratio", "slic_region_size", "slic_ruler",
    "watershed_seed_spacing",
    "weight_color", "weight_texture", "weight_boundary", "weight_sam", "weight_size",
    "sam_sample_count", "sam_top_k",
)

__all__ = [
    "PARTITION_NPZ_NAME",
    "PARTITION_MANIFEST_NAME",
    "PARTITION_REVIEW_NAME",
    "partition_settings_hash",
    "build_partition_manifest",
    "build_partition_review",
    "update_partition_review",
    "partition_cache_status",
    "backup_review",
]


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def partition_settings_hash(settings: dict[str, Any]) -> str:
    """partition 設定から安定した settings_hash を生成する。"""
    payload = {k: settings.get(k) for k in SETTINGS_KEYS}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_partition_manifest(
    *,
    image_key: str,
    source_path: str,
    original_width: int,
    original_height: int,
    working_width: int,
    working_height: int,
    backend_requested: str,
    backend_used: str,
    leaf_count: int,
    node_count: int,
    root_id: int,
    default_visible_count: int,
    segments_npz_sha256: Optional[str],
    partition_npz_sha256: str,
    settings_hash: str,
    coverage: dict[str, Any],
    processing_time_sec: float,
    fingerprint: Optional[dict[str, int]] = None,
    warnings: Optional[list[str]] = None,
) -> dict[str, Any]:
    """partition_manifest.json 構造を作る (不変)。"""
    fp = fingerprint if fingerprint is not None else source_fingerprint(source_path)
    return {
        "schema_version": PARTITION_MANIFEST_SCHEMA_VERSION,
        "image_key": image_key,
        "source_path": str(source_path),
        "source_fingerprint": fp,
        "segments_npz_sha256": segments_npz_sha256,
        "partition_npz_sha256": partition_npz_sha256,
        "settings_hash": settings_hash,
        "backend_requested": backend_requested,
        "backend_used": backend_used,
        "working_width": int(working_width),
        "working_height": int(working_height),
        "original_width": int(original_width),
        "original_height": int(original_height),
        "leaf_count": int(leaf_count),
        "node_count": int(node_count),
        "root_id": int(root_id),
        "default_visible_count": int(default_visible_count),
        "coverage": coverage,
        "created_at": now_iso(),
        "processing_time_sec": float(processing_time_sec),
        "warnings": list(warnings or []),
    }


def build_partition_review(
    *,
    partition_npz_sha256: str,
    target_visible_count: int,
    node_decisions: Optional[dict] = None,
    expanded_nodes: Optional[list] = None,
    completed: bool = False,
    last_selected_node: Optional[int] = None,
) -> dict[str, Any]:
    """partition_review.json 構造を作る (判断/UI 状態のみ)。"""
    return {
        "schema_version": PARTITION_REVIEW_SCHEMA_VERSION,
        "partition_npz_sha256": partition_npz_sha256,
        "completed": bool(completed),
        "updated_at": now_iso(),
        "target_visible_count": int(target_visible_count),
        "expanded_nodes": list(expanded_nodes or []),
        "node_decisions": normalize_node_decisions(node_decisions or {}),
        "last_selected_node": last_selected_node,
    }


def update_partition_review(review_path, *, node_decisions=None,
                            target_visible_count=None, expanded_nodes=None,
                            completed=None, last_selected_node="__keep__",
                            node_count=None) -> dict[str, Any]:
    """
    partition_review.json だけを原子的に更新する。partition.npz は触らない。
    """
    review = read_json(review_path)
    if node_decisions is not None:
        review["node_decisions"] = normalize_node_decisions(node_decisions, node_count)
    if target_visible_count is not None:
        review["target_visible_count"] = int(target_visible_count)
    if expanded_nodes is not None:
        review["expanded_nodes"] = list(expanded_nodes)
    if completed is not None:
        review["completed"] = bool(completed)
    if last_selected_node != "__keep__":
        review["last_selected_node"] = last_selected_node
    review["updated_at"] = now_iso()
    atomic_write_json(review_path, review)
    return review


def partition_cache_status(
    manifest: dict[str, Any],
    *,
    source_fingerprint: dict[str, int],
    original_width: int,
    original_height: int,
    segments_npz_sha256: Optional[str],
    partition_npz_sha256: str,
    settings_hash: str,
) -> tuple[bool, str]:
    """
    キャッシュが再利用可能かを判定する。(valid, reason)。

    spec の全一致条件: image fingerprint / 幅・高さ / segments.npz SHA / settings_hash /
    partition.npz SHA / schema version。一つでも不一致なら stale。
    """
    if int(manifest.get("schema_version", -1)) != PARTITION_MANIFEST_SCHEMA_VERSION:
        return False, "schema_version 不一致"
    fp = manifest.get("source_fingerprint", {})
    if (int(fp.get("file_size", -1)) != int(source_fingerprint.get("file_size", -2))
            or int(fp.get("mtime_ns", -1)) != int(source_fingerprint.get("mtime_ns", -2))):
        return False, "元画像 fingerprint 不一致"
    if (int(manifest.get("original_width", -1)) != int(original_width)
            or int(manifest.get("original_height", -1)) != int(original_height)):
        return False, "画像サイズ不一致"
    if manifest.get("segments_npz_sha256") != segments_npz_sha256:
        return False, "segments.npz SHA-256 不一致 (V0.8 再生成)"
    if manifest.get("partition_npz_sha256") != partition_npz_sha256:
        return False, "partition.npz SHA-256 不一致 (破損/改変)"
    if manifest.get("settings_hash") != settings_hash:
        return False, "partition settings_hash 不一致"
    return True, "valid"


def backup_review(review_path) -> Optional[Path]:
    """既存 review を *.bak へ退避する (stale 時)。無ければ None。"""
    p = Path(review_path)
    if not p.exists():
        return None
    bak = p.with_suffix(p.suffix + f".{int(Path(p).stat().st_mtime_ns)}.bak")
    p.replace(bak)
    return bak
