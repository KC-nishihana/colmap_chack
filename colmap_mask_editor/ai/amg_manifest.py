"""
V0.8: 画像ごとの manifest.json と batch_manifest.json の構築・原子的更新。

- segments.npz は不変。判断状態 (decisions) は manifest.json のみ原子更新する。
- cache_id は image_key の SHA-256 先頭 16 文字。元画像名をフォルダ名に使わない。
- settings_hash は generator + model 設定をキー順で正規化した JSON の SHA-256。

stdlib のみ (numpy/torch/sam2/PySide6 非依存)。
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ai.amg_protocol import AmgImageStatus
from ai.amg_review_state import VALID_DECISIONS, SegmentDecision, normalize_decisions

MANIFEST_SCHEMA_VERSION = 1
BATCH_MANIFEST_SCHEMA_VERSION = 1

# レビュー方式 (manifest.review.workflow)。従来データに workflow が無ければ standard。
REVIEW_WORKFLOW_STANDARD = "standard"
REVIEW_WORKFLOW_REMOVE_ONLY = "remove_only"

CACHE_DIRNAME = "segmentation_cache"
IMAGES_DIRNAME = "images"
BATCH_MANIFEST_NAME = "batch_manifest.json"
SEGMENTS_NPZ_NAME = "segments.npz"
MANIFEST_NAME = "manifest.json"

# ------------------------------------------------------------------ #
# Automatic Mask Generator プリセット
# ------------------------------------------------------------------ #

PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "points_per_side": 16,
        "points_per_batch": 64,
        "pred_iou_thresh": 0.85,
        "stability_score_thresh": 0.95,
        "box_nms_thresh": 0.7,
        "crop_n_layers": 0,
        "crop_n_points_downscale_factor": 1,
        "min_mask_region_area": 100,
        "use_m2m": False,
        "multimask_output": True,
    },
    "standard": {
        "points_per_side": 32,
        "points_per_batch": 64,
        "pred_iou_thresh": 0.8,
        "stability_score_thresh": 0.95,
        "box_nms_thresh": 0.7,
        "crop_n_layers": 0,
        "crop_n_points_downscale_factor": 1,
        "min_mask_region_area": 100,
        "use_m2m": False,
        "multimask_output": True,
    },
    "detailed": {
        "points_per_side": 32,
        "points_per_batch": 32,
        "pred_iou_thresh": 0.8,
        "stability_score_thresh": 0.95,
        "box_nms_thresh": 0.7,
        "crop_n_layers": 1,
        "crop_n_points_downscale_factor": 2,
        "min_mask_region_area": 50,
        "use_m2m": False,
        "multimask_output": True,
    },
}

DEFAULT_PRESET = "fast"

GENERATOR_KEYS = (
    "points_per_side",
    "points_per_batch",
    "pred_iou_thresh",
    "stability_score_thresh",
    "box_nms_thresh",
    "crop_n_layers",
    "crop_n_points_downscale_factor",
    "min_mask_region_area",
    "use_m2m",
    "multimask_output",
)

__all__ = [
    "PRESETS",
    "DEFAULT_PRESET",
    "GENERATOR_KEYS",
    "MANIFEST_SCHEMA_VERSION",
    "BATCH_MANIFEST_SCHEMA_VERSION",
    "preset_settings",
    "match_preset",
    "cache_id_for",
    "cache_dir_for",
    "settings_hash",
    "source_fingerprint",
    "now_iso",
    "atomic_write_json",
    "read_json",
    "build_image_manifest",
    "update_manifest_decisions",
    "update_manifest_review",
    "get_review_workflow",
    "REVIEW_WORKFLOW_STANDARD",
    "REVIEW_WORKFLOW_REMOVE_ONLY",
    "build_batch_manifest",
    "update_batch_image_entry",
]


# ------------------------------------------------------------------ #
# プリセット
# ------------------------------------------------------------------ #


def preset_settings(name: str) -> dict[str, Any]:
    """プリセット名から generator 設定 dict を返す (コピー)。"""
    if name not in PRESETS:
        raise ValueError(f"不明なプリセット: {name!r}")
    return dict(PRESETS[name])


def match_preset(settings: dict[str, Any]) -> str:
    """settings がいずれかのプリセットと一致すればその名前、なければ 'custom'。"""
    norm = {k: settings.get(k) for k in GENERATOR_KEYS}
    for name, preset in PRESETS.items():
        if all(norm.get(k) == preset.get(k) for k in GENERATOR_KEYS):
            return name
    return "custom"


# ------------------------------------------------------------------ #
# cache_id / settings_hash
# ------------------------------------------------------------------ #


def cache_id_for(image_key: str, length: int = 16) -> str:
    """image_key の SHA-256 先頭 length 文字。元画像名はフォルダ名に使わない。"""
    return hashlib.sha256(image_key.encode("utf-8")).hexdigest()[:length]


def cache_dir_for(project_root, image_key: str) -> Path:
    """project/segmentation_cache/images/<cache_id>/ を返す (作成しない)。"""
    cid = cache_id_for(image_key)
    return Path(project_root) / CACHE_DIRNAME / IMAGES_DIRNAME / cid


def _canonical_json(obj: Any) -> str:
    """キー順で正規化した JSON 文字列 (settings_hash 計算用)。"""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def settings_hash(generator: dict[str, Any], model: dict[str, Any]) -> str:
    """
    generator 設定と model 情報から安定した settings_hash を生成する。

    JSON をキー順で正規化してから SHA-256。点数・しきい値・モデル・コミット・
    checkpoint fingerprint が変われば hash が変わる。
    """
    payload = {
        "generator": {k: generator.get(k) for k in GENERATOR_KEYS},
        "model": {
            "model_id": model.get("model_id"),
            "sam2_commit": model.get("sam2_commit"),
            "checkpoint_fingerprint": model.get("checkpoint_fingerprint"),
        },
    }
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def source_fingerprint(source_path) -> dict[str, int]:
    """元画像の file_size と mtime_ns を取得する。"""
    st = os.stat(source_path)
    return {"file_size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ #
# 原子的 JSON I/O
# ------------------------------------------------------------------ #


def atomic_write_json(path, data: dict[str, Any]) -> None:
    """JSON を tmp へ書き flush + fsync して os.replace で確定する。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, p)


def read_json(path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ #
# 画像 manifest
# ------------------------------------------------------------------ #


def build_image_manifest(
    *,
    image_key: str,
    source_path: str,
    width: int,
    height: int,
    model: dict[str, Any],
    generator: dict[str, Any],
    preset: str,
    segment_count: int,
    segment_ids,
    segments_npz_sha256: str,
    processing_time_sec: float,
    fingerprint: Optional[dict[str, int]] = None,
    status: str = AmgImageStatus.READY,
    warnings: Optional[list[str]] = None,
    generator_effective: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """画像 1 枚分の manifest.json 構造を作る。decisions は全 unreviewed で初期化。

    `generator` はユーザー要求値 (キャッシュ判定 / settings_hash の基準)。
    高解像度で points_per_batch を自動縮小した等の実行時の実効値は
    `generator_effective` へ分離して記録する (キャッシュ判定には使わない)。
    こうしないと縮小実行された画像が次回起動で必ず stale 判定になる。
    """
    fp = fingerprint if fingerprint is not None else source_fingerprint(source_path)
    gen_block = {k: generator.get(k) for k in GENERATOR_KEYS}
    gen_block["preset"] = preset
    decisions = {str(int(sid)): "unreviewed" for sid in segment_ids}
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "image_key": image_key,
        "source_path": str(source_path),
        "source_fingerprint": fp,
        "width": int(width),
        "height": int(height),
        "model": {
            "model_id": model.get("model_id"),
            "sam2_commit": model.get("sam2_commit"),
            "checkpoint_fingerprint": model.get("checkpoint_fingerprint"),
        },
        "generator": gen_block,
        "generator_effective": dict(generator_effective or {}),
        "settings_hash": settings_hash(generator, model),
        "segment_count": int(segment_count),
        "segments_npz": SEGMENTS_NPZ_NAME,
        "segments_npz_sha256": segments_npz_sha256,
        "created_at": now_iso(),
        "processing_time_sec": float(processing_time_sec),
        "status": status,
        "review": {
            "completed": False,
            "updated_at": None,
            "decisions": decisions,
        },
        "warnings": list(warnings or []),
    }


def update_manifest_decisions(
    manifest_path,
    decisions: dict[str, str],
    completed: Optional[bool] = None,
) -> dict[str, Any]:
    """
    manifest.json の review.decisions のみを原子的に更新する。NPZ は触らない。

    decisions は normalize_decisions で検証してから書き込む。
    """
    manifest = read_json(manifest_path)
    segment_ids = list(manifest.get("review", {}).get("decisions", {}).keys())
    normalized = normalize_decisions(decisions, segment_ids=segment_ids)
    review = manifest.setdefault("review", {})
    review["decisions"] = normalized
    review["updated_at"] = now_iso()
    if completed is not None:
        review["completed"] = bool(completed)
    atomic_write_json(manifest_path, manifest)
    return manifest


def get_review_workflow(manifest: dict[str, Any]) -> str:
    """manifest からレビュー方式を取得する。未設定 (従来データ) は standard。"""
    return manifest.get("review", {}).get("workflow", REVIEW_WORKFLOW_STANDARD)


def update_manifest_review(
    manifest_path,
    *,
    decisions: dict[str, str],
    workflow: str = REVIEW_WORKFLOW_STANDARD,
    base_mode: Optional[str] = None,
    ui: Optional[dict[str, Any]] = None,
    completed: Optional[bool] = None,
) -> dict[str, Any]:
    """
    manifest.json の review ブロックを後方互換で更新する (NPZ は触らない)。

    workflow=remove_only のときは decisions を最小化し、REMOVE だけを保存する
    (大量の keep / unreviewed を書かない)。standard のときは従来どおり全件正規化する。
    base_mode / ui / completed は指定時のみ更新する。
    """
    manifest = read_json(manifest_path)
    segment_ids = list(manifest.get("review", {}).get("decisions", {}).keys())
    valid_ids = {str(s) for s in segment_ids}

    if workflow == REVIEW_WORKFLOW_REMOVE_ONLY:
        normalized: dict[str, str] = {}
        for key, value in (decisions or {}).items():
            skey = str(key)
            if value not in VALID_DECISIONS:
                raise ValueError(f"segment {skey}: 不明な判断状態 {value!r}")
            if valid_ids and skey not in valid_ids:
                continue
            if value == SegmentDecision.REMOVE.value:   # 最小保存: remove のみ
                normalized[skey] = value
    else:
        normalized = normalize_decisions(decisions, segment_ids=segment_ids or None)

    review = manifest.setdefault("review", {})
    review["workflow"] = workflow
    review["decisions"] = normalized
    review["updated_at"] = now_iso()
    if base_mode is not None:
        review["base_mode"] = base_mode
    if ui is not None:
        review["ui"] = dict(ui)
    if completed is not None:
        review["completed"] = bool(completed)
    atomic_write_json(manifest_path, manifest)
    return manifest


# ------------------------------------------------------------------ #
# batch_manifest
# ------------------------------------------------------------------ #


def build_batch_manifest(
    *,
    images: Optional[dict[str, Any]] = None,
    active_job_id: Optional[str] = None,
    last_job_id: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "schema_version": BATCH_MANIFEST_SCHEMA_VERSION,
        "active_job_id": active_job_id,
        "last_job_id": last_job_id,
        "updated_at": now_iso(),
        "images": images or {},
    }


def update_batch_image_entry(
    batch_manifest_path,
    image_key: str,
    *,
    cache_id: str,
    status: str,
    segment_count: int = 0,
    review_completed: bool = False,
    error: Optional[str] = None,
    active_job_id: Any = "__keep__",
    last_job_id: Any = "__keep__",
) -> dict[str, Any]:
    """batch_manifest.json の 1 画像エントリを原子的に更新する。"""
    p = Path(batch_manifest_path)
    if p.exists():
        batch = read_json(p)
    else:
        batch = build_batch_manifest()
    if status not in AmgImageStatus.ALL:
        raise ValueError(f"不明な画像状態: {status!r}")
    batch.setdefault("images", {})[image_key] = {
        "cache_id": cache_id,
        "status": status,
        "segment_count": int(segment_count),
        "review_completed": bool(review_completed),
        "error": error,
    }
    if active_job_id != "__keep__":
        batch["active_job_id"] = active_job_id
    if last_job_id != "__keep__":
        batch["last_job_id"] = last_job_id
    batch["updated_at"] = now_iso()
    atomic_write_json(p, batch)
    return batch
