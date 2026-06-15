"""
V0.9: 完全被覆・階層型 partition の生成オーケストレーション (CPU 専用)。

torch / sam2 / PySide6 を import しない。日本語・全角スペースを含む Windows パスへ
対応する (cv2.imdecode + np.fromfile)。8K 画像は作業解像度で色/テクスチャ/グラフを
計算し、面積/bbox/重心と region map RLE・SAM シグネチャのみ元解像度で扱う
(全解像度 float 配列を大量保持しない)。

ステージ: loading -> base_partition -> boundary_cleanup -> sam_guidance ->
region_graph -> hierarchy_merge -> encoding -> validation -> saving -> completed。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from ai import partition_npz, partition_rle
from ai import partition_manifest as pman
from partition_backend import base_partition as bp
from partition_backend import region_features, region_graph, sam_guidance
from partition_backend import hierarchy_builder as hb

__all__ = ["build_partition", "PartitionCancelled", "load_image_unicode"]

STAGES = [
    "loading", "base_partition", "boundary_cleanup", "sam_guidance",
    "region_graph", "hierarchy_merge", "encoding", "validation",
    "saving", "completed",
]


class PartitionCancelled(Exception):
    """キャンセル要求で処理を中断したときに送出する。"""


def load_image_unicode(path) -> np.ndarray:
    """日本語/全角スペースを含むパスから BGR uint8 画像を読む。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        raise FileNotFoundError(f"画像が読めません (空): {path}")
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"画像のデコードに失敗しました: {path}")
    return img


def _settings_int(settings, key, default):
    try:
        return int(settings.get(key, default))
    except (TypeError, ValueError):
        return default


def build_partition(
    image_path,
    *,
    image_key: str,
    output_dir,
    settings: dict[str, Any],
    segments_path: Optional[str] = None,
    progress: Optional[Callable[[str, float, dict], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict[str, Any]:
    """
    1 画像の partition.npz / partition_manifest.json / partition_review.json を
    原子的に生成する。返り値は manifest dict。
    """
    t0 = time.time()

    def emit(stage: str, frac: float, **info):
        if progress is not None:
            progress(stage, float(frac), dict(info))

    def check_cancel():
        if should_cancel is not None and should_cancel():
            raise PartitionCancelled()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- loading ---- #
    emit("loading", 0.0)
    img = load_image_unicode(image_path)
    oh, ow = img.shape[:2]
    working_max_side = _settings_int(settings, "working_max_side", 2048)
    ww, wh = bp.compute_working_size(ow, oh, working_max_side)
    if (ww, wh) != (ow, oh):
        work_img = cv2.resize(img, (ww, wh), interpolation=cv2.INTER_AREA)
    else:
        work_img = img
    work_lab = bp.to_lab(work_img)
    work_grad = bp.gradient_magnitude(work_lab)
    check_cancel()

    # ---- base_partition + boundary_cleanup ---- #
    emit("base_partition", 0.1)
    backend_req = str(settings.get("backend", "auto"))
    base_region_count = _settings_int(settings, "base_region_count", 800)
    work_area = ww * wh
    ratio_units = _settings_int(settings, "min_region_area_ratio", 10)  # 0.01% 単位
    min_area_work = max(1, int(work_area * ratio_units / 10000.0))
    slic_region_size = _settings_int(settings, "slic_region_size", 0) or None
    slic_ruler = float(settings.get("slic_ruler", 10))
    seed_spacing = _settings_int(settings, "watershed_seed_spacing", 0) or None

    emit("boundary_cleanup", 0.2)
    work_labels, backend_used = bp.run_base_partition(
        work_img, backend_req,
        seed_spacing=seed_spacing, base_region_count=base_region_count,
        slic_region_size=slic_region_size, slic_ruler=slic_ruler,
        min_area=min_area_work,
    )
    work_labels = bp.relabel_sequential(work_labels).astype(np.int32)
    check_cancel()

    # 元解像度へ最近傍で戻し完全被覆を再検証
    full_labels = bp.upscale_labels_nearest(work_labels, ow, oh)
    full_labels = bp.relabel_sequential(full_labels).astype(np.int32)
    # work_labels も同じ id 体系へ揃える (relabel が一致するよう再マップ)
    if int(full_labels.max()) != int(work_labels.max()):
        work_labels = bp.upscale_labels_nearest(full_labels, ww, wh)
    bp.validate_base_labels(full_labels, oh, ow, check_connectivity=False)
    leaf_count = int(full_labels.max())

    # ---- 特徴量 (色/テクスチャは作業解像度, 面積/bbox/重心は元解像度) ---- #
    feat = region_features.compute_leaf_features(work_labels, work_lab, work_grad, leaf_count)
    feat.area = bp.region_areas(full_labels, leaf_count)
    feat.bbox = bp.region_bboxes(full_labels, leaf_count)
    feat.centroid = _full_centroid(full_labels, leaf_count)
    check_cancel()

    # ---- sam_guidance ---- #
    emit("sam_guidance", 0.4)
    segments_data = None
    segments_sha = None
    if segments_path and Path(segments_path).exists():
        from ai import amg_npz
        segments_data = amg_npz.load_segments_npz(segments_path)
        seg_shape = np.asarray(segments_data["image_shape"])
        if int(seg_shape[0]) != oh or int(seg_shape[1]) != ow:
            segments_data = None  # サイズ不一致の SAM は無視 (色/境界のみで統合)
        else:
            segments_sha = amg_npz.file_sha256(segments_path)
    sam_sample = _settings_int(settings, "sam_sample_count", 64)
    sam_top_k = _settings_int(settings, "sam_top_k", 4)
    sig = sam_guidance.compute_sam_signatures(
        full_labels, segments_data, sample_count=sam_sample, top_k=sam_top_k)
    check_cancel()

    # ---- region_graph ---- #
    emit("region_graph", 0.6)
    graph = region_graph.build_region_graph(work_labels, work_grad, leaf_count)
    check_cancel()

    # ---- hierarchy_merge ---- #
    emit("hierarchy_merge", 0.7)
    weights = hb.MergeWeights(
        color=_settings_int(settings, "weight_color", 30) / 100.0,
        texture=_settings_int(settings, "weight_texture", 10) / 100.0,
        boundary=_settings_int(settings, "weight_boundary", 30) / 100.0,
        sam=_settings_int(settings, "weight_sam", 25) / 100.0,
        size=_settings_int(settings, "weight_size", 5) / 100.0,
    )
    result = hb.build_hierarchy(feat, graph, sig, weights, ow * oh)
    check_cancel()

    # ---- encoding ---- #
    emit("encoding", 0.85)
    run_ids, run_len = partition_rle.encode_label_map(full_labels)
    arrays = partition_npz.build_partition_arrays(
        height=oh, width=ow,
        run_region_ids=run_ids, run_lengths=run_len,
        leaf_count=result.leaf_count,
        node_left=result.node_left, node_right=result.node_right,
        node_parent=result.node_parent, node_area=result.node_area,
        node_bbox=result.node_bbox, node_centroid=result.node_centroid,
        node_mean_lab=result.node_mean_lab, node_texture=result.node_texture,
        node_merge_cost=result.node_merge_cost, node_level=result.node_level,
        root_id=result.root_id,
        sam_sig_offsets=sig.offsets, sam_segment_ids=sig.segment_ids,
        sam_coverages=sig.coverages, sam_scores=sig.scores,
    )

    # ---- validation + saving (原子保存, 検証成功までは旧 partition を消さない) ---- #
    emit("validation", 0.92)
    check_cancel()
    emit("saving", 0.95)
    npz_path = output_dir / pman.PARTITION_NPZ_NAME
    partition_sha = partition_npz.save_partition_npz(npz_path, arrays)  # 内部 verify 込み

    coverage = bp.coverage_stats(full_labels)
    default_visible = _settings_int(settings, "default_visible_count", 30)
    settings_hash = pman.partition_settings_hash(settings)
    manifest = pman.build_partition_manifest(
        image_key=image_key, source_path=str(image_path),
        original_width=ow, original_height=oh,
        working_width=ww, working_height=wh,
        backend_requested=backend_req, backend_used=backend_used,
        leaf_count=result.leaf_count, node_count=result.node_count,
        root_id=result.root_id, default_visible_count=default_visible,
        segments_npz_sha256=segments_sha, partition_npz_sha256=partition_sha,
        settings_hash=settings_hash, coverage=coverage,
        processing_time_sec=time.time() - t0,
    )
    from ai.amg_manifest import atomic_write_json
    atomic_write_json(output_dir / pman.PARTITION_MANIFEST_NAME, manifest)

    # 既存 review が stale なら退避し新規作成
    review_path = output_dir / pman.PARTITION_REVIEW_NAME
    _reset_review_if_stale(review_path, partition_sha)
    if not review_path.exists():
        review = pman.build_partition_review(
            partition_npz_sha256=partition_sha,
            target_visible_count=default_visible)
        atomic_write_json(review_path, review)

    emit("completed", 1.0, leaf_count=result.leaf_count,
         node_count=result.node_count, coverage_ratio=coverage["coverage_ratio"])
    return manifest


def _full_centroid(labels: np.ndarray, k: int) -> np.ndarray:
    h, w = labels.shape
    flat = labels.reshape(-1).astype(np.int64)
    area = np.bincount(flat, minlength=k + 1).astype(np.float64)
    safe = np.where(area == 0, 1.0, area)
    ys, xs = np.indices((h, w))
    cx = np.bincount(flat, weights=xs.reshape(-1).astype(np.float64), minlength=k + 1) / safe
    cy = np.bincount(flat, weights=ys.reshape(-1).astype(np.float64), minlength=k + 1) / safe
    return np.stack([cx, cy], axis=1).astype(np.float32)


def _reset_review_if_stale(review_path, partition_sha: str) -> None:
    from ai.amg_manifest import read_json
    p = Path(review_path)
    if not p.exists():
        return
    try:
        review = read_json(p)
    except Exception:
        pman.backup_review(p)
        return
    if review.get("partition_npz_sha256") != partition_sha:
        pman.backup_review(p)  # ノード id が変わり得るため自動移行しない
