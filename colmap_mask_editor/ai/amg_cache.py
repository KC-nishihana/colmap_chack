"""
V0.8: 解析キャッシュの有効性 (reuse) / stale / corrupt 判定。

既存結果を再利用するのは、以下が「すべて」一致する場合だけ:
  image_key / file_size / mtime_ns / width / height /
  SAM 2 commit / model_id / checkpoint fingerprint /
  generator 設定 / NPZ schema / manifest schema / NPZ SHA-256

元画像または解析条件が変わったら stale。NPZ が壊れていたら corrupt。
自動削除はしない (再解析対象にするだけ)。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ai import amg_manifest, amg_npz
from ai.amg_manifest import (
    GENERATOR_KEYS,
    MANIFEST_SCHEMA_VERSION,
    MANIFEST_NAME,
    SEGMENTS_NPZ_NAME,
)
from ai.amg_protocol import AmgImageStatus

# キャッシュ判定結果
REUSABLE = "reusable"
STALE = "stale"
CORRUPT = "corrupt"
MISSING = "missing"

__all__ = [
    "REUSABLE",
    "STALE",
    "CORRUPT",
    "MISSING",
    "CacheCheck",
    "evaluate_cache",
    "recover_processing_states",
]


@dataclass
class CacheCheck:
    state: str
    reason: str = ""
    manifest: Optional[dict[str, Any]] = None


def _current_fingerprint(source_path) -> Optional[dict[str, int]]:
    try:
        st = os.stat(source_path)
    except OSError:
        return None
    return {"file_size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def evaluate_cache(
    cache_dir,
    *,
    source_path: str,
    model: dict[str, Any],
    generator: dict[str, Any],
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> CacheCheck:
    """
    1 画像のキャッシュ状態を判定する。

    返り値:
      MISSING  : manifest か NPZ が存在しない
      CORRUPT  : NPZ 検証失敗 / SHA-256 不一致 / manifest 破損
      STALE    : 元画像 or 設定変更で再解析が必要
      REUSABLE : 完全一致。既存結果を再利用してよい
    """
    cdir = Path(cache_dir)
    manifest_path = cdir / MANIFEST_NAME
    npz_path = cdir / SEGMENTS_NPZ_NAME

    if not manifest_path.exists() or not npz_path.exists():
        return CacheCheck(MISSING, "manifest または NPZ が存在しません")

    try:
        manifest = amg_manifest.read_json(manifest_path)
    except (ValueError, OSError) as e:
        return CacheCheck(CORRUPT, f"manifest 読込失敗: {e}")

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        return CacheCheck(STALE, "manifest schema version が異なります", manifest)

    # --- NPZ 整合・SHA-256 (corrupt 判定を stale より優先) ---
    try:
        amg_npz.verify_segments_npz(npz_path)
    except Exception as e:
        return CacheCheck(CORRUPT, f"NPZ 検証失敗: {e}", manifest)

    expected_sha = manifest.get("segments_npz_sha256")
    if expected_sha:
        actual_sha = amg_npz.file_sha256(npz_path)
        if actual_sha != expected_sha:
            return CacheCheck(CORRUPT, "NPZ SHA-256 不一致", manifest)

    # --- 元画像 fingerprint ---
    fp = _current_fingerprint(source_path)
    if fp is None:
        return CacheCheck(STALE, "元画像が見つかりません", manifest)
    mfp = manifest.get("source_fingerprint", {})
    if fp["file_size"] != mfp.get("file_size") or fp["mtime_ns"] != mfp.get("mtime_ns"):
        return CacheCheck(STALE, "元画像が更新されています", manifest)

    # width/height は指定時のみ比較 (skip 判定の高速パスでは省略可。fingerprint で代替)
    if width is not None and int(manifest.get("width", -1)) != int(width):
        return CacheCheck(STALE, "画像幅が異なります", manifest)
    if height is not None and int(manifest.get("height", -1)) != int(height):
        return CacheCheck(STALE, "画像高さが異なります", manifest)

    # --- モデル ---
    mmodel = manifest.get("model", {})
    for key in ("model_id", "sam2_commit", "checkpoint_fingerprint"):
        if mmodel.get(key) != model.get(key):
            return CacheCheck(STALE, f"モデル設定 ({key}) が異なります", manifest)

    # --- generator 設定 / settings_hash ---
    expected_hash = amg_manifest.settings_hash(generator, model)
    if manifest.get("settings_hash") != expected_hash:
        return CacheCheck(STALE, "解析設定 (settings_hash) が異なります", manifest)

    mgen = manifest.get("generator", {})
    for key in GENERATOR_KEYS:
        if mgen.get(key) != generator.get(key):
            return CacheCheck(STALE, f"generator 設定 ({key}) が異なります", manifest)

    return CacheCheck(REUSABLE, "完全一致", manifest)


def recover_processing_states(batch: dict[str, Any]) -> dict[str, Any]:
    """
    起動時の回復: batch_manifest 内で processing のまま残った画像を
    unprocessed へ戻す (前回異常終了とみなす)。新しい dict を返す。
    """
    images = batch.get("images", {})
    recovered = {}
    for key, entry in images.items():
        e = dict(entry)
        if e.get("status") == AmgImageStatus.PROCESSING:
            e["status"] = AmgImageStatus.UNPROCESSED
            e["error"] = None
        recovered[key] = e
    new_batch = dict(batch)
    new_batch["images"] = recovered
    new_batch["active_job_id"] = None
    return new_batch
