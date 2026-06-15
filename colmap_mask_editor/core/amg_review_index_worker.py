"""
V0.10: review_index (重複グループ計算) を GUI スレッド外で構築する CPU Worker。

候補グループ計算は O(候補数^2) の bbox 判定と RLE 比較を含むため、GUI スレッドで
実行しない。純粋関数 ensure_review_index() を通常 pytest でテストでき、QThread
ラッパー AmgReviewIndexWorker で非同期実行・キャンセルできる。

torch / sam2 に依存しない (NumPy + stdlib。QThread ラッパーのみ PySide6)。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ai import amg_manifest, amg_npz, amg_review_index as ri

_log = logging.getLogger(__name__)

__all__ = [
    "ReviewIndexResult",
    "ReviewIndexCancelled",
    "ensure_review_index",
    "AmgReviewIndexWorker",
]


class ReviewIndexCancelled(Exception):
    """キャンセルされたときに送出する。"""


@dataclass(frozen=True)
class ReviewIndexResult:
    status: str                       # "reused" | "built"
    arrays: dict[str, np.ndarray]
    group_count: int
    segment_count: int


def ensure_review_index(
    cache_dir,
    *,
    iou_threshold: float = 0.85,
    containment_threshold: float = 0.95,
    force: bool = False,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> ReviewIndexResult:
    """
    cache_dir の review_index.npz を最新化する。

    キャッシュが有効 (NPZ SHA-256・しきい値一致) なら再計算せず再利用する。
    stale または force なら再計算して原子的に保存する。キャンセル可能。
    """
    cdir = Path(cache_dir)
    npz_path = cdir / amg_manifest.SEGMENTS_NPZ_NAME
    index_path = cdir / ri.REVIEW_INDEX_NPZ_NAME
    index_manifest_path = cdir / ri.REVIEW_INDEX_MANIFEST_NAME

    segments_sha = amg_npz.file_sha256(npz_path)
    settings_hash = ri.grouping_settings_hash(iou_threshold, containment_threshold)

    existing_manifest: dict[str, Any] | None = None
    if index_manifest_path.exists() and index_path.exists():
        try:
            existing_manifest = amg_manifest.read_json(index_manifest_path)
        except (ValueError, OSError):
            existing_manifest = None

    stale = ri.is_review_index_stale(
        existing_manifest, segments_npz_sha256=segments_sha, settings_hash=settings_hash)

    if not force and not stale:
        try:
            arrays = ri.load_review_index(index_path)
            return ReviewIndexResult(
                status="reused", arrays=arrays,
                group_count=int(existing_manifest.get("group_count", 0)),
                segment_count=int(arrays["segment_ids"].shape[0]),
            )
        except Exception as e:  # noqa: BLE001 - 壊れていれば作り直す
            _log.warning("review_index 再利用失敗 (再計算します): %s", e)

    if cancel_check and cancel_check():
        raise ReviewIndexCancelled()

    npz_data = amg_npz.load_segments_npz(npz_path)

    if cancel_check and cancel_check():
        raise ReviewIndexCancelled()

    arrays = ri.build_review_index_arrays(
        npz_data, iou_threshold=iou_threshold, containment_threshold=containment_threshold)
    group_count = int(np.unique(arrays["group_ids"]).size) if arrays["group_ids"].size else 0
    segment_count = int(arrays["segment_ids"].shape[0])

    if cancel_check and cancel_check():
        raise ReviewIndexCancelled()

    ri.save_review_index(index_path, arrays)
    manifest = ri.build_review_index_manifest(
        segments_npz_sha256=segments_sha, settings_hash=settings_hash,
        group_count=group_count, segment_count=segment_count)
    amg_manifest.atomic_write_json(index_manifest_path, manifest)

    return ReviewIndexResult(
        status="built", arrays=arrays,
        group_count=group_count, segment_count=segment_count)


# --------------------------------------------------------------------------- #
# QThread ラッパー
# --------------------------------------------------------------------------- #

try:
    from PySide6.QtCore import QThread, Signal

    class AmgReviewIndexWorker(QThread):
        finished_ok = Signal(str, dict)   # cache_id, {"status","group_count","segment_count"}
        failed = Signal(str, str)         # cache_id, message
        cancelled = Signal(str)           # cache_id

        def __init__(self, cache_dir, *, cache_id="", iou_threshold=0.85,
                     containment_threshold=0.95, force=False, parent=None):
            super().__init__(parent)
            self._cache_dir = cache_dir
            self._cache_id = cache_id
            self._iou = iou_threshold
            self._cont = containment_threshold
            self._force = force
            self._cancel = False

        def cancel(self):
            self._cancel = True

        def run(self):
            try:
                result = ensure_review_index(
                    self._cache_dir, iou_threshold=self._iou,
                    containment_threshold=self._cont, force=self._force,
                    cancel_check=lambda: self._cancel,
                )
            except ReviewIndexCancelled:
                self.cancelled.emit(self._cache_id)
                return
            except Exception as e:  # noqa: BLE001
                self.failed.emit(self._cache_id, str(e))
                return
            self.finished_ok.emit(self._cache_id, {
                "status": result.status,
                "group_count": result.group_count,
                "segment_count": result.segment_count,
            })

except ImportError:  # PySide6 が無い環境
    AmgReviewIndexWorker = None  # type: ignore
