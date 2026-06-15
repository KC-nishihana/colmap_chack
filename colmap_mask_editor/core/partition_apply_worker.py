"""
V0.9: partition レビュー結果から最終マスクを生成し通常マスクへ一括適用する。

最終マスクは SAM RLE ではなく partition の葉 region_id と実効判断から生成する
(KEEP=255 / REMOVE=0)。判定/IO の本体は純粋関数として通常 pytest でテスト可能にし、
複数画像 IO は GUI スレッド外の QThread で実行する。共通トランザクション基盤
core.mask_apply_tx を再利用し、原子保存・ロールバック・バッチ取り消しを提供する。

torch / sam2 に依存しない (NumPy のみ。QThread ラッパーのみ PySide6)。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ai import partition_manifest as pman
from ai import partition_mask_composer as pmc
from ai import partition_npz
from ai.amg_manifest import read_json
from core.mask_apply_tx import (
    ApplyTxError,
    ApplyTxOutcome,
    apply_masks_transaction,
    undo_from_record,
)

_log = logging.getLogger(__name__)

PARTITION_APPLY_RECORD_NAME = "partition_apply_record.json"


class PartitionApplyError(Exception):
    pass


@dataclass(frozen=True)
class PartitionApplyTarget:
    image_key: str
    cache_dir: str          # segmentation_cache/images/<cache_id>/
    save_path: str          # 通常マスクの保存先 (上書き対象)


def compose_target_mask(target: PartitionApplyTarget,
                        unreviewed_action: str = "ask") -> np.ndarray:
    """
    1 画像の最終マスクを (H,W) uint8(KEEP=255/REMOVE=0) で生成する。

    未確認が残る場合、unreviewed_action が 'keep'/'remove' ならその値で確定。
    'ask' (既定) のままなら未確認エラーで中止する (黙って確定しない)。
    """
    cache_dir = Path(target.cache_dir)
    npz_path = cache_dir / pman.PARTITION_NPZ_NAME
    review_path = cache_dir / pman.PARTITION_REVIEW_NAME

    data = partition_npz.load_partition_npz(npz_path)
    review = read_json(review_path)
    decisions = review.get("node_decisions", {})

    shape = np.asarray(data["image_shape"])
    h, w = int(shape[0]), int(shape[1])
    leaf_count = int(np.asarray(data["leaf_count"])[0])
    parent = np.asarray(data["node_parent"])

    unreviewed_as = None if unreviewed_action == "ask" else unreviewed_action
    try:
        lut = pmc.leaf_decision_values(parent, leaf_count, decisions,
                                       unreviewed_as=unreviewed_as)
    except ValueError as e:
        raise PartitionApplyError(
            f"{target.image_key}: 未確認領域が残っています ({e})") from e
    return pmc.compose_mask(data["run_region_ids"], data["run_lengths"], h, w, lut)


def apply_partition_batch(
    targets: list[PartitionApplyTarget],
    backup_dir,
    *,
    unreviewed_action: str = "ask",
    job_id: str = "",
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> ApplyTxOutcome:
    """レビュー済み画像の最終マスクを一括適用する (原子的・ロールバック付き)。"""
    try:
        return apply_masks_transaction(
            targets,
            produce_fn=lambda t: compose_target_mask(t, unreviewed_action),
            backup_dir=backup_dir,
            key_of=lambda t: t.image_key,
            save_path_of=lambda t: t.save_path,
            job_id=job_id, apply_mode="partition_replace",
            record_name=PARTITION_APPLY_RECORD_NAME,
            progress_cb=progress_cb, cancel_check=cancel_check,
        )
    except ApplyTxError as e:
        raise PartitionApplyError(str(e)) from e


def undo_partition_batch(record_or_path) -> list[str]:
    """最後の partition 一括適用を取り消す。"""
    return undo_from_record(record_or_path)


# --------------------------------------------------------------------------- #
# QThread ラッパー
# --------------------------------------------------------------------------- #

try:
    from PySide6.QtCore import QThread, Signal

    class PartitionApplyWorker(QThread):
        progress = Signal(int, int, str)
        finished_ok = Signal(dict)
        failed = Signal(str)
        cancelled = Signal()

        def __init__(self, targets, backup_dir, unreviewed_action="ask",
                     job_id="", parent=None):
            super().__init__(parent)
            self._targets = targets
            self._backup_dir = backup_dir
            self._unreviewed_action = unreviewed_action
            self._job_id = job_id
            self._cancel = False

        def cancel(self):
            self._cancel = True

        def run(self):
            try:
                outcome = apply_partition_batch(
                    self._targets, self._backup_dir,
                    unreviewed_action=self._unreviewed_action, job_id=self._job_id,
                    progress_cb=lambda d, t, k: self.progress.emit(d, t, k),
                    cancel_check=lambda: self._cancel,
                )
            except PartitionApplyError as e:
                if self._cancel:
                    self.cancelled.emit()
                else:
                    self.failed.emit(str(e))
                return
            except Exception as e:  # noqa: BLE001
                self.failed.emit(str(e))
                return
            self.finished_ok.emit(outcome.record)

except ImportError:  # PySide6 が無い環境 (純粋関数のみ利用)
    PartitionApplyWorker = None  # type: ignore
