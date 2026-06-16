"""
V0.8: AMG レビュー結果から最終マスクを生成し、通常マスクへ一括適用する。

最終マスク生成時だけ RLE を復号する (compose_final_mask)。判定/IO の本体は純粋関数
として通常 pytest でテスト可能にし、複数画像IOを GUI スレッド外で実行する QThread
ラッパーも提供する。共通トランザクション基盤 core.mask_apply_tx を再利用する。

torch / sam2 に依存しない (NumPy のみ。QThread ラッパーのみ PySide6)。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ai import amg_manifest, amg_mask_composer, amg_npz
from ai.amg_mask_composer import FINAL_MASK_MODES
from ai.amg_remove_only import BASE_FULL
from core.mask_apply_tx import (
    ApplyTxError,
    ApplyTxOutcome,
    apply_masks_transaction,
    undo_from_record,
)
from core.mask_io import imread_jp

_log = logging.getLogger(__name__)

AMG_APPLY_RECORD_NAME = "amg_apply_record.json"


class AmgApplyError(Exception):
    pass


@dataclass(frozen=True)
class AmgApplyTarget:
    image_key: str
    cache_dir: str          # segmentation_cache/images/<cache_id>/
    save_path: str          # 通常マスクの保存先 (上書き対象)


def compose_target_mask(target: AmgApplyTarget, mode: str) -> np.ndarray:
    """1 画像の最終マスクを (H,W) uint8(0/255) で生成する。"""
    cache_dir = Path(target.cache_dir)
    npz_path = cache_dir / amg_manifest.SEGMENTS_NPZ_NAME
    manifest_path = cache_dir / amg_manifest.MANIFEST_NAME

    data = amg_npz.load_segments_npz(npz_path)
    manifest = amg_manifest.read_json(manifest_path)
    review = manifest.get("review", {})
    decisions = review.get("decisions", {})
    # base_mode=full のときは既存マスクがあっても全面 255 を基準にする。
    # (REMOVE_ONLY で「全面を基準に不要領域だけ除外」を選んだ場合の意図を尊重する)
    base_mode = review.get("base_mode")
    force_full_base = (base_mode == BASE_FULL)

    image_shape = np.asarray(data["image_shape"])
    h, w = int(image_shape[0]), int(image_shape[1])

    existing = None
    if (not force_full_base) and mode in (
        amg_mask_composer.MODE_EXCLUDE_REMOVE, amg_mask_composer.MODE_ADD_REMOVE
    ):
        img = imread_jp(Path(target.save_path))
        if img is not None:
            if img.ndim >= 3:
                img = img[:, :, 0]
            # 既存マスクがあってサイズ不一致なら黙って無視せず中止する。
            # 特に「不要領域を除外」では無視すると全面 255 から処理が始まり、
            # 既存マスクの内容が静かに失われてしまう。
            if img.shape != (h, w):
                raise AmgApplyError(
                    f"既存マスクと解析画像のサイズが一致しません: "
                    f"{target.image_key}: mask={img.shape}, image={(h, w)}"
                )
            existing = (img > 127).astype(np.uint8) * 255
    return amg_mask_composer.compose_final_mask(data, decisions, mode, existing_mask=existing)


def apply_amg_batch(
    targets: list[AmgApplyTarget],
    mode: str,
    backup_dir,
    *,
    job_id: str = "",
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> ApplyTxOutcome:
    """レビュー済み画像の最終マスクを通常マスクへ一括適用する (原子的・ロールバック付き)。"""
    if mode not in FINAL_MASK_MODES:
        raise AmgApplyError(f"不明な最終マスク方式: {mode!r}")
    try:
        return apply_masks_transaction(
            targets,
            produce_fn=lambda t: compose_target_mask(t, mode),
            backup_dir=backup_dir,
            key_of=lambda t: t.image_key,
            save_path_of=lambda t: t.save_path,
            job_id=job_id, apply_mode=mode, record_name=AMG_APPLY_RECORD_NAME,
            progress_cb=progress_cb, cancel_check=cancel_check,
        )
    except ApplyTxError as e:
        raise AmgApplyError(str(e)) from e


def undo_amg_batch(record_or_path) -> list[str]:
    """最後の AMG 一括適用を取り消す。"""
    return undo_from_record(record_or_path)


# --------------------------------------------------------------------------- #
# QThread ラッパー
# --------------------------------------------------------------------------- #

try:
    from PySide6.QtCore import QThread, Signal

    class AmgApplyWorker(QThread):
        progress = Signal(int, int, str)
        finished_ok = Signal(dict)
        failed = Signal(str)
        cancelled = Signal()

        def __init__(self, targets, mode, backup_dir, job_id="", parent=None):
            super().__init__(parent)
            self._targets = targets
            self._mode = mode
            self._backup_dir = backup_dir
            self._job_id = job_id
            self._cancel = False

        def cancel(self):
            self._cancel = True

        def run(self):
            try:
                outcome = apply_amg_batch(
                    self._targets, self._mode, self._backup_dir, job_id=self._job_id,
                    progress_cb=lambda d, t, k: self.progress.emit(d, t, k),
                    cancel_check=lambda: self._cancel,
                )
            except AmgApplyError as e:
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
    AmgApplyWorker = None  # type: ignore
