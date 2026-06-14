"""
V0.7: 伝播レビュー結果の一括適用 (トランザクション + バックアップ/ロールバック) と
最後のバッチの取り消し。複数画像の読込・保存を GUI スレッドで行わないため QThread
Worker も提供するが、判定/IOの本体は純粋関数として通常 pytest でテスト可能にする。

通常マスク保存処理は既存の core.mask_io / ai.ai_mask_ops を共通利用する (独自複製しない)。

トランザクション手順:
  1. 対象確定
  2. 既存マスクをバックアップ (コピー)
  3. 全出力マスクを一時生成 (staged tmp)
  4. 全一時生成成功を確認
  5. 保存先へ os.replace (commit)
  6. 失敗時はバックアップから復元・新規作成を削除しエラー

torch / sam2 / PySide6(コア関数) に依存しない (QThread ラッパーのみ PySide6)。
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ai.ai_mask_ops import APPLY_ADD, APPLY_EXCLUDE, APPLY_REPLACE, apply_ai_mask
from ai.propagation_manifest import atomic_write_json, read_json
from ai.propagation_staging import read_mask_png
from core.mask_io import imwrite_jp

_log = logging.getLogger(__name__)

APPLY_RECORD_NAME = "apply_record.json"


class ApplyError(Exception):
    pass


@dataclass(frozen=True)
class ApplyTarget:
    entry_key: str
    save_path: str          # 通常マスクの保存先 (上書き対象)
    result_mask_path: str   # 伝播結果PNG (0/255)


@dataclass
class ApplyOutcome:
    applied: list[str] = field(default_factory=list)
    record_path: Optional[str] = None
    record: dict[str, Any] = field(default_factory=dict)


def _safe_name(entry_key: str, idx: int) -> str:
    base = entry_key.replace("\\", "_").replace("/", "_").replace(":", "_")
    return f"{idx:06d}__{base}.png"


def apply_batch(
    targets: list[ApplyTarget],
    mode: str,
    backup_dir,
    *,
    job_id: str = "",
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> ApplyOutcome:
    """採用フレームを通常マスクへ一括適用する。原子的・ロールバック付き。"""
    if mode not in (APPLY_ADD, APPLY_EXCLUDE, APPLY_REPLACE):
        raise ApplyError(f"不明な適用モード: {mode!r}")
    backup_dir = Path(backup_dir)
    staged_dir = backup_dir / "staged"
    backup_dir.mkdir(parents=True, exist_ok=True)
    staged_dir.mkdir(parents=True, exist_ok=True)

    total = len(targets)
    records: list[dict[str, Any]] = []

    # --- フェーズ1: バックアップ + 一時生成 (保存先は未変更) --------------- #
    for i, t in enumerate(targets):
        if cancel_check is not None and cancel_check():
            _cleanup_staged(records)
            raise ApplyError("キャンセルされました")

        save_path = Path(t.save_path)
        result = read_mask_png(t.result_mask_path)  # (H,W) 0/255

        existed = save_path.exists()
        backup_path: Optional[Path] = None
        if existed:
            base = read_mask_png(save_path)
            if base.shape != result.shape:
                # サイズ不一致の既存マスクは安全のため空を土台にする
                base = np.zeros_like(result)
            backup_path = backup_dir / _safe_name(t.entry_key, i)
            shutil.copy2(save_path, backup_path)
        else:
            base = np.zeros_like(result)

        new_mask = apply_ai_mask(base, result, mode)

        staged_path = staged_dir / _safe_name(t.entry_key, i)
        if not imwrite_jp(staged_path, new_mask):
            _cleanup_staged(records)
            raise ApplyError(f"一時マスク生成に失敗: {t.entry_key}")

        records.append({
            "entry_key": t.entry_key,
            "save_path": str(save_path),
            "existed": existed,
            "backup_path": str(backup_path) if backup_path else None,
            "staged_path": str(staged_path),
        })
        if progress_cb is not None:
            progress_cb(i + 1, total, t.entry_key)

    # --- フェーズ2: コミット (os.replace)。失敗したらロールバック ---------- #
    committed: list[dict[str, Any]] = []
    try:
        for r in records:
            dest = Path(r["save_path"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(r["staged_path"], dest)
            committed.append(r)
    except Exception as e:  # noqa: BLE001
        _rollback(committed)
        _cleanup_staged(records)
        raise ApplyError(f"適用のコミットに失敗しロールバックしました: {e}") from e

    # --- フェーズ3: 取り消し用レコードを記録 ----------------------------- #
    import datetime
    record = {
        "job_id": job_id,
        "apply_mode": mode,
        "applied_at": datetime.datetime.now().astimezone().isoformat(),
        "targets": [
            {k: r[k] for k in ("entry_key", "save_path", "existed", "backup_path")}
            for r in committed
        ],
    }
    record_path = backup_dir / APPLY_RECORD_NAME
    atomic_write_json(record_path, record)

    return ApplyOutcome(
        applied=[r["entry_key"] for r in committed],
        record_path=str(record_path), record=record,
    )


def _rollback(committed: list[dict[str, Any]]) -> None:
    """コミット済みを元へ戻す (既存はバックアップから復元、新規は削除)。"""
    for r in reversed(committed):
        dest = Path(r["save_path"])
        try:
            if r["existed"] and r["backup_path"]:
                shutil.copy2(r["backup_path"], dest)
            elif not r["existed"] and dest.exists():
                dest.unlink()
        except OSError:
            _log.error("ロールバック失敗: %s", dest)


def _cleanup_staged(records: list[dict[str, Any]]) -> None:
    for r in records:
        sp = r.get("staged_path")
        try:
            if sp and Path(sp).exists():
                Path(sp).unlink()
        except OSError:
            pass


def undo_batch(record_or_path) -> list[str]:
    """最後の一括適用を取り消す。既存はバックアップ復元、新規作成は削除。

    record_or_path: apply_batch の record dict か apply_record.json のパス。
    取り消した entry_key のリストを返す。
    """
    record = record_or_path if isinstance(record_or_path, dict) else read_json(record_or_path)
    undone: list[str] = []
    for t in record.get("targets", []):
        dest = Path(t["save_path"])
        if t.get("existed") and t.get("backup_path"):
            try:
                shutil.copy2(t["backup_path"], dest)
                undone.append(t["entry_key"])
            except OSError:
                _log.error("取り消し復元失敗: %s", dest)
        else:
            # 新規作成だったもの -> 削除
            try:
                if dest.exists():
                    dest.unlink()
                undone.append(t["entry_key"])
            except OSError:
                _log.error("取り消し削除失敗: %s", dest)
    return undone


# --------------------------------------------------------------------------- #
# QThread ラッパー (複数画像IOをGUIスレッド外で実行)
# --------------------------------------------------------------------------- #

try:
    from PySide6.QtCore import QThread, Signal

    class PropagationApplyWorker(QThread):
        progress = Signal(int, int, str)   # done, total, entry_key
        finished_ok = Signal(dict)         # ApplyOutcome.record
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
                outcome = apply_batch(
                    self._targets, self._mode, self._backup_dir, job_id=self._job_id,
                    progress_cb=lambda d, t, k: self.progress.emit(d, t, k),
                    cancel_check=lambda: self._cancel,
                )
            except ApplyError as e:
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
    PropagationApplyWorker = None  # type: ignore
