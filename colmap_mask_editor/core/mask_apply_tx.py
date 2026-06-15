"""
共通マスク一括適用トランザクション基盤 (V0.8 で抽出)。

V0.7 の伝播一括適用と V0.8 の AMG 最終マスク一括適用が、バックアップ/一時生成/
コミット/ロールバック/取り消しの同じ仕組みを共有するための土台。独自複製を避ける。

手順:
  1. 各 target の出力マスクを produce_fn(target) で生成 (保存先は未変更)
  2. 既存マスクをバックアップ、出力を staged tmp へ保存
  3. 全 staged 成功を確認
  4. 保存先へ os.replace (commit)
  5. 失敗時はバックアップ復元・新規削除でロールバック
  6. 取り消し用 record を原子的に記録

torch / sam2 / PySide6 に依存しない。マスク保存は core.mask_io.imwrite_jp。
"""

from __future__ import annotations

import datetime
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ai.propagation_manifest import atomic_write_json, read_json
from core.mask_io import imwrite_jp

_log = logging.getLogger(__name__)

DEFAULT_RECORD_NAME = "apply_record.json"


class ApplyTxError(Exception):
    pass


@dataclass
class ApplyTxOutcome:
    applied: list[str] = field(default_factory=list)
    record_path: Optional[str] = None
    record: dict[str, Any] = field(default_factory=dict)


def _safe_name(key: str, idx: int) -> str:
    base = str(key).replace("\\", "_").replace("/", "_").replace(":", "_")
    return f"{idx:06d}__{base}.png"


def apply_masks_transaction(
    targets: list[Any],
    produce_fn: Callable[[Any], np.ndarray],
    backup_dir,
    *,
    key_of: Callable[[Any], str],
    save_path_of: Callable[[Any], str],
    job_id: str = "",
    apply_mode: str = "",
    record_name: str = DEFAULT_RECORD_NAME,
    extra_record: Optional[dict[str, Any]] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> ApplyTxOutcome:
    """
    targets を produce_fn が返すマスクで一括上書きする。原子的・ロールバック付き。

    produce_fn(target) -> (H,W) uint8 (0/255)。生成失敗は例外でよい (全体ロールバック)。
    """
    backup_dir = Path(backup_dir)
    staged_dir = backup_dir / "staged"
    backup_dir.mkdir(parents=True, exist_ok=True)
    staged_dir.mkdir(parents=True, exist_ok=True)

    total = len(targets)
    records: list[dict[str, Any]] = []

    # --- フェーズ1: 生成 + バックアップ + staged 保存 --------------------- #
    for i, t in enumerate(targets):
        if cancel_check is not None and cancel_check():
            _cleanup_staged(records)
            raise ApplyTxError("キャンセルされました")

        key = key_of(t)
        save_path = Path(save_path_of(t))

        try:
            new_mask = produce_fn(t)
        except Exception as e:  # noqa: BLE001
            _cleanup_staged(records)
            raise ApplyTxError(f"マスク生成に失敗: {key}: {e}") from e

        existed = save_path.exists()
        backup_path: Optional[Path] = None
        if existed:
            backup_path = backup_dir / _safe_name(key, i)
            shutil.copy2(save_path, backup_path)

        staged_path = staged_dir / _safe_name(key, i)
        if not imwrite_jp(staged_path, new_mask):
            _cleanup_staged(records)
            raise ApplyTxError(f"一時マスク生成に失敗: {key}")

        records.append({
            "entry_key": key,
            "save_path": str(save_path),
            "existed": existed,
            "backup_path": str(backup_path) if backup_path else None,
            "staged_path": str(staged_path),
        })
        if progress_cb is not None:
            progress_cb(i + 1, total, key)

    # --- フェーズ2: コミット (os.replace)。失敗でロールバック ------------- #
    committed: list[dict[str, Any]] = []
    try:
        for r in records:
            dest = Path(r["save_path"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(r["staged_path"], dest)
            committed.append(r)
    except Exception as e:  # noqa: BLE001
        rollback_committed(committed)
        _cleanup_staged(records)
        raise ApplyTxError(f"適用のコミットに失敗しロールバックしました: {e}") from e

    # --- フェーズ3: 取り消し用 record ----------------------------------- #
    record = {
        "job_id": job_id,
        "apply_mode": apply_mode,
        "applied_at": datetime.datetime.now().astimezone().isoformat(),
        "targets": [
            {k: r[k] for k in ("entry_key", "save_path", "existed", "backup_path")}
            for r in committed
        ],
    }
    if extra_record:
        record.update(extra_record)
    record_path = backup_dir / record_name
    atomic_write_json(record_path, record)

    return ApplyTxOutcome(
        applied=[r["entry_key"] for r in committed],
        record_path=str(record_path), record=record,
    )


def rollback_committed(committed: list[dict[str, Any]]) -> None:
    """コミット済みを元へ戻す (既存はバックアップ復元、新規は削除)。"""
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


def undo_from_record(record_or_path) -> list[str]:
    """最後の一括適用を取り消す。既存はバックアップ復元、新規作成は削除。"""
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
            try:
                if dest.exists():
                    dest.unlink()
                undone.append(t["entry_key"])
            except OSError:
                _log.error("取り消し削除失敗: %s", dest)
    return undone
