"""
V0.11: 統合レビューの候補適用状態 (unified_review.json) の構築・原子保存・stale 判定。

位置: segmentation_cache/images/<cache_id>/unified_review.json

設計上の正本は「通常マスク」。この JSON は UI 補助情報 (どの候補をどう適用したか・
表示設定・完了フラグ) を保持するだけで、ここから勝手にマスクを再構成しない。
マスク SHA が一致しなければ stale とする。segments.npz は判断変更で書き換えない。

stdlib + numpy のみ (torch / sam2 / PySide6 非依存)。原子 I/O は amg_manifest を再利用。
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

import numpy as np

from ai.amg_manifest import atomic_write_json, now_iso, read_json

SCHEMA_VERSION = 1
UNIFIED_REVIEW_NAME = "unified_review.json"

# 候補に対する操作 (AMG候補の適用状態)。ADD/REMOVE のみ。
ACTION_ADD = "add"
ACTION_REMOVE = "remove"
VALID_ACTIONS = frozenset({ACTION_ADD, ACTION_REMOVE})

__all__ = [
    "SCHEMA_VERSION",
    "UNIFIED_REVIEW_NAME",
    "ACTION_ADD",
    "ACTION_REMOVE",
    "VALID_ACTIONS",
    "compute_mask_sha256",
    "normalize_candidate_actions",
    "build_unified_review_state",
    "save_unified_review_state",
    "load_unified_review_state",
    "is_state_stale",
    "set_candidate_action",
]


def compute_mask_sha256(mask: np.ndarray) -> str:
    """マスク配列の安定した SHA-256 (dtype + shape + bytes)。"""
    arr = np.ascontiguousarray(np.asarray(mask))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(str(arr.shape).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def normalize_candidate_actions(
    actions: dict[str, str],
    valid_segment_ids=None,
) -> dict[str, str]:
    """
    candidate_actions を検証・正規化する (キーは str(int)、値は add/remove)。

    valid_segment_ids を渡すと、その不変 ID 一覧に無い候補は捨てる
    (REMOVE_ONLY の追加保存問題と同じく、可変なキーを唯一の有効一覧にしない)。
    不明な action は ValueError。
    """
    valid = None if valid_segment_ids is None else {str(int(s)) for s in valid_segment_ids}
    out: dict[str, str] = {}
    for key, value in (actions or {}).items():
        skey = str(int(key))
        if value not in VALID_ACTIONS:
            raise ValueError(f"segment {skey}: 不明な候補操作 {value!r}")
        if valid is not None and skey not in valid:
            continue
        out[skey] = value
    return out


def build_unified_review_state(
    *,
    image_key: str,
    segments_npz_sha256: str,
    mask_sha256: str,
    candidate_actions: Optional[dict[str, str]] = None,
    ui: Optional[dict[str, Any]] = None,
    completed: bool = False,
    valid_segment_ids=None,
) -> dict[str, Any]:
    """unified_review.json の構造を作る。"""
    return {
        "schema_version": SCHEMA_VERSION,
        "image_key": image_key,
        "segments_npz_sha256": segments_npz_sha256,
        "mask_sha256": mask_sha256,
        "completed": bool(completed),
        "candidate_actions": normalize_candidate_actions(
            candidate_actions or {}, valid_segment_ids),
        "ui": dict(ui or {}),
        "updated_at": now_iso(),
    }


def save_unified_review_state(path, state: dict[str, Any]) -> None:
    """unified_review.json を原子的に保存する (updated_at を更新)。"""
    data = dict(state)
    data["updated_at"] = now_iso()
    atomic_write_json(path, data)


def load_unified_review_state(path) -> Optional[dict[str, Any]]:
    """unified_review.json を読む。存在しない / 壊れている場合は None。"""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return None
    try:
        return read_json(p)
    except (ValueError, OSError):
        return None


def is_state_stale(
    state: Optional[dict[str, Any]],
    *,
    mask_sha256: str,
    segments_npz_sha256: Optional[str] = None,
) -> bool:
    """
    状態が現在のマスク / segments.npz と整合しないか判定する。

    True (stale): state 無し / schema 不一致 / マスク SHA 不一致 /
                  (segments_npz_sha256 指定時) NPZ SHA 不一致。
    マスクが正本なので、マスクが変わった (SHA 変化) 状態は信頼しない。
    """
    if not state:
        return True
    if int(state.get("schema_version", -1)) != SCHEMA_VERSION:
        return True
    if state.get("mask_sha256") != mask_sha256:
        return True
    if segments_npz_sha256 is not None and \
            state.get("segments_npz_sha256") != segments_npz_sha256:
        return True
    return False


def set_candidate_action(
    state: dict[str, Any],
    segment_id: int,
    action: Optional[str],
) -> dict[str, Any]:
    """
    候補 1 件の操作を更新した新しい state を返す (元を破壊しない)。

    action=None でその候補の操作を解除する。
    """
    if action is not None and action not in VALID_ACTIONS:
        raise ValueError(f"不明な候補操作: {action!r}")
    updated = dict(state)
    actions = dict(updated.get("candidate_actions", {}))
    key = str(int(segment_id))
    if action is None:
        actions.pop(key, None)
    else:
        actions[key] = action
    updated["candidate_actions"] = actions
    return updated
