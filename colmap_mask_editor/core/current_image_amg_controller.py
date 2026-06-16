"""
V0.11: 現在画像の AMG キャッシュ読込と候補プロバイダ (動線2 のデータ基盤・非GUI)。

既存プリミティブを束ねるだけの薄いオーケストレーション層:
  - ai.amg_cache.evaluate_cache       : キャッシュ状態 (missing/stale/corrupt/reusable)
  - core.amg_review_index_worker      : review_index (グループ/代表/親子/確認順) の構築・再利用
  - ai.amg_hit_test.MaskDecodeCache   : 候補マスクの遅延復号 LRU (4K/8K で全 dense を持たない)
  - ai.amg_hit_test.candidates_at_point : クリック位置の候補
  - ai.amg_remove_only.is_covered     : REMOVE 和集合に包含された候補の抑制

このコントローラは現在候補 / ホバー候補 / 適用済み和集合だけを復号する。全候補の
dense マスクを同時に保持しない。GUI へは候補 index と (要求時に) マスクだけを渡す。

torch / sam2 / PySide6 非依存。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ai import amg_cache, amg_hit_test, amg_manifest, amg_npz, amg_remove_only

# UI 状態 (動線2 の 4 分岐へ対応)
STATE_MISSING = "missing"     # 未解析 -> 「この画像を自動分割」
STATE_READY = "ready"         # 有効キャッシュ -> 候補を即時表示
STATE_STALE = "stale"         # 再解析を案内
STATE_CORRUPT = "corrupt"     # 破損 -> エラー表示
STATE_ERROR = "error"

# amg_cache の状態 -> UI 状態
_CACHE_STATE_MAP = {
    amg_cache.MISSING: STATE_MISSING,
    amg_cache.REUSABLE: STATE_READY,
    amg_cache.STALE: STATE_STALE,
    amg_cache.CORRUPT: STATE_CORRUPT,
}

__all__ = [
    "STATE_MISSING", "STATE_READY", "STATE_STALE", "STATE_CORRUPT", "STATE_ERROR",
    "AmgCacheStatus", "CurrentImageAmgController",
]


@dataclass(frozen=True)
class AmgCacheStatus:
    state: str
    reason: str = ""
    total_candidates: int = 0


class CurrentImageAmgController:
    """現在画像 1 枚分の AMG 候補を読み込み、表示用に問い合わせる。"""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.85,
        containment_threshold: float = 0.95,
        covered_threshold: float = amg_remove_only.DEFAULT_COVERED_THRESHOLD,
        decode_cache_size: int = 12,
    ) -> None:
        self._iou = float(iou_threshold)
        self._cont = float(containment_threshold)
        self._covered_threshold = float(covered_threshold)
        self._decode_cache_size = int(decode_cache_size)

        self._cache_dir: Optional[Path] = None
        self._npz: Optional[dict[str, np.ndarray]] = None
        self._index: Optional[dict[str, np.ndarray]] = None
        self._decode: Optional[amg_hit_test.MaskDecodeCache] = None
        self._id_to_index: dict[int, int] = {}
        self._segment_ids: list[int] = []

    # ------------------------------------------------------------------ #
    # キャッシュ状態
    # ------------------------------------------------------------------ #

    @staticmethod
    def evaluate(
        cache_dir,
        *,
        source_path: str,
        model: dict[str, Any],
        generator: dict[str, Any],
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> AmgCacheStatus:
        """現在画像のキャッシュ状態を UI 状態へマップして返す。"""
        check = amg_cache.evaluate_cache(
            cache_dir, source_path=source_path, model=model, generator=generator,
            width=width, height=height)
        state = _CACHE_STATE_MAP.get(check.state, STATE_ERROR)
        total = 0
        if check.manifest is not None:
            total = int(check.manifest.get("segment_count", 0))
        return AmgCacheStatus(state=state, reason=check.reason, total_candidates=total)

    @staticmethod
    def status_for(cache_dir) -> str:
        """設定無しの簡易判定 (manifest / NPZ の存在と整合のみ)。"""
        cdir = Path(cache_dir)
        npz_path = cdir / amg_manifest.SEGMENTS_NPZ_NAME
        manifest_path = cdir / amg_manifest.MANIFEST_NAME
        if not npz_path.exists() or not manifest_path.exists():
            return STATE_MISSING
        try:
            amg_npz.verify_segments_npz(npz_path)
        except Exception:  # noqa: BLE001
            return STATE_CORRUPT
        return STATE_READY

    # ------------------------------------------------------------------ #
    # 読込 / 破棄
    # ------------------------------------------------------------------ #

    def load(self, cache_dir, *, cancel_check: Optional[Callable[[], bool]] = None) -> int:
        """
        segments.npz と review_index を読み込み、候補プロバイダを準備する。

        review_index は有効なら再利用、stale なら再計算する (CPU)。候補総数を返す。
        マスクは遅延復号 (ここでは dense を作らない)。
        """
        from core.amg_review_index_worker import ensure_review_index

        cdir = Path(cache_dir)
        npz_path = cdir / amg_manifest.SEGMENTS_NPZ_NAME
        self._npz = amg_npz.load_segments_npz(npz_path)
        result = ensure_review_index(
            cdir, iou_threshold=self._iou, containment_threshold=self._cont,
            cancel_check=cancel_check)
        self._index = result.arrays
        self._cache_dir = cdir
        self._decode = amg_hit_test.MaskDecodeCache(self._npz, max_size=self._decode_cache_size)
        self._segment_ids = [int(s) for s in np.asarray(self._npz["segment_ids"]).tolist()]
        self._id_to_index = {sid: i for i, sid in enumerate(self._segment_ids)}
        return len(self._segment_ids)

    def unload(self) -> None:
        """画像切替時など。保持を解放しオーバーレイ状態を消す。"""
        if self._decode is not None:
            self._decode.clear()
        self._cache_dir = None
        self._npz = None
        self._index = None
        self._decode = None
        self._id_to_index = {}
        self._segment_ids = []

    @property
    def is_loaded(self) -> bool:
        return self._npz is not None and self._index is not None

    # ------------------------------------------------------------------ #
    # 候補メタ情報
    # ------------------------------------------------------------------ #

    @property
    def total_candidates(self) -> int:
        return len(self._segment_ids)

    @property
    def segment_ids(self) -> list[int]:
        return list(self._segment_ids)

    def index_of(self, segment_id: int) -> Optional[int]:
        return self._id_to_index.get(int(segment_id))

    def id_of(self, index: int) -> int:
        return self._segment_ids[int(index)]

    def _require(self) -> None:
        if not self.is_loaded:
            raise RuntimeError("AMG キャッシュが読み込まれていません")

    def representative_indices(self) -> set[int]:
        """代表候補 (重複グループの代表) の index 集合。"""
        self._require()
        reps = np.asarray(self._index["representative_segment_ids"])
        sids = np.asarray(self._npz["segment_ids"])
        return {int(i) for i in range(len(sids)) if int(reps[i]) == int(sids[i])}

    def parent_of(self, segment_id: int) -> Optional[int]:
        """入れ子の親候補 segment_id (タイヤ→ホイール等)。親が無ければ None。"""
        self._require()
        idx = self.index_of(segment_id)
        if idx is None:
            return None
        parent = int(np.asarray(self._index["parent_segment_ids"])[idx])
        return parent if parent >= 0 else None

    def representative_count(self) -> int:
        return len(self.representative_indices())

    # ------------------------------------------------------------------ #
    # マスク (遅延復号)
    # ------------------------------------------------------------------ #

    def candidate_mask(self, index: int) -> np.ndarray:
        """候補 1 件の (H,W) uint8(0/255) マスクを遅延復号して返す (LRU)。"""
        self._require()
        return self._decode.get(int(index))

    def union_mask(self, indices) -> np.ndarray:
        """指定 index 群の和集合 (bool, H,W)。適用済み REMOVE/ADD 和集合用。"""
        self._require()
        return self._decode.union([int(i) for i in indices])

    def candidates_at(self, x: int, y: int) -> list[int]:
        """クリック位置 (x,y) の候補 index を面積昇順で返す。"""
        self._require()
        return amg_hit_test.candidates_at_point(self._npz, x, y)

    # ------------------------------------------------------------------ #
    # 表示候補の絞り込み / 並べ替え
    # ------------------------------------------------------------------ #

    def visible_indices(
        self,
        *,
        representatives_only: bool = True,
        hide_covered: bool = True,
        removed_indices=None,
        sort_mode: str = "priority",
    ) -> list[int]:
        """
        表示すべき候補 index を絞り込み・並べ替えて返す。

        representatives_only: 代表候補だけ表示
        hide_covered: removed_indices の REMOVE 和集合へ包含された候補を隠す
        sort_mode: priority / area / quality / edge (降順)
        """
        self._require()
        n = self.total_candidates
        if representatives_only:
            idxs = sorted(self.representative_indices())
        else:
            idxs = list(range(n))

        if hide_covered and removed_indices:
            removed = {int(i) for i in removed_indices}
            union = self.union_mask(removed)
            kept = []
            for i in idxs:
                if i in removed:
                    kept.append(i)
                    continue
                if amg_remove_only.is_covered(
                        self.candidate_mask(i) > 0, union, self._covered_threshold):
                    continue
                kept.append(i)
            idxs = kept

        return self._sort_indices(idxs, sort_mode)

    def _sort_indices(self, idxs: list[int], sort_mode: str) -> list[int]:
        if not idxs:
            return idxs
        if sort_mode == "area":
            key_arr = np.asarray(self._npz["area"])
        elif sort_mode == "quality":
            key_arr = np.asarray(self._index["quality_scores"])
        elif sort_mode == "edge":
            key_arr = np.asarray(self._index["edge_touch_flags"])
        else:  # priority (確認順)
            key_arr = np.asarray(self._index["priority_scores"])
        # 降順、同値は segment index 昇順で決定的に
        return sorted(idxs, key=lambda i: (-float(key_arr[i]), i))
