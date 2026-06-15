"""
V0.9: V0.8 segments.npz を統合ヒントへ変換する SAM シグネチャ生成 (numpy のみ)。

SAM マスクを最終 partition へ直接コピーしない。各葉リージョンについて最大
sample_count 点を決定的にサンプリングし、bbox が交差する SAM 候補のみを
rle_contains_point で評価して coverage_ratio を求める。guidance_score 上位
top_k 件を保存する。SAM 候補が無くても処理は続行する (空シグネチャ)。

  guidance_score = coverage_ratio * predicted_iou * stability_score
"""

from __future__ import annotations

import numpy as np

__all__ = ["SamSignatures", "compute_sam_signatures", "empty_signatures"]


class SamSignatures:
    """葉ごとの SAM シグネチャ (CSR 風 offsets + 連結配列)。"""

    __slots__ = ("leaf_count", "offsets", "segment_ids", "coverages", "scores")

    def __init__(self, leaf_count, offsets, segment_ids, coverages, scores):
        self.leaf_count = int(leaf_count)
        self.offsets = np.asarray(offsets, dtype=np.uint64)
        self.segment_ids = np.asarray(segment_ids, dtype=np.uint32)
        self.coverages = np.asarray(coverages, dtype=np.float32)
        self.scores = np.asarray(scores, dtype=np.float32)

    def for_leaf(self, leaf_id: int):
        """leaf_id (1..K) の (segment_ids, coverages, scores) を返す。"""
        i = int(leaf_id) - 1
        s = int(self.offsets[i])
        e = int(self.offsets[i + 1])
        return self.segment_ids[s:e], self.coverages[s:e], self.scores[s:e]


def empty_signatures(leaf_count: int) -> SamSignatures:
    """SAM 候補が無い場合の空シグネチャ。"""
    offsets = np.zeros(int(leaf_count) + 1, dtype=np.uint64)
    return SamSignatures(leaf_count, offsets,
                         np.zeros(0, dtype=np.uint32),
                         np.zeros(0, dtype=np.float32),
                         np.zeros(0, dtype=np.float32))


def _leaf_sample_points(labels: np.ndarray, leaf_count: int, sample_count: int):
    """
    各葉 (1..K) の代表サンプル点 (x, y) を最大 sample_count 個、決定的に返す。

    flatten を label でグルーピング (argsort) し、各葉から等間隔に抽出する。
    返り値: dict leaf_id -> (xs ndarray, ys ndarray)。
    """
    h, w = labels.shape
    flat = labels.reshape(-1).astype(np.int64)
    order = np.argsort(flat, kind="stable")
    sorted_lab = flat[order]
    # 各 label の連続区間境界
    boundaries = np.searchsorted(sorted_lab, np.arange(1, leaf_count + 2))
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for lid in range(1, leaf_count + 1):
        s = int(boundaries[lid - 1])
        e = int(boundaries[lid])
        n = e - s
        if n <= 0:
            out[lid] = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
            continue
        if n <= sample_count:
            sel = order[s:e]
        else:
            idx = np.linspace(0, n - 1, sample_count).astype(np.int64)
            sel = order[s + idx]
        ys, xs = np.divmod(sel, w)
        out[lid] = (xs.astype(np.int64), ys.astype(np.int64))
    return out


def _counts_contains(cum: np.ndarray, h: int, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    SAM uncompressed RLE (Fortran order) の累積和 cum に対し、点群 (xs, ys) が
    前景かを判定する bool 配列を返す。flat_index = y + x * h。
    """
    target = ys + xs * h
    run_index = np.searchsorted(cum, target, side="right")
    return (run_index % 2) == 1


def compute_sam_signatures(
    labels: np.ndarray,
    segments_data: dict | None,
    *,
    sample_count: int = 64,
    top_k: int = 4,
) -> SamSignatures:
    """
    葉ラベル (元解像度, 1..K) と segments.npz データから SAM シグネチャを作る。

    segments_data は amg_npz.load_segments_npz の結果 (None や空なら空シグネチャ)。
    labels と segments の座標系 (画像サイズ) は一致している前提。
    """
    arr = np.asarray(labels)
    leaf_count = int(arr.max())
    if segments_data is None:
        return empty_signatures(leaf_count)

    seg_ids = np.asarray(segments_data["segment_ids"]).astype(np.int64)
    n_seg = int(seg_ids.shape[0])
    if n_seg == 0:
        return empty_signatures(leaf_count)

    img_shape = np.asarray(segments_data["image_shape"])
    h, w = int(img_shape[0]), int(img_shape[1])
    seg_bbox = np.asarray(segments_data["bbox_xywh"]).astype(np.int64)
    pred_iou = np.asarray(segments_data["predicted_iou"]).astype(np.float64)
    stab = np.asarray(segments_data["stability_score"]).astype(np.float64)
    offsets_npz = np.asarray(segments_data["rle_offsets"]).astype(np.int64)
    counts_npz = np.asarray(segments_data["rle_counts"]).astype(np.int64)

    from partition_backend import base_partition as bp
    leaf_bbox = bp.region_bboxes(arr, leaf_count)
    samples = _leaf_sample_points(arr, leaf_count, sample_count)

    # 各候補の Fortran cumsum を遅延計算してキャッシュ
    cum_cache: dict[int, np.ndarray] = {}

    def seg_cum(i: int) -> np.ndarray:
        if i not in cum_cache:
            s = int(offsets_npz[i]); e = int(offsets_npz[i + 1])
            cum_cache[i] = np.cumsum(counts_npz[s:e])
        return cum_cache[i]

    seg_offsets = [0]
    out_ids: list[int] = []
    out_cov: list[float] = []
    out_score: list[float] = []

    for lid in range(1, leaf_count + 1):
        xs, ys = samples[lid]
        if xs.size == 0:
            seg_offsets.append(len(out_ids))
            continue
        lx, ly, lw, lh = (int(v) for v in leaf_bbox[lid])
        lx2, ly2 = lx + lw, ly + lh
        # bbox 交差する候補のみ
        sx, sy, sw, sh = (seg_bbox[:, 0], seg_bbox[:, 1],
                          seg_bbox[:, 2], seg_bbox[:, 3])
        inter = (sx < lx2) & (sx + sw > lx) & (sy < ly2) & (sy + sh > ly)
        cand = np.flatnonzero(inter)
        scored: list[tuple[float, float, int]] = []  # (guidance, coverage, seg_id)
        for i in cand.tolist():
            cum = seg_cum(i)
            inside = _counts_contains(cum, h, xs, ys)
            cov = float(np.count_nonzero(inside)) / float(xs.size)
            if cov <= 0.0:
                continue
            guidance = cov * float(pred_iou[i]) * float(stab[i])
            scored.append((guidance, cov, int(seg_ids[i])))
        # guidance 降順 -> coverage 降順 -> segment_id 昇順 (決定的)
        scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
        for guidance, cov, sid in scored[:top_k]:
            out_ids.append(sid)
            out_cov.append(cov)
            out_score.append(guidance)
        seg_offsets.append(len(out_ids))

    return SamSignatures(
        leaf_count,
        np.asarray(seg_offsets, dtype=np.uint64),
        np.asarray(out_ids, dtype=np.uint32),
        np.asarray(out_cov, dtype=np.float32),
        np.asarray(out_score, dtype=np.float32),
    )
