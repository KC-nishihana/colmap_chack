"""
GrabCutによる半自動マスク生成ツール (v0.4B)
ROIベース縮小処理・入力検証・ログ対応
v0.4B追加: GrabCutSession・HintStroke・再推定機能
"""

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import cv2
import numpy as np

_log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# データクラス / Enum
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class GrabCutOptions:
    iter_count: int = 5
    use_downscale: bool = True
    max_processing_size: int = 2048
    roi_margin_ratio: float = 0.10
    min_margin_px: int = 32
    use_existing_mask_as_bgd: bool = False  # 既存除外領域を背景制約として使用


@dataclass
class GrabCutResult:
    mask: np.ndarray
    original_size: tuple[int, int]        # (width, height)
    roi: tuple[int, int, int, int]        # (x, y, w, h)
    processing_size: tuple[int, int]      # (width, height)
    scale: float
    processing_time_sec: float
    was_downscaled: bool


class GrabCutHintLabel(IntEnum):
    """GrabCutヒントの種別。cv2のGC_*定数と対応。"""
    FOREGROUND = cv2.GC_FGD   # 確実な前景 = 1
    BACKGROUND = cv2.GC_BGD   # 確実な背景 = 0


@dataclass
class HintStroke:
    """
    ユーザーが描いた1ストローク分のヒント。
    座標は元画像座標 (original image coordinates) で保持する。
    label=None はヒント消去 (base_label_maskへ戻す)。
    """
    label: Optional[GrabCutHintLabel]  # None = 消去
    points: list[tuple[int, int]]
    radius: int


@dataclass
class GrabCutSession:
    """
    初回GrabCutと再推定に必要な全状態を保持する。
    再推定はこのセッションを使ってGC_INIT_WITH_MASKで実行する。
    """
    original_size: tuple[int, int]           # (width, height) 元画像サイズ
    original_rect: tuple[int, int, int, int] # (x, y, w, h) 元画像座標での矩形

    roi: tuple[int, int, int, int]           # (x, y, w, h) 元画像座標でのROI
    processing_size: tuple[int, int]         # (width, height) roi_image_bgrのサイズ
    scale: float                             # 縮小率 (1.0=縮小なし)
    was_downscaled: bool

    roi_image_bgr: np.ndarray                # 処理解像度のROI画像

    base_label_mask: np.ndarray              # 初回GrabCut直後ラベル (不変, ヒント消去の復元元)
    label_mask: np.ndarray                   # 現在のGrabCut内部ラベル (ヒント反映済み)
    bgd_model: np.ndarray                    # GrabCut内部の背景モデル
    fgd_model: np.ndarray                    # GrabCut内部の前景モデル

    preview_mask: np.ndarray                 # 0/255 uint8 元画像サイズ

    processing_time_sec: float = 0.0
    refine_count: int = 0


# ------------------------------------------------------------------ #
# 公開関数
# ------------------------------------------------------------------ #

def create_grabcut_session(
    image_bgr: np.ndarray,
    rect: tuple[int, int, int, int],
    options: GrabCutOptions,
    current_mask: Optional[np.ndarray] = None,
) -> GrabCutSession:
    """
    ROI切り出し・縮小・初回GrabCutを行い、
    再推定可能なGrabCutSessionを返す。

    current_mask が指定され options.use_existing_mask_as_bgd=True の場合、
    current_mask==0 の領域をGC_BGD制約として使用する。

    GrabCutSession の制約:
      - preview_mask.shape == image_bgr.shape[:2]
      - preview_mask.dtype == np.uint8
      - preview_mask の値は {0, 255} のみ
      - base_label_mask/label_mask の値は {0,1,2,3} のみ (GC_BGD,GC_FGD,GC_PR_BGD,GC_PR_FGD)
    """
    t_start = time.perf_counter()

    img = _validate_and_convert_image(image_bgr)
    ih, iw = img.shape[:2]

    validated_rect = _validate_rect(rect, iw, ih)
    opts = _validate_options(options)

    _log.info("GrabCutSession生成開始: 元画像 %dx%d, 矩形 %s, 反復 %d",
              iw, ih, validated_rect, opts.iter_count)

    roi_x, roi_y, roi_w, roi_h = _compute_roi(validated_rect, iw, ih, opts)
    _log.info("ROI: x=%d y=%d w=%d h=%d", roi_x, roi_y, roi_w, roi_h)

    roi_img = img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w].copy()

    scale = 1.0
    was_downscaled = False
    if opts.use_downscale:
        scale = min(1.0, opts.max_processing_size / max(roi_w, roi_h))
        was_downscaled = scale < 1.0

    if was_downscaled:
        proc_w = max(1, int(roi_w * scale))
        proc_h = max(1, int(roi_h * scale))
        _log.info("縮小: %dx%d → %dx%d (scale=%.4f)", roi_w, roi_h, proc_w, proc_h, scale)
        roi_proc = cv2.resize(roi_img, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    else:
        proc_w, proc_h = roi_w, roi_h
        roi_proc = roi_img.copy()

    rx, ry, rw, rh = validated_rect
    gc_x = max(0, int((rx - roi_x) * scale))
    gc_y = max(0, int((ry - roi_y) * scale))
    gc_w = max(1, int(rw * scale))
    gc_h = max(1, int(rh * scale))
    gc_x = min(gc_x, proc_w - 1)
    gc_y = min(gc_y, proc_h - 1)
    gc_w = min(gc_w, proc_w - gc_x)
    gc_h = min(gc_h, proc_h - gc_y)

    if gc_w < 2 or gc_h < 2:
        raise ValueError(
            f"縮小後の矩形が小さすぎます (w={gc_w}, h={gc_h})。"
            "最大処理サイズを大きくするか、矩形を広めに指定してください。"
        )

    _log.info("GrabCut実行: 処理解像度 %dx%d, 矩形 (%d,%d,%d,%d), 反復 %d",
              proc_w, proc_h, gc_x, gc_y, gc_w, gc_h, opts.iter_count)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    gc_mask = np.zeros((proc_h, proc_w), np.uint8)

    cv2.grabCut(
        roi_proc, gc_mask, (gc_x, gc_y, gc_w, gc_h),
        bgd_model, fgd_model,
        opts.iter_count, cv2.GC_INIT_WITH_RECT,
    )

    # 既存マスクを背景制約として使用 (options.use_existing_mask_as_bgd=True 時)
    if opts.use_existing_mask_as_bgd and current_mask is not None:
        roi_cm = current_mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        if was_downscaled:
            roi_cm_proc = cv2.resize(roi_cm, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
        else:
            roi_cm_proc = roi_cm
        # current_mask==0 (除外領域) を GC_BGD に設定
        gc_mask[roi_cm_proc == 0] = cv2.GC_BGD

    base_label_mask = gc_mask.copy()

    # ラベル値検証
    unique_vals = set(np.unique(gc_mask).tolist())
    valid_labels = {int(cv2.GC_BGD), int(cv2.GC_FGD), int(cv2.GC_PR_BGD), int(cv2.GC_PR_FGD)}
    if not unique_vals.issubset(valid_labels):
        _log.warning("GrabCutラベルに不正な値があります: %s", unique_vals - valid_labels)

    # プレビューマスク生成
    fg_small = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        np.uint8(255), np.uint8(0),
    )

    fg_count = int(np.sum(fg_small == 255))
    fg_ratio = fg_count / (proc_w * proc_h)
    if fg_count == 0:
        raise ValueError(
            "GrabCutに前景候補が見つかりませんでした。別の範囲を指定してください。"
        )
    if fg_ratio < 0.0001:
        _log.warning("前景率が極端に小さい: %.4f%%", fg_ratio * 100)
    elif fg_ratio > 0.99:
        _log.warning("前景率が極端に大きい: %.1f%%", fg_ratio * 100)

    if was_downscaled:
        fg_roi = cv2.resize(fg_small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    else:
        fg_roi = fg_small

    preview_mask = np.zeros((ih, iw), dtype=np.uint8)
    preview_mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = fg_roi

    t_elapsed = time.perf_counter() - t_start
    _log.info(
        "GrabCutSession生成完了: 縮小率 %.4f, 処理解像度 %dx%d, 前景率 %.2f%%, 処理時間 %.3f秒",
        scale, proc_w, proc_h, fg_ratio * 100, t_elapsed,
    )

    return GrabCutSession(
        original_size=(iw, ih),
        original_rect=validated_rect,
        roi=(roi_x, roi_y, roi_w, roi_h),
        processing_size=(proc_w, proc_h),
        scale=scale,
        was_downscaled=was_downscaled,
        roi_image_bgr=roi_proc,
        base_label_mask=base_label_mask,
        label_mask=gc_mask,
        bgd_model=bgd_model,
        fgd_model=fgd_model,
        preview_mask=preview_mask,
        processing_time_sec=t_elapsed,
        refine_count=0,
    )


def run_grabcut_optimized(
    image_bgr: np.ndarray,
    rect: tuple[int, int, int, int],
    options: GrabCutOptions,
) -> GrabCutResult:
    """
    ROI切り出しと必要に応じた縮小を行い、
    元画像サイズの0/255マスクを返す。(後方互換性維持)

    result.mask は以下を満たす:
      - shape == image_bgr.shape[:2]
      - dtype == np.uint8
      - unique values in {0, 255}
    """
    session = create_grabcut_session(image_bgr, rect, options)
    return GrabCutResult(
        mask=session.preview_mask,
        original_size=session.original_size,
        roi=session.roi,
        processing_size=session.processing_size,
        scale=session.scale,
        processing_time_sec=session.processing_time_sec,
        was_downscaled=session.was_downscaled,
    )


def run_grabcut_with_rect(
    image_bgr: np.ndarray,
    rect: tuple[int, int, int, int],
    iter_count: int = 5,
) -> np.ndarray:
    """
    矩形指定でGrabCutを実行し、0/255マスクを返す。
    後方互換性維持。内部で run_grabcut_optimized を呼ぶ (縮小なし)。
    """
    options = GrabCutOptions(iter_count=iter_count, use_downscale=False)
    result = run_grabcut_optimized(image_bgr, rect, options)
    return result.mask


def run_grabcut_with_existing_mask(
    image_bgr: np.ndarray,
    current_mask: np.ndarray,
    rect: tuple[int, int, int, int],
    iter_count: int = 5,
) -> np.ndarray:
    """既存マスクの除外領域を背景制約として使用してGrabCutを実行する。"""
    options = GrabCutOptions(
        iter_count=iter_count,
        use_downscale=True,
        use_existing_mask_as_bgd=True,
    )
    session = create_grabcut_session(image_bgr, rect, options, current_mask)
    return session.preview_mask


def apply_hint_strokes(
    session: GrabCutSession,
    strokes: list,  # list[HintStroke]
) -> np.ndarray:
    """
    base_label_maskをコピーし、全ヒントストロークを処理座標へ変換して反映した
    GrabCut内部ラベルマスクを返す。入力Sessionは破壊しない。

    処理順序:
      base_label_maskをコピー → 古いストロークから順に描画
      対象ヒント: GC_FGD / 背景ヒント: GC_BGD / 消去: base_label_maskへ戻す
    """
    label_mask = session.base_label_mask.copy()
    roi_x, roi_y, roi_w, roi_h = session.roi
    scale = session.scale
    proc_h, proc_w = label_mask.shape

    fgd_px = 0
    bgd_px = 0

    for stroke in strokes:
        if not stroke.points:
            continue

        # 元画像座標 → 処理座標へ変換 (ROI外の点は無視)
        proc_pts: list[tuple[int, int]] = []
        for (ox, oy) in stroke.points:
            if ox < roi_x or ox >= roi_x + roi_w or oy < roi_y or oy >= roi_y + roi_h:
                continue
            px = int(round((ox - roi_x) * scale))
            py = int(round((oy - roi_y) * scale))
            px = max(0, min(px, proc_w - 1))
            py = max(0, min(py, proc_h - 1))
            proc_pts.append((px, py))

        if not proc_pts:
            continue

        proc_radius = max(1, int(round(stroke.radius * scale)))

        if stroke.label is None:
            # 消去: base_label_maskの値へ戻す
            temp = np.zeros((proc_h, proc_w), dtype=np.uint8)
            _draw_stroke_on_mask(temp, proc_pts, proc_radius, 1)
            label_mask[temp == 1] = session.base_label_mask[temp == 1]
        else:
            label_val = int(stroke.label)  # GC_FGD(1) or GC_BGD(0)
            _draw_stroke_on_mask(label_mask, proc_pts, proc_radius, label_val)
            if stroke.label == GrabCutHintLabel.FOREGROUND:
                fgd_px += len(proc_pts)
            else:
                bgd_px += len(proc_pts)

    _log.debug("apply_hint_strokes: ストローク数=%d, 前景ヒント~%dpx, 背景ヒント~%dpx",
               len(strokes), fgd_px, bgd_px)
    return label_mask


def refine_grabcut_session(
    session: GrabCutSession,
    strokes: list,  # list[HintStroke]
    iter_count: int = 2,
) -> GrabCutSession:
    """
    ヒントを反映し、GC_INIT_WITH_MASKでGrabCutを再実行する。
    新しいGrabCutSessionを返し、入力Sessionは破壊しない。
    """
    t_start = time.perf_counter()

    proc_h, proc_w = session.base_label_mask.shape
    iw, ih = session.original_size
    roi_x, roi_y, roi_w, roi_h = session.roi

    # Sessionの整合性検証
    if session.roi_image_bgr.shape[:2] != (proc_h, proc_w):
        raise ValueError(
            f"Sessionの画像サイズとラベルマスクサイズが一致しません: "
            f"画像={session.roi_image_bgr.shape[:2]}, ラベル=({proc_h}, {proc_w})"
        )

    # ヒントストロークを処理座標へ反映
    label_mask = apply_hint_strokes(session, strokes)

    # ラベル値検証
    unique_vals = set(np.unique(label_mask).tolist())
    valid_labels = {int(cv2.GC_BGD), int(cv2.GC_FGD), int(cv2.GC_PR_BGD), int(cv2.GC_PR_FGD)}
    if not unique_vals.issubset(valid_labels):
        raise ValueError(
            f"GrabCut内部ラベルに無効な値があります: {unique_vals - valid_labels}"
        )

    # ヒントピクセル数の確認
    fgd_count = int(np.sum(label_mask == cv2.GC_FGD))
    bgd_count = int(np.sum(label_mask == cv2.GC_BGD))

    _log.info(
        "再推定開始: 前景ヒント %dpx, 背景ヒント %dpx, ストローク数 %d, 再推定 %d回目",
        fgd_count, bgd_count, len(strokes), session.refine_count + 1,
    )

    if fgd_count == 0:
        _log.warning("前景ヒントがありません")
    if bgd_count == 0:
        _log.warning("背景ヒントがありません")

    # モデルのコピー (入力Sessionを破壊しない)
    bgd_model = session.bgd_model.copy()
    fgd_model = session.fgd_model.copy()

    # GC_INIT_WITH_MASK で再推定 (rect=None)
    cv2.grabCut(
        session.roi_image_bgr,
        label_mask,
        None,
        bgd_model,
        fgd_model,
        iter_count,
        cv2.GC_INIT_WITH_MASK,
    )

    # プレビューマスク生成
    fg_small = np.where(
        (label_mask == cv2.GC_FGD) | (label_mask == cv2.GC_PR_FGD),
        np.uint8(255), np.uint8(0),
    )

    fg_count = int(np.sum(fg_small == 255))
    fg_ratio = fg_count / (proc_w * proc_h)

    if fg_count == 0:
        raise ValueError("再推定後の前景候補が0ピクセルです。ヒントを見直してください。")
    if fg_ratio < 0.0001:
        _log.warning("再推定後の前景率が極端に小さい: %.4f%%", fg_ratio * 100)
    elif fg_ratio > 0.99:
        _log.warning("再推定後の前景率が極端に大きい: %.1f%%", fg_ratio * 100)

    if session.was_downscaled:
        fg_roi = cv2.resize(fg_small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    else:
        fg_roi = fg_small

    preview_mask = np.zeros((ih, iw), dtype=np.uint8)
    preview_mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = fg_roi

    t_elapsed = time.perf_counter() - t_start
    prev_fg_ratio = float(np.sum(session.preview_mask == 255)) / (iw * ih)
    _log.info(
        "再推定完了: 前景率 %.2f%% → %.2f%%, 処理時間 %.3f秒",
        prev_fg_ratio * 100, fg_ratio * 100, t_elapsed,
    )

    return GrabCutSession(
        original_size=session.original_size,
        original_rect=session.original_rect,
        roi=session.roi,
        processing_size=session.processing_size,
        scale=session.scale,
        was_downscaled=session.was_downscaled,
        roi_image_bgr=session.roi_image_bgr,   # 同じ画像を再利用
        base_label_mask=session.base_label_mask,  # base は変更しない
        label_mask=label_mask,
        bgd_model=bgd_model,
        fgd_model=fgd_model,
        preview_mask=preview_mask,
        processing_time_sec=t_elapsed,
        refine_count=session.refine_count + 1,
    )


def apply_grabcut_result(
    current_mask: np.ndarray,
    grabcut_mask: np.ndarray,
    mode: str,
) -> np.ndarray:
    """
    GrabCut結果を現在マスクへ合成する。

    mode:
      add     -> grabcut_mask が255の領域を255にする
      remove  -> grabcut_mask が255の領域を0にする
      replace -> grabcut_mask そのものに置換する
    """
    result = current_mask.copy()
    gc_region = grabcut_mask == 255

    if mode == "add":
        result[gc_region] = 255
    elif mode == "remove":
        result[gc_region] = 0
    elif mode == "replace":
        result[:] = grabcut_mask
    else:
        raise ValueError(f"不明なmode: {mode!r}")

    return result


# ------------------------------------------------------------------ #
# 内部ヘルパー (テストから直接参照可)
# ------------------------------------------------------------------ #

def _draw_stroke_on_mask(
    mask: np.ndarray,
    pts: list[tuple[int, int]],
    radius: int,
    value: int,
) -> None:
    """処理座標の点列をマスクへ描画する (円+線補間, 高速移動でも途切れない)。"""
    for i, (px, py) in enumerate(pts):
        cv2.circle(mask, (px, py), radius, value, -1)
        if i > 0:
            cv2.line(mask, pts[i - 1], (px, py), value, radius * 2)


def _validate_and_convert_image(image: np.ndarray) -> np.ndarray:
    """画像の検証とBGR uint8への変換。"""
    if image is None:
        raise ValueError("画像がNoneです")
    if not isinstance(image, np.ndarray):
        raise ValueError("画像はnp.ndarrayである必要があります")
    if image.size == 0:
        raise ValueError("画像が空です")

    if image.ndim == 2:
        h, w = image.shape
        if h < 2 or w < 2:
            raise ValueError(f"画像が小さすぎます: {w}x{h}")
        img = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3:
        h, w, c = image.shape
        if h < 2 or w < 2:
            raise ValueError(f"画像が小さすぎます: {w}x{h}")
        if c == 3:
            img = image.copy()
        elif c == 4:
            img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError(f"対応していないチャンネル数: {c} (対応: 1/3/4)")
    else:
        raise ValueError(f"対応していない配列形状: {image.shape}")

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def _validate_rect(
    rect: tuple,
    img_w: int,
    img_h: int,
    min_w: int = 5,
    min_h: int = 5,
) -> tuple[int, int, int, int]:
    """矩形の検証・クリップ。(xc, yc, wc, hc) を返す。"""
    if len(rect) != 4:
        raise ValueError(f"矩形は (x, y, w, h) の4要素が必要です: {rect}")

    x, y, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])

    if w <= 0:
        raise ValueError(f"矩形の幅が正数でありません: w={w}")
    if h <= 0:
        raise ValueError(f"矩形の高さが正数でありません: h={h}")

    x2 = min(x + w, img_w)
    y2 = min(y + h, img_h)
    xc = max(0, x)
    yc = max(0, y)
    wc = x2 - xc
    hc = y2 - yc

    if wc <= 0 or hc <= 0:
        raise ValueError("矩形が画像と交差していません。画像上に矩形を指定してください。")

    if wc < min_w or hc < min_h:
        raise ValueError(
            f"矩形が小さすぎます (w={wc}, h={hc})。"
            "対象物の周囲を広めに指定してください。"
        )

    if (wc / img_w) > 0.95 and (hc / img_h) > 0.95:
        raise ValueError(
            "矩形が画像全体に近すぎるため、背景を推定できません。"
            "対象物の周囲に背景が入るように指定してください。"
        )

    return (xc, yc, wc, hc)


def _validate_options(options: GrabCutOptions) -> GrabCutOptions:
    """オプション値域の補正。"""
    return GrabCutOptions(
        iter_count=max(1, min(20, options.iter_count)),
        use_downscale=options.use_downscale,
        max_processing_size=max(256, min(8192, options.max_processing_size)),
        roi_margin_ratio=max(0.0, min(1.0, options.roi_margin_ratio)),
        min_margin_px=max(0, options.min_margin_px),
        use_existing_mask_as_bgd=options.use_existing_mask_as_bgd,
    )


def _compute_roi(
    rect: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    options: GrabCutOptions,
) -> tuple[int, int, int, int]:
    """矩形の周囲に余白を加えたROIを計算する。"""
    rx, ry, rw, rh = rect
    margin_x = max(options.min_margin_px, int(rw * options.roi_margin_ratio))
    margin_y = max(options.min_margin_px, int(rh * options.roi_margin_ratio))

    roi_x = max(0, rx - margin_x)
    roi_y = max(0, ry - margin_y)
    roi_x2 = min(img_w, rx + rw + margin_x)
    roi_y2 = min(img_h, ry + rh + margin_y)

    roi_w = roi_x2 - roi_x
    roi_h = roi_y2 - roi_y

    if roi_w < 4 or roi_h < 4:
        raise ValueError("ROIの計算に失敗しました。矩形が画像外の可能性があります。")

    return (roi_x, roi_y, roi_w, roi_h)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    """後方互換のためのヘルパー"""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()
