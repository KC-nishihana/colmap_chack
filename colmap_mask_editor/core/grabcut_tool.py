"""
GrabCutによる半自動マスク生成ツール (v0.4A.1)
ROIベース縮小処理・入力検証・ログ対応
"""

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

_log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# データクラス
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class GrabCutOptions:
    iter_count: int = 5
    use_downscale: bool = True
    max_processing_size: int = 2048
    roi_margin_ratio: float = 0.10
    min_margin_px: int = 32


@dataclass
class GrabCutResult:
    mask: np.ndarray
    original_size: tuple[int, int]        # (width, height)
    roi: tuple[int, int, int, int]        # (x, y, w, h)
    processing_size: tuple[int, int]      # (width, height)
    scale: float
    processing_time_sec: float
    was_downscaled: bool


# ------------------------------------------------------------------ #
# 公開関数
# ------------------------------------------------------------------ #

def run_grabcut_optimized(
    image_bgr: np.ndarray,
    rect: tuple[int, int, int, int],
    options: GrabCutOptions,
) -> GrabCutResult:
    """
    ROI切り出しと必要に応じた縮小を行い、
    元画像サイズの0/255マスクを返す。

    result.mask は以下を満たす:
      - shape == image_bgr.shape[:2]
      - dtype == np.uint8
      - unique values in {0, 255}
    """
    t_start = time.perf_counter()

    img = _validate_and_convert_image(image_bgr)
    ih, iw = img.shape[:2]

    validated_rect = _validate_rect(rect, iw, ih)
    opts = _validate_options(options)

    _log.info("GrabCut開始: 元画像 %dx%d, 矩形 %s, 反復 %d",
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
        roi_proc = roi_img

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

    result_mask = np.zeros((ih, iw), dtype=np.uint8)
    result_mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = fg_roi

    t_elapsed = time.perf_counter() - t_start
    _log.info(
        "GrabCut完了: 縮小率 %.4f, 処理解像度 %dx%d, 処理時間 %.3f秒",
        scale, proc_w, proc_h, t_elapsed,
    )

    return GrabCutResult(
        mask=result_mask,
        original_size=(iw, ih),
        roi=(roi_x, roi_y, roi_w, roi_h),
        processing_size=(proc_w, proc_h),
        scale=scale,
        processing_time_sec=t_elapsed,
        was_downscaled=was_downscaled,
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
    current_mask: np.ndarray,  # noqa: ARG001 (将来拡張用)
    rect: tuple[int, int, int, int],
    iter_count: int = 5,
) -> np.ndarray:
    """将来拡張用。現在は run_grabcut_with_rect に委譲する。"""
    return run_grabcut_with_rect(image_bgr, rect, iter_count)


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
