"""
V0.7: フレームステージングと結果PNGの原子的書き込み (cv2/numpy のみ・torch非依存)。

SAM 2 Video Predictor は連番JPEG (<int>.jpg) のフォルダを要求するため、対象画像を
000000.jpg 形式へ連番化してステージングする。元画像は変更しない。

EXIF Orientation は GUI 側 (core/mask_io.imread_jp / sam_backend.image_loader) と同じく
適用しない (両者で同一のピクセル配置・幅高さになるようにするため)。
RGB/BGR の扱いも既存単一画像推論と統一 (cv2 BGR で読み、cv2 で JPEG エンコード)。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from sam_backend.image_loader import ImageLoadError, load_image_bgr


class StagingError(Exception):
    pass


def _atomic_write_bytes(dest: Path, data: bytes) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, dest)


def stage_sequence(
    entries: list[dict[str, Any]],
    frames_dir,
    *,
    reference_frame_index: int,
    jpeg_quality: int = 95,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> dict[str, Any]:
    """
    entries: [{frame_index, entry_key, source_path}] を frame_index 昇順・連続 (0..N-1) で渡す。
    各画像を frames_dir/<frame_index:06d>.jpg へ連番ステージングする。

    すべて同一サイズである前提 (preflight で検証済み)。ステージング前後で幅高さが
    変化していないことを確認する。frame_manifest 用 dict を返す。
    """
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    n = len(entries)
    ref_w: Optional[int] = None
    ref_h: Optional[int] = None
    manifest_frames: list[dict[str, Any]] = []

    for i, e in enumerate(entries):
        fidx = int(e["frame_index"])
        if fidx != i:
            raise StagingError(f"frame_index は 0..N-1 の連番である必要があります: {fidx} != {i}")
        src = e["source_path"]
        try:
            bgr = load_image_bgr(src)
        except ImageLoadError as ex:
            raise StagingError(f"画像を読み込めません: {src}: {ex}") from ex

        h, w = bgr.shape[:2]
        if ref_w is None:
            ref_w, ref_h = w, h
        elif (w, h) != (ref_w, ref_h):
            raise StagingError(
                f"ステージング中に異なるサイズを検出: {e['entry_key']} {w}x{h} != {ref_w}x{ref_h}"
            )

        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        if not ok:
            raise StagingError(f"JPEGエンコードに失敗: {src}")

        staged = frames_dir / f"{fidx:06d}.jpg"
        _atomic_write_bytes(staged, buf.tobytes())

        # ステージング後の幅高さが変化していないことを確認
        chk = cv2.imdecode(np.fromfile(str(staged), dtype=np.uint8), cv2.IMREAD_COLOR)
        if chk is None or chk.shape[1] != w or chk.shape[0] != h:
            raise StagingError(f"ステージング後にサイズが変化しました: {staged}")

        manifest_frames.append({
            "frame_index": fidx,
            "entry_key": e["entry_key"],
            "source_path": str(src),
            "staged_path": str(staged),
        })
        if progress_cb is not None:
            progress_cb(i + 1, n)

    if ref_w is None:
        raise StagingError("ステージング対象が空です")

    return {
        "reference_frame_index": int(reference_frame_index),
        "width": int(ref_w),
        "height": int(ref_h),
        "frames": manifest_frames,
    }


def write_mask_png_atomic(dest, mask: np.ndarray) -> None:
    """uint8 0/255 (H,W) マスクを単チャンネルPNGとして原子的に書く。"""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"mask は2次元 (H,W) が必要: shape={mask.shape}")
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise ValueError("PNGエンコードに失敗")
    _atomic_write_bytes(dest, buf.tobytes())


def read_mask_png(path) -> np.ndarray:
    """JP対応でマスクPNGを読み uint8 0/255 (H,W) を返す。"""
    arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise StagingError(f"マスクPNGを読み込めません: {path}")
    return (arr > 0).astype(np.uint8) * 255
