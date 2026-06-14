"""
V0.7: 伝播対象画像の順序決定と範囲選択 (純粋ロジック・torch非依存)。

画像順序を暗黙に決めない。COLMAP images.txt 順を撮影時系列と決めつけない。
ユーザーが選んだ PropagationOrder に従って明示的に並べ、重複を除去する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class PropagationOrder(Enum):
    CURRENT_LIST = auto()      # 現在の一覧順
    COLMAP_PRIORITY = auto()   # COLMAP images.txt 優先順
    FILE_NAME = auto()         # ファイル名順 (自然順)
    CAPTURE_TIME = auto()      # 撮影日時順 (EXIF DateTimeOriginal)


@dataclass(frozen=True)
class SourceImage:
    """並べ替えに必要な1画像の最小メタデータ。"""
    entry_key: str                     # プロジェクト内の一意キー (相対パス等)
    source_path: str
    list_index: int                    # 現在の一覧での位置
    file_name: str
    colmap_index: Optional[int] = None # images.txt 内の出現順 (無ければ None)
    capture_time: Optional[float] = None  # EXIF 撮影時刻 (epoch秒, 無ければ None)


_NUM_RE = re.compile(r"(\d+)")


def _natural_key(name: str):
    """ファイル名の自然順キー (IMG_2 < IMG_10)。"""
    parts = _NUM_RE.split(name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def order_images(images: list[SourceImage], mode: PropagationOrder) -> list[SourceImage]:
    """指定モードで安定ソートし、entry_key の重複を除去 (先頭優先) して返す。"""
    items = list(images)

    if mode == PropagationOrder.CURRENT_LIST:
        ordered = sorted(items, key=lambda im: im.list_index)
    elif mode == PropagationOrder.FILE_NAME:
        ordered = sorted(items, key=lambda im: _natural_key(im.file_name))
    elif mode == PropagationOrder.COLMAP_PRIORITY:
        # colmap_index がある画像を優先 (昇順)、無い画像は list_index で後ろへ。
        ordered = sorted(
            items,
            key=lambda im: (im.colmap_index is None,
                            im.colmap_index if im.colmap_index is not None else 0,
                            im.list_index),
        )
    elif mode == PropagationOrder.CAPTURE_TIME:
        # capture_time が無い画像は末尾へ (None last)、安定のため list_index を副キー。
        ordered = sorted(
            items,
            key=lambda im: (im.capture_time is None,
                            im.capture_time if im.capture_time is not None else 0.0,
                            im.list_index),
        )
    else:
        raise ValueError(f"未知の順序モード: {mode!r}")

    return _dedup(ordered)


def _dedup(images: list[SourceImage]) -> list[SourceImage]:
    seen: set[str] = set()
    out: list[SourceImage] = []
    for im in images:
        if im.entry_key in seen:
            continue
        seen.add(im.entry_key)
        out.append(im)
    return out


def select_range(
    ordered: list[SourceImage],
    reference_entry_key: str,
    direction: str,
    count: int,
) -> tuple[list[SourceImage], int]:
    """
    基準画像を中心に direction / count 枚を選ぶ。
    戻り: (対象画像リスト(順序保持), そのリスト内での基準のindex)。
    direction は propagation_protocol.PropagationDirection の値。
    基準画像は必ず含む。範囲は端でクランプする。
    """
    from ai.propagation_protocol import PropagationDirection

    if count < 0:
        raise ValueError("count は 0 以上")
    ref_pos = _find_ref(ordered, reference_entry_key)

    if direction == PropagationDirection.FORWARD:
        lo, hi = ref_pos, ref_pos + count
    elif direction == PropagationDirection.BACKWARD:
        lo, hi = ref_pos - count, ref_pos
    elif direction == PropagationDirection.BOTH:
        lo, hi = ref_pos - count, ref_pos + count
    else:
        raise ValueError(f"未知の方向: {direction!r}")

    lo = max(0, lo)
    hi = min(len(ordered) - 1, hi)
    frames = ordered[lo:hi + 1]
    new_ref = ref_pos - lo
    return frames, new_ref


def select_explicit(
    ordered: list[SourceImage],
    selected_entry_keys: list[str],
    reference_entry_key: str,
) -> tuple[list[SourceImage], int]:
    """一覧で選択された範囲 (entry_key集合) を ordered の順序で抽出する。

    基準画像が選択に含まれていなければ追加する。重複は除去。
    """
    wanted = set(selected_entry_keys)
    wanted.add(reference_entry_key)
    frames = [im for im in ordered if im.entry_key in wanted]
    frames = _dedup(frames)
    new_ref = _find_ref(frames, reference_entry_key)
    return frames, new_ref


def _find_ref(images: list[SourceImage], reference_entry_key: str) -> int:
    for i, im in enumerate(images):
        if im.entry_key == reference_entry_key:
            return i
    raise ValueError(f"基準画像が対象に含まれていません: {reference_entry_key!r}")


def format_order_preview(frames: list[SourceImage], reference_index: int) -> str:
    """順序確認用のプレビュー文字列を作る (UI/ログ共用)。"""
    lines = []
    for i, im in enumerate(frames):
        marker = "  ← 基準画像" if i == reference_index else ""
        lines.append(f"{i}  {im.file_name}{marker}")
    return "\n".join(lines)
