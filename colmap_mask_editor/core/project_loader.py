"""
プロジェクトフォルダを解析して画像一覧を構築するモジュール
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.colmap_images_txt import parse_images_txt

# 対応する画像拡張子
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


@dataclass
class ImageEntry:
    """1枚の画像に関するメタデータ"""
    image_path: Path          # 画像の絶対パス
    rel_path: Path            # images/ からの相対パス
    mask_path: Optional[Path] = None   # 元マスクの絶対パス(なければNone)
    has_mask: bool = False
    mask_size_mismatch: bool = False   # マスクとサイズ不一致
    is_modified: bool = False          # 未保存の編集あり
    check_result: Optional[Any] = None  # CheckResult (mask_checker.py)


@dataclass
class ProjectInfo:
    """プロジェクト全体の情報"""
    root: Path
    images_dir: Path
    masks_dir: Optional[Path]
    sparse_dir: Optional[Path]
    images_txt: Optional[Path]
    entries: list[ImageEntry] = field(default_factory=list)


def load_project(root: Path) -> ProjectInfo:
    """
    プロジェクトフォルダを読み込み、画像一覧を構築して返す。
    """
    images_dir = root / "images"
    masks_dir = root / "masks" if (root / "masks").exists() else None
    sparse_dir = root / "sparse" / "0" if (root / "sparse" / "0").exists() else None
    images_txt = sparse_dir / "images.txt" if sparse_dir else None

    info = ProjectInfo(
        root=root,
        images_dir=images_dir,
        masks_dir=masks_dir,
        sparse_dir=sparse_dir,
        images_txt=images_txt,
    )

    if not images_dir.exists():
        return info

    # images.txt から画像名取得を試みる
    colmap_names: list[str] = []
    if images_txt and images_txt.exists():
        colmap_names = parse_images_txt(images_txt)

    if colmap_names:
        # COLMAPの登録順を使用
        for name in colmap_names:
            img_path = images_dir / name
            if img_path.exists() and img_path.suffix.lower() in {e.lower() for e in IMAGE_EXTENSIONS}:
                rel = Path(name)
                mask_path = _find_mask(masks_dir, rel) if masks_dir else None
                entry = ImageEntry(
                    image_path=img_path,
                    rel_path=rel,
                    mask_path=mask_path,
                    has_mask=mask_path is not None,
                )
                info.entries.append(entry)
    else:
        # images/ を再帰検索
        for img_path in sorted(images_dir.rglob("*")):
            if img_path.is_file() and img_path.suffix in IMAGE_EXTENSIONS:
                rel = img_path.relative_to(images_dir)
                mask_path = _find_mask(masks_dir, rel) if masks_dir else None
                entry = ImageEntry(
                    image_path=img_path,
                    rel_path=rel,
                    mask_path=mask_path,
                    has_mask=mask_path is not None,
                )
                info.entries.append(entry)

    return info


def _find_mask(masks_dir: Path, rel: Path) -> Optional[Path]:
    """
    画像の相対パスに対応するマスクファイルを優先順位に従い探す。
    1. 画像ファイル名 + ".png"  例: IMG_0001.jpg.png
    2. 拡張子を .png に変えたもの 例: IMG_0001.png
    3. 画像と同じファイル名     例: IMG_0001.jpg
    """
    stem = rel.stem
    suffix = rel.suffix
    parent = rel.parent

    candidates = [
        masks_dir / parent / (rel.name + ".png"),      # 優先度1
        masks_dir / parent / (stem + ".png"),           # 優先度2
        masks_dir / parent / rel.name,                  # 優先度3
    ]

    for c in candidates:
        if c.exists():
            return c
    return None
