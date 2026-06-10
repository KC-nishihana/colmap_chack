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
    colmap_registered: bool = False    # images.txt に登録済みか


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

    # images.txt からCOLMAP登録済み画像名のセットを取得（フィルタではなくメタデータ用）
    colmap_name_set: set[str] = set()
    if images_txt and images_txt.exists():
        colmap_name_set = set(parse_images_txt(images_txt))

    # images/ フォルダ内の全画像をスキャン（COLMAPの登録有無に関わらず）
    # COLMAP登録済みの画像を先に、未登録を後に並べる
    all_imgs: list[Path] = sorted(
        (p for p in images_dir.rglob("*") if p.is_file() and p.suffix in IMAGE_EXTENSIONS)
    )

    # COLMAP登録順で先頭に並べ替え
    if colmap_name_set:
        registered: list[Path] = []
        unregistered: list[Path] = []
        for img_path in all_imgs:
            rel_name = img_path.relative_to(images_dir).as_posix()
            if rel_name in colmap_name_set or img_path.name in colmap_name_set:
                registered.append(img_path)
            else:
                unregistered.append(img_path)
        all_imgs = registered + unregistered

    for img_path in all_imgs:
        rel = img_path.relative_to(images_dir)
        rel_name = rel.as_posix()
        in_colmap = rel_name in colmap_name_set or img_path.name in colmap_name_set
        mask_path = _find_mask(masks_dir, rel) if masks_dir else None
        entry = ImageEntry(
            image_path=img_path,
            rel_path=rel,
            mask_path=mask_path,
            has_mask=mask_path is not None,
            colmap_registered=in_colmap,
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
