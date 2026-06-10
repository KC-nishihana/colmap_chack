"""
COLMAP の images.txt を読み取るモジュール
"""

from pathlib import Path


def parse_images_txt(images_txt_path: Path) -> list[str]:
    """
    images.txt からCOLMAP登録済みの画像名一覧を返す。
    コメント行(#)と空行はスキップ。
    images.txt が存在しない場合は空リストを返す。
    """
    if not images_txt_path.exists():
        return []

    image_names: list[str] = []
    # COLMAPのimages.txtフォーマット:
    # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME  (奇数行)
    # POINTS2D[] (偶数行)
    try:
        text = images_txt_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        # コメント行のみ除外。空行(POINTS2D[]が空の場合)は除外しない
        data_lines = [l for l in lines if not l.startswith("#")]
        # 奇数番目の行(0-indexed: 0, 2, 4...)が画像情報行
        for i in range(0, len(data_lines), 2):
            parts = data_lines[i].split()
            if len(parts) >= 10:
                image_names.append(parts[9])  # NAMEフィールド
    except Exception as e:
        print(f"[WARN] images.txt 読み込みエラー: {e}")
        return []

    return image_names
