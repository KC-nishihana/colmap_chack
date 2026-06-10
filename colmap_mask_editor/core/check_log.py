"""
チェックログCSV出力: mask_check_log.csv を生成する
"""

import csv
from pathlib import Path

from core.mask_io import get_colmap_mask_path

_FIELDNAMES = [
    "image_path",
    "input_mask_path",
    "edited_mask_path",
    "colmap_mask_path",
    "status",
    "width",
    "height",
    "mask_width",
    "mask_height",
    "mask_ratio",
    "has_intermediate_values",
    "is_empty_mask",
    "is_full_mask",
    "note",
]


def export_check_log(project) -> Path:
    """
    全画像のチェック結果を CSV に出力する。
    出力先: project.root / mask_check_log.csv
    utf-8-sig (BOM付き) で書き込み、Windows Excel でも文字化けしない。
    """
    log_path = project.root / "mask_check_log.csv"
    rows = []

    for entry in project.entries:
        colmap_path = get_colmap_mask_path(project.root, entry.rel_path)

        def to_rel(p: Path) -> str:
            if p is None:
                return ""
            try:
                return str(p.relative_to(project.root))
            except ValueError:
                return str(p)

        cr = entry.check_result
        if cr is not None:
            row = {
                "image_path":              to_rel(entry.image_path),
                "input_mask_path":         to_rel(entry.mask_path) if entry.mask_path else "",
                "edited_mask_path":        "",
                "colmap_mask_path":        to_rel(colmap_path) if colmap_path.exists() else "",
                "status":                  cr.status,
                "width":                   cr.image_width,
                "height":                  cr.image_height,
                "mask_width":              cr.mask_width,
                "mask_height":             cr.mask_height,
                "mask_ratio":              f"{cr.mask_ratio:.4f}",
                "has_intermediate_values": cr.has_intermediate_values,
                "is_empty_mask":           cr.is_empty_mask,
                "is_full_mask":            cr.is_full_mask,
                "note":                    cr.note,
            }
        else:
            row = {
                "image_path":              to_rel(entry.image_path),
                "input_mask_path":         to_rel(entry.mask_path) if entry.mask_path else "",
                "edited_mask_path":        "",
                "colmap_mask_path":        to_rel(colmap_path) if colmap_path.exists() else "",
                "status":                  "needs_check",
                "width":                   "",
                "height":                  "",
                "mask_width":              "",
                "mask_height":             "",
                "mask_ratio":              "",
                "has_intermediate_values": "",
                "is_empty_mask":           "",
                "is_full_mask":            "",
                "note":                    "チェック未実行",
            }
        rows.append(row)

    try:
        with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        print(f"[ERROR] CSV出力エラー: {e}")

    return log_path
