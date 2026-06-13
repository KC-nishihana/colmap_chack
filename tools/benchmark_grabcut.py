"""
4K・8K画像のGrabCut処理確認スクリプト (v0.5.1)

NumPyで合成画像を生成してGrabCutの処理時間・メモリ使用量を測定する。
実画像ファイルをリポジトリへ追加せずに大画像の動作を確認できる。

使用例:
    python tools/benchmark_grabcut.py
    python tools/benchmark_grabcut.py --iter 1 --no-downscale
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# リポジトリ内のモジュールを import できるようにパスを追加
_repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(_repo_root / "colmap_mask_editor"))

import numpy as np


# ------------------------------------------------------------------ #
# 合成画像生成
# ------------------------------------------------------------------ #

def make_synthetic_image(width: int, height: int) -> np.ndarray:
    """
    グラデーションと矩形を含む合成BGR画像を生成する。
    GrabCut の動作確認に適した背景/前景の境界を持つ。
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # 背景: 青系グラデーション
    for x in range(width):
        img[:, x, 0] = int(x / width * 100)  # B
        img[:, x, 1] = int(x / width * 80)   # G
        img[:, x, 2] = 20                      # R

    # 前景: 中央に明るい矩形
    cy, cx = height // 2, width // 2
    rh, rw = int(height * 0.4), int(width * 0.4)
    y0, y1 = cy - rh // 2, cy + rh // 2
    x0, x1 = cx - rw // 2, cx + rw // 2
    img[y0:y1, x0:x1] = (200, 200, 200)  # 明るいグレー

    # 前景内部にランダムテクスチャ
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 40, (y1 - y0, x1 - x0, 3), dtype=np.uint8)
    img[y0:y1, x0:x1] = np.clip(img[y0:y1, x0:x1].astype(np.int16) + noise - 20, 0, 255).astype(np.uint8)

    return img


# ------------------------------------------------------------------ #
# メモリ使用量取得
# ------------------------------------------------------------------ #

def _get_memory_mb() -> float | None:
    """現在のプロセスのメモリ使用量 (MB) を返す。psutil がなければ None。"""
    try:
        import psutil
        proc = psutil.Process()
        return proc.memory_info().rss / 1024 / 1024
    except ImportError:
        return None


# ------------------------------------------------------------------ #
# ベンチマーク実行
# ------------------------------------------------------------------ #

def run_benchmark(
    width: int,
    height: int,
    iter_count: int,
    use_downscale: bool,
    max_processing_size: int,
) -> None:
    """1解像度のGrabCutベンチマークを実行して結果を表示する。"""
    from core.grabcut_tool import GrabCutOptions, create_grabcut_session

    print(f"\n{'─' * 60}")
    print(f"  解像度: {width} x {height}")
    print(f"  反復回数: {iter_count}, 大画像縮小: {use_downscale}, 最大処理サイズ: {max_processing_size}")

    img = make_synthetic_image(width, height)

    # 矩形: 中央40%の領域
    cy, cx = height // 2, width // 2
    rh, rw = int(height * 0.4), int(width * 0.4)
    rect = (cx - rw // 2, cy - rh // 2, rw, rh)

    options = GrabCutOptions(
        iter_count=iter_count,
        use_downscale=use_downscale,
        max_processing_size=max_processing_size,
    )

    mem_before = _get_memory_mb()
    t0 = time.perf_counter()

    try:
        session, result = create_grabcut_session(img, rect, options)
    except Exception as e:
        print(f"  [エラー] {e}")
        return

    elapsed = time.perf_counter() - t0
    mem_after = _get_memory_mb()

    # 出力マスクの検証
    output_mask = result.mask
    unique_vals = np.unique(output_mask)
    invalid_vals = [v for v in unique_vals if v not in (0, 255)]

    print(f"  元画像サイズ:     {result.original_size[0]} x {result.original_size[1]}")
    print(f"  ROIサイズ:        {result.roi[2]} x {result.roi[3]}")
    print(f"  処理解像度:       {result.processing_size[0]} x {result.processing_size[1]}")
    print(f"  縮小率:           {result.scale:.4f}")
    print(f"  処理時間:         {elapsed:.3f} 秒")
    print(f"  出力マスクサイズ: {output_mask.shape[1]} x {output_mask.shape[0]}")
    print(f"  出力マスク dtype: {output_mask.dtype}")
    print(f"  0/255以外の値:    {'なし' if not invalid_vals else str(invalid_vals)}")

    if mem_before is not None and mem_after is not None:
        print(f"  メモリ増加:       {mem_after - mem_before:.1f} MB  (使用中: {mem_after:.1f} MB)")
    else:
        print(f"  メモリ使用量:     取得不可 (psutil をインストールすると表示されます)")

    assert output_mask.dtype == np.uint8, "出力マスクは uint8 でなければならない"
    assert output_mask.shape == (height, width), f"出力マスクサイズ不一致: {output_mask.shape}"
    if invalid_vals:
        print(f"  [警告] 0/255以外の値が含まれています: {invalid_vals}")

    print(f"  ✓ 正常完了")


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GrabCut 4K/8K 動作確認スクリプト",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iter", type=int, default=1,
                        help="GrabCut反復回数 (ベンチマーク用に1推奨)")
    parser.add_argument("--no-downscale", action="store_true",
                        help="大画像縮小を無効にする")
    parser.add_argument("--max-size", type=int, default=2048,
                        help="最大処理サイズ (px)")
    parser.add_argument("--only", choices=["fhd", "4k", "8k"],
                        help="特定解像度のみ実行する")
    args = parser.parse_args()

    use_downscale = not args.no_downscale

    sizes = {
        "fhd": (1920, 1080),
        "4k":  (3840, 2160),
        "8k":  (7680, 4320),
    }

    if args.only:
        targets = {args.only: sizes[args.only]}
    else:
        targets = sizes

    print("=" * 60)
    print("  GrabCut 大画像ベンチマーク")
    print(f"  反復回数={args.iter}, 縮小={use_downscale}, 最大サイズ={args.max_size}")
    print("=" * 60)

    for name, (w, h) in targets.items():
        run_benchmark(
            width=w,
            height=h,
            iter_count=args.iter,
            use_downscale=use_downscale,
            max_processing_size=args.max_size,
        )

    print("\n" + "=" * 60)
    print("  全ベンチマーク完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
