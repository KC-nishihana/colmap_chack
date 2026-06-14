"""
Phase 3 / V0.6.1: SAM 2 CUDA拡張 検証スクリプト (実機・実モデルが必要)。

sam2._C の import 可否「だけ」では成功にしない。以下をすべて実機で確認する:
  - sam2._C import
  - get_connected_components() による CUDA カーネルの直接実行
    (複数連結成分の面積を正しく取得できること)
  - fill_holes_in_mask_scores() の後処理 (補助確認)
  - SAM 2 モデルロード・画像Embedding・正/負クリック・矩形・multimask 推論

成功判定は ai.cuda_verification.evaluate_verification() に集約 (通常 pytest と共有)。

使い方:
    python colmap_mask_editor/scripts/verify_sam2_cuda_extension.py ^
        --checkpoint models/sam2/sam2.1_hiera_small.pt ^
        --model sam2.1_hiera_small

結果は logs/sam2_cuda_verification.json へ保存。

終了コード:
    0 = すべての検証に成功
    1 = 環境または入力不足
    2 = PyTorch CUDA と CUDA Toolkit の不整合
    3 = SAM 2 または CUDA 拡張のビルド・ロード・実行失敗
    4 = チェックポイント不足により実機検証未完了
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from ai import model_registry  # noqa: E402
from ai.cuda_verification import evaluate_verification  # noqa: E402


def _kernel_test(torch, result: dict, device: str) -> bool:
    """get_connected_components() で CUDA 拡張カーネルを直接実行し検証する。

    成功時 True。失敗時は result にエラーを記録し False を返す。
    """
    from sam2.utils.misc import get_connected_components

    # 2 つの連結成分 (100px と 300px) を持つマスクを CUDA 上に作る
    test_mask = torch.zeros((1, 1, 64, 64), dtype=torch.bool, device=device)
    test_mask[:, :, 5:15, 5:15] = True       # 10x10 = 100px
    test_mask[:, :, 30:45, 35:55] = True      # 15x20 = 300px

    torch.cuda.synchronize()
    t_ext = time.perf_counter()
    labels, areas = get_connected_components(test_mask)
    torch.cuda.synchronize()
    extension_elapsed = time.perf_counter() - t_ext

    assert labels.is_cuda, "labels が CUDA Tensor ではありません"
    assert areas.is_cuda, "areas が CUDA Tensor ではありません"
    assert labels.shape == test_mask.shape, f"labels.shape={tuple(labels.shape)}"
    assert areas.shape == test_mask.shape, f"areas.shape={tuple(areas.shape)}"
    assert labels.dtype in (torch.int32, torch.int64), f"labels.dtype={labels.dtype}"
    assert int(labels.max().item()) >= 2, "連結成分が 2 未満です"

    component_areas = sorted(
        int(v) for v in torch.unique(areas[test_mask]).detach().cpu().tolist()
    )
    assert 100 in component_areas, f"100px 成分が見つかりません: {component_areas}"
    assert 300 in component_areas, f"300px 成分が見つかりません: {component_areas}"

    # 連結成分数 = 前景ラベルの distinct 個数 (labels.max() はラベルIDで個数ではない)
    component_count = int(torch.unique(labels[labels > 0]).numel())
    assert component_count >= 2, f"連結成分数が 2 未満: {component_count}"

    result["cuda_extension_kernel_executed"] = True
    result["connected_components_count"] = component_count
    result["connected_component_areas"] = component_areas
    result["cuda_extension_time_sec"] = round(extension_elapsed, 6)
    return True


def _fill_holes_test(torch, result: dict, device: str) -> None:
    """fill_holes_in_mask_scores() の後処理を補助確認する。

    fill_holes は内部で例外を警告へ変換しうるため、成功判定の基準にはせず
    補助記録に留める (基準は _kernel_test の直接呼び出し)。
    """
    try:
        from sam2.utils.misc import fill_holes_in_mask_scores

        mask_scores = torch.ones((1, 1, 64, 64), dtype=torch.float32, device=device)
        mask_scores[:, :, 20:44, 20:44] = 1.0
        mask_scores[:, :, 28:32, 28:32] = -1.0   # 4x4=16px の穴 (<= max_area)

        filled = fill_holes_in_mask_scores(mask_scores, max_area=32)
        torch.cuda.synchronize()

        ok = bool(
            filled.is_cuda
            and tuple(filled.shape) == (1, 1, 64, 64)
            and bool(torch.all(filled[:, :, 28:32, 28:32] > 0).item())
        )
        result["fill_holes_test"] = ok
    except Exception as e:
        result["fill_holes_test"] = False
        result["fill_holes_error"] = repr(e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="チェックポイント(.pt)のパス")
    ap.add_argument("--model", default=model_registry.DEFAULT_MODEL_ID)
    ap.add_argument("--precision", default="bf16", choices=list(model_registry.PRECISIONS))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    result: dict = {"success": False, "python": sys.executable}

    # --- torch / CUDA ---
    try:
        import numpy as np
        import torch
    except Exception as e:
        result["error"] = f"torch import 失敗: {e!r}"
        return _finish(result, 1)

    result["torch_version"] = torch.__version__
    result["torch_cuda_version"] = getattr(torch.version, "cuda", None)
    result["cuda_home"] = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")

    if not torch.cuda.is_available():
        result["error"] = "torch.cuda.is_available() == False"
        return _finish(result, 1)

    result["gpu_name"] = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    result["compute_capability"] = f"{cc[0]}.{cc[1]}"

    # --- sam2 / 拡張 import ---
    try:
        import sam2  # noqa: F401
    except Exception as e:
        result["error"] = f"sam2 import 失敗: {e!r}"
        return _finish(result, 3)
    try:
        import sam2._C  # noqa: F401
        result["cuda_extension_imported"] = True
    except Exception as e:
        result["cuda_extension_imported"] = False
        result["error"] = f"sam2._C import 失敗: {e!r}"
        return _finish(result, 3)

    # --- CUDA 拡張カーネルの直接実行 (最重要) ---
    try:
        _kernel_test(torch, result, args.device)
    except Exception as e:
        import traceback
        result["cuda_extension_kernel_executed"] = False
        result["cuda_extension_error"] = repr(e)
        result["traceback"] = traceback.format_exc()
        return _finish(result, 3)

    # --- fill_holes 後処理 (補助確認) ---
    _fill_holes_test(torch, result, args.device)

    # --- マニフェスト commit (記録用) ---
    try:
        manifest = json.loads(
            (_PKG_ROOT / "sam_backend" / "sam2_manifest.json").read_text(encoding="utf-8")
        )
        result["sam2_commit"] = manifest.get("commit", "")
    except Exception:
        result["sam2_commit"] = ""

    # --- モデル / チェックポイント ---
    if not model_registry.has_model(args.model):
        result["error"] = f"未登録モデル: {args.model}"
        return _finish(result, 1)
    info = model_registry.get_model(args.model)
    result["sam2_model"] = args.model

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        result["error"] = f"チェックポイントが見つかりません: {ckpt}"
        return _finish(result, 4)

    # --- 実モデル推論 ---
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
        torch.cuda.reset_peak_memory_stats(0)

        model = build_sam2(info.config_name, str(ckpt), device=args.device)
        model.to(dtype=dtype)
        predictor = SAM2ImagePredictor(model)
        result["model_loaded"] = True

        rgb = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)

        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            predictor.set_image(rgb)
        torch.cuda.synchronize()
        result["embedding_time_sec"] = round(time.perf_counter() - t0, 4)
        result["embedding_ok"] = True

        def _predict(**kw):
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                return predictor.predict(multimask_output=True, **kw)

        # 正クリック
        t1 = time.perf_counter()
        m_pos, s_pos, _ = _predict(
            point_coords=np.array([[640, 360]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
        )
        torch.cuda.synchronize()
        result["prediction_time_sec"] = round(time.perf_counter() - t1, 4)
        result["positive_click_masks"] = int(np.asarray(m_pos).shape[0])
        result["positive_click_ok"] = result["positive_click_masks"] >= 1

        # 負クリック
        m_neg, _s, _ = _predict(
            point_coords=np.array([[640, 360], [100, 100]], dtype=np.float32),
            point_labels=np.array([1, 0], dtype=np.int32),
        )
        result["negative_click_ok"] = int(np.asarray(m_neg).shape[0]) >= 1

        # 矩形
        m_box, s_box, _ = _predict(box=np.array([200, 150, 1000, 600], dtype=np.float32))
        n_box = int(np.asarray(m_box).shape[0])
        result["box_masks"] = n_box
        result["box_prompt_ok"] = n_box >= 1
        # multimask_output=True で複数候補が返ること
        result["multimask_output"] = n_box >= 2
        result["candidate_count"] = n_box

        # 出力が 0/255 の 2 値で元画像サイズであること
        m0 = (np.asarray(m_box)[0] > 0.5).astype(np.uint8) * 255
        result["mask_binary_ok"] = bool(
            set(np.unique(m0)).issubset({0, 255}) and m0.shape == (720, 1280)
        )

        result["peak_vram_mb"] = int(torch.cuda.max_memory_allocated(0) / (1024 * 1024))

        del predictor, model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        result["error"] = f"推論検証中に失敗: {e!r}"
        result["traceback"] = traceback.format_exc()
        return _finish(result, 3)

    result["success"] = evaluate_verification(result)
    return _finish(result, 0 if result["success"] else 3)


def _finish(result: dict, code: int) -> int:
    repo_root = _PKG_ROOT.parent
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    out = logs_dir / "sam2_cuda_verification.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n保存: {out}  (exit={code})")
    return code


if __name__ == "__main__":
    sys.exit(main())
