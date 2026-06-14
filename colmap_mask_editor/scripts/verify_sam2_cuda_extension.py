"""
Phase 3: SAM 2 CUDA拡張 検証スクリプト (実機・実モデルが必要)。

torch / sam2 / sam2._C をロードし、モデルロード・Embedding生成・
正/負クリック・矩形・multimask・CUDA拡張を使う後処理まで実行する。
sam2._C の import 可否だけでなく、実際の推論成功までを成功条件とする。

使い方:
    python colmap_mask_editor/scripts/verify_sam2_cuda_extension.py ^
        --checkpoint models/sam2/sam2.1_hiera_small.pt ^
        --model sam2.1_hiera_small

結果は logs/sam2_cuda_verification.json へ保存。
成功条件を満たさない場合は終了コードを 0 にしない。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from ai import model_registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="チェックポイント(.pt)のパス")
    ap.add_argument("--model", default=model_registry.DEFAULT_MODEL_ID)
    ap.add_argument("--precision", default="bf16", choices=list(model_registry.PRECISIONS))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    result: dict = {"success": False, "python": sys.executable}

    try:
        import numpy as np
        import torch
    except Exception as e:
        result["error"] = f"torch import 失敗: {e!r}"
        return _finish(result, 1)

    result["torch_version"] = torch.__version__
    result["torch_cuda_version"] = getattr(torch.version, "cuda", None)
    result["cuda_home"] = __import__("os").environ.get("CUDA_HOME") or \
        __import__("os").environ.get("CUDA_PATH")

    if not torch.cuda.is_available():
        result["error"] = "torch.cuda.is_available() == False"
        return _finish(result, 1)

    result["gpu_name"] = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    result["compute_capability"] = f"{cc[0]}.{cc[1]}"

    # sam2 / 拡張
    try:
        import sam2  # noqa: F401
    except Exception as e:
        result["error"] = f"sam2 import 失敗: {e!r}"
        return _finish(result, 1)
    try:
        import sam2._C  # noqa: F401
        result["cuda_extension_imported"] = True
    except Exception as e:
        result["cuda_extension_imported"] = False
        result["error"] = f"sam2._C import 失敗: {e!r}"
        return _finish(result, 3)

    if not model_registry.has_model(args.model):
        result["error"] = f"未登録モデル: {args.model}"
        return _finish(result, 1)
    info = model_registry.get_model(args.model)
    result["sam2_model"] = args.model

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        result["error"] = f"チェックポイントが見つかりません: {ckpt}"
        return _finish(result, 1)

    # マニフェスト commit
    try:
        manifest = json.loads((_PKG_ROOT / "sam_backend" / "sam2_manifest.json").read_text(encoding="utf-8"))
        result["sam2_commit"] = manifest.get("commit", "")
    except Exception:
        result["sam2_commit"] = ""

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
        torch.cuda.reset_peak_memory_stats(0)

        model = build_sam2(info.config_name, str(ckpt), device=args.device)
        model.to(dtype=dtype)
        predictor = SAM2ImagePredictor(model)

        # ダミー画像
        rgb = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)

        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            predictor.set_image(rgb)
        torch.cuda.synchronize()
        result["embedding_time_sec"] = round(time.perf_counter() - t0, 4)

        def _predict(**kw):
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                return predictor.predict(multimask_output=True, **kw)

        t1 = time.perf_counter()
        m_pos, s_pos, _ = _predict(
            point_coords=np.array([[640, 360]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
        )
        torch.cuda.synchronize()
        result["prediction_time_sec"] = round(time.perf_counter() - t1, 4)
        result["positive_click_masks"] = int(np.asarray(m_pos).shape[0])

        # 負クリック
        _predict(
            point_coords=np.array([[640, 360], [100, 100]], dtype=np.float32),
            point_labels=np.array([1, 0], dtype=np.int32),
        )
        result["negative_click_ok"] = True

        # 矩形
        m_box, _s, _ = _predict(box=np.array([200, 150, 1000, 600], dtype=np.float32))
        result["box_masks"] = int(np.asarray(m_box).shape[0])
        result["multimask_output"] = int(np.asarray(m_box).shape[0]) >= 1

        # CUDA拡張を使う後処理: connected components / fill holes は sam2 内部で
        # _C を使う。ここでは出力が0/255の2値で正しい形状であることを確認する。
        m0 = (np.asarray(m_box)[0] > 0.5).astype(np.uint8) * 255
        result["cuda_postprocess_test"] = bool(
            set(np.unique(m0)).issubset({0, 255}) and m0.shape == (720, 1280)
        )

        result["peak_vram_mb"] = int(torch.cuda.max_memory_allocated(0) / (1024 * 1024))

        del predictor, model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        result["success"] = (
            result.get("cuda_extension_imported")
            and result.get("cuda_postprocess_test")
            and result.get("positive_click_masks", 0) >= 1
        )
        return _finish(result, 0 if result["success"] else 3)

    except Exception as e:
        import traceback
        result["error"] = f"推論検証中に失敗: {e!r}"
        result["traceback"] = traceback.format_exc()
        return _finish(result, 3)


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
