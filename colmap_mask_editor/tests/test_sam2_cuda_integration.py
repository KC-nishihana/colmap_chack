"""
RTX 4090 実機 CUDA 統合テスト (Phase 22)。

通常の pytest からは分離する。実行するには:
    $env:RUN_SAM2_CUDA_TESTS = "1"
    python -m pytest -m sam2_cuda -v

これらは本物の torch / sam2 / sam2._C / CUDA / チェックポイントを要求する。
チェックポイントは models/sam2/sam2.1_hiera_small.pt を想定 (環境変数で上書き可)。
"""

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.sam2_cuda

if os.environ.get("RUN_SAM2_CUDA_TESTS") != "1":
    pytest.skip(
        "実機CUDAテストは RUN_SAM2_CUDA_TESTS=1 のときのみ実行します",
        allow_module_level=True,
    )

PKG_ROOT = Path(__file__).resolve().parent.parent


def _checkpoint() -> Path:
    p = os.environ.get("SAM2_CHECKPOINT")
    if p:
        return Path(p)
    return PKG_ROOT.parent / "models" / "sam2" / "sam2.1_hiera_small.pt"


@pytest.fixture(scope="module")
def loaded():
    import torch
    assert torch.cuda.is_available(), "CUDA が利用できません"
    import sam2  # noqa: F401
    import sam2._C  # noqa: F401  CUDA拡張

    from ai import model_registry
    from sam_backend.sam2_model_manager import Sam2ModelManager
    from sam_backend.sam2_predictor import Sam2Predictor

    info = model_registry.get_model("sam2.1_hiera_small")
    ckpt = _checkpoint()
    assert ckpt.exists(), f"チェックポイントがありません: {ckpt}"

    mm = Sam2ModelManager()
    mm.load("sam2.1_hiera_small", info.config_name, str(ckpt), "bf16", "cuda:0")
    predictor = Sam2Predictor(mm)
    yield mm, predictor
    mm.unload()


def test_gpu_is_rtx4090():
    import torch
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    assert cc[0] >= 8  # Ada/Ampere 以降
    print("GPU:", name, "CC:", cc)


def test_cuda_extension_importable():
    import sam2._C  # noqa: F401


def test_embedding_and_predictions(loaded):
    import torch
    mm, predictor = loaded
    rgb = (np.random.rand(1080, 1920, 3) * 255).astype(np.uint8)
    predictor.set_image(rgb, image_key="t")

    # 正クリック
    masks, scores, _ = predictor.predict(
        points=[{"x": 960, "y": 540, "label": 1}], box=None, multimask_output=True
    )
    assert masks.shape[0] >= 1
    assert masks.shape[1:] == (1080, 1920)
    assert set(np.unique(masks)).issubset({0, 255})

    # 負クリック
    predictor.predict(
        points=[{"x": 960, "y": 540, "label": 1}, {"x": 10, "y": 10, "label": 0}],
        box=None, multimask_output=True,
    )
    # 矩形
    m_box, s_box, _ = predictor.predict(points=[], box=[200, 150, 1600, 900],
                                        multimask_output=True)
    assert m_box.shape[0] >= 1
    # 3候補・スコア取得
    assert len(s_box) >= 1
    assert mm.peak_vram_mb() > 0


def test_worker_restart_frees_vram(loaded):
    import torch
    mm, _ = loaded
    before = mm.vram_allocated_mb()
    assert before > 0
