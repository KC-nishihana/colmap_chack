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


def test_cuda_extension_kernel_executes():
    """get_connected_components() の CUDA カーネルを直接実行し面積を検証する。

    sam2._C の import だけでなく、実際にカーネルが走り正しい連結成分面積を
    返すことを確認する (NumPy の形状確認では代替しない)。
    """
    import torch
    from sam2.utils.misc import get_connected_components

    mask = torch.zeros((1, 1, 64, 64), dtype=torch.bool, device="cuda:0")
    mask[:, :, 5:15, 5:15] = True       # 10x10 = 100px
    mask[:, :, 30:45, 35:55] = True      # 15x20 = 300px

    labels, areas = get_connected_components(mask)
    torch.cuda.synchronize()

    assert labels.is_cuda
    assert areas.is_cuda
    assert labels.shape == mask.shape
    assert areas.shape == mask.shape
    assert labels.dtype in (torch.int32, torch.int64)
    assert int(labels.max().item()) >= 2

    component_areas = set(
        int(v) for v in torch.unique(areas[mask]).detach().cpu().tolist()
    )
    assert 100 in component_areas
    assert 300 in component_areas


def test_fill_holes_postprocess():
    """fill_holes_in_mask_scores() の CUDA 後処理を確認する (補助)。"""
    import torch
    from sam2.utils.misc import fill_holes_in_mask_scores

    mask_scores = torch.ones((1, 1, 64, 64), dtype=torch.float32, device="cuda:0")
    mask_scores[:, :, 28:32, 28:32] = -1.0   # 4x4=16px の穴

    filled = fill_holes_in_mask_scores(mask_scores, max_area=32)
    torch.cuda.synchronize()
    assert filled.is_cuda
    assert filled.shape == mask_scores.shape
    assert bool(torch.all(filled[:, :, 28:32, 28:32] > 0).item())


def test_embedding_and_predictions(loaded):
    mm, predictor = loaded
    rgb = (np.random.rand(1080, 1920, 3) * 255).astype(np.uint8)
    predictor.set_image(rgb, image_key="t")

    # 正クリック (multimask_output=True で 3 候補)
    masks, scores, _ = predictor.predict(
        points=[{"x": 960, "y": 540, "label": 1}], box=None, multimask_output=True
    )
    assert masks.shape[0] == 3
    assert len(scores) == masks.shape[0]
    assert masks.shape[1:] == (1080, 1920)
    assert set(np.unique(masks)).issubset({0, 255})

    # 負クリック
    m_neg, _s, _ = predictor.predict(
        points=[{"x": 960, "y": 540, "label": 1}, {"x": 10, "y": 10, "label": 0}],
        box=None, multimask_output=True,
    )
    assert m_neg.shape[0] == 3

    # 矩形: multimask_output=True は原則 3 候補を返す。
    # (モデル/プロンプト仕様上保証されないケースに備え 1..3 を最低条件とし、
    #  スコア数とマスク数の一致を必須にする)
    m_box, s_box, _ = predictor.predict(points=[], box=[200, 150, 1600, 900],
                                        multimask_output=True)
    assert m_box.shape[0] == 3
    assert len(s_box) == m_box.shape[0]
    assert 1 <= m_box.shape[0] <= 3
    assert mm.peak_vram_mb() > 0
