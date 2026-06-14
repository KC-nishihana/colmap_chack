"""
Worker 側の SAM 2.1 モデル管理。

torch / sam2 / sam2._C をこのモジュール内でだけ import する (遅延 import)。
モデルは load_model でロードし常駐させる。画像変更時もモデルは保持する。

CUDA 拡張 (sam2._C) が読み込めない場合は明示的にエラーとし、CPU や拡張なし処理へ
フォールバックしない (CLAUDE.md V0.6 要件)。
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Optional

_log = logging.getLogger("sam_worker.model")


@dataclass
class EnvCapabilities:
    torch_version: str = ""
    torch_cuda_version: Optional[str] = None
    torchvision_version: Optional[str] = None
    cuda_available: bool = False
    cuda_extension_loaded: bool = False
    gpu_name: Optional[str] = None
    compute_capability: Optional[str] = None
    torch_import_error: Optional[str] = None
    sam2_import_error: Optional[str] = None
    cuda_extension_error: Optional[str] = None


def probe_environment(device_index: int = 0) -> EnvCapabilities:
    """torch / sam2 / sam2._C / CUDA の可用性を調べる (副作用なし)。"""
    caps = EnvCapabilities()
    try:
        import torch  # noqa: WPS433 (遅延 import)
    except Exception as e:  # torch 未導入
        caps.torch_import_error = repr(e)
        return caps

    caps.torch_version = getattr(torch, "__version__", "")
    caps.torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    caps.cuda_available = bool(torch.cuda.is_available())

    try:
        import torchvision  # noqa: WPS433
        caps.torchvision_version = getattr(torchvision, "__version__", None)
    except Exception:
        caps.torchvision_version = None

    if caps.cuda_available:
        try:
            caps.gpu_name = torch.cuda.get_device_name(device_index)
            major, minor = torch.cuda.get_device_capability(device_index)
            caps.compute_capability = f"{major}.{minor}"
        except Exception as e:
            caps.cuda_extension_error = f"GPU情報取得失敗: {e!r}"

    # sam2 本体
    try:
        import sam2  # noqa: F401,WPS433
    except Exception as e:
        caps.sam2_import_error = repr(e)
        return caps

    # CUDA 拡張 (必須)
    try:
        import sam2._C  # noqa: F401,WPS433
        caps.cuda_extension_loaded = True
    except Exception as e:
        caps.cuda_extension_loaded = False
        caps.cuda_extension_error = repr(e)

    return caps


class Sam2ModelManager:
    """SAM 2.1 モデルのロード・保持・解放を担当する。"""

    def __init__(self) -> None:
        self._model = None
        self._predictor = None
        self._model_id: Optional[str] = None
        self._device: Optional[str] = None
        self._precision: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    @property
    def predictor(self):
        return self._predictor

    @property
    def model_id(self) -> Optional[str]:
        return self._model_id

    @property
    def device(self) -> Optional[str]:
        return self._device

    def _resolve_dtype(self, precision: str):
        import torch
        if precision == "bf16":
            if not torch.cuda.is_bf16_supported():
                raise PrecisionUnavailable("bf16 はこのGPUで利用できません")
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16
        if precision == "fp32":
            return torch.float32
        raise ValueError(f"不明な precision: {precision!r}")

    def load(
        self,
        model_id: str,
        config_name: str,
        checkpoint_path: str,
        precision: str = "bf16",
        device: str = "cuda:0",
    ) -> dict:
        """モデルをロードして常駐させる。VRAM割当量等を辞書で返す。"""
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if not torch.cuda.is_available():
            raise CudaUnavailable("CUDA が利用できません")

        # 拡張チェック (必須)
        try:
            import sam2._C  # noqa: F401
        except Exception as e:
            raise CudaExtensionUnavailable(f"sam2._C を import できません: {e!r}") from e

        self.unload()  # 既存を解放

        dtype = self._resolve_dtype(precision)
        dev = torch.device(device)
        dev_index = dev.index if dev.index is not None else 0

        torch.cuda.reset_peak_memory_stats(dev_index)

        _log.info("モデルロード開始: %s (%s, %s, %s)", model_id, config_name, precision, device)
        model = build_sam2(config_name, checkpoint_path, device=device)
        model.to(dtype=dtype)
        predictor = SAM2ImagePredictor(model)

        self._model = model
        self._predictor = predictor
        self._model_id = model_id
        self._device = device
        self._precision = precision

        vram_mb = int(torch.cuda.memory_allocated(dev_index) / (1024 * 1024))
        _log.info("モデルロード完了: %s, VRAM=%dMB", model_id, vram_mb)
        return {"vram_allocated_mb": vram_mb}

    def unload(self) -> None:
        if self._predictor is None and self._model is None:
            return
        _log.info("モデル解放: %s", self._model_id)
        self._predictor = None
        self._model = None
        self._model_id = None
        try:
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    def vram_allocated_mb(self) -> int:
        try:
            import torch
            if not torch.cuda.is_available():
                return 0
            return int(torch.cuda.memory_allocated() / (1024 * 1024))
        except Exception:
            return 0

    def peak_vram_mb(self) -> int:
        try:
            import torch
            if not torch.cuda.is_available():
                return 0
            return int(torch.cuda.max_memory_allocated() / (1024 * 1024))
        except Exception:
            return 0

    def clear_cuda_cache(self) -> None:
        try:
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


# ------------------------------------------------------------------ #
# 例外
# ------------------------------------------------------------------ #


class CudaUnavailable(Exception):
    pass


class CudaExtensionUnavailable(Exception):
    pass


class PrecisionUnavailable(Exception):
    pass
