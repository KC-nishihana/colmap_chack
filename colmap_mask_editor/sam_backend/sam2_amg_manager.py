"""
V0.8 Worker側: SAM 2.1 Automatic Mask Generator のラッパ。

既存 Sam2ModelManager が保持するモデルを再利用し、AMG 用にモデルを二重ロードしない。
torch / sam2 はこのモジュール内でだけ遅延 import する (GUI からは import されない)。

OOM 時は points_per_batch のみを半分にして 1 回だけ再試行する
(points_per_side / crop 設定 / 品質しきい値は自動変更しない)。CPU フォールバックしない。

generate() は SAM 2 公式の output_mode="uncompressed_rle" の annotations を返す
(segmentation = {"size":[h,w], "counts":[...]})。
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

_log = logging.getLogger("sam_worker.amg")

# AMG は 1 バッチで points_per_batch × 3(multimask) × H × W × 4byte 規模の
# マスク logits を確保する。高解像 (360°パノラマ等) では points_per_batch=64 が
# 1 枚で 24GB を超え、Windows では OOM にならず共有メモリへあふれて激遅になる。
# 解像度に応じて初期 points_per_batch を安全側へ自動調整する (結果は不変・メモリのみ削減)。
# 既定予算 ~2.0GB 相当 (env AMG_MASK_MEM_BUDGET_MB で上書き可)。
_DEFAULT_MASK_BUDGET_BYTES = 2_000 * 1024 * 1024


def _mask_budget_bytes() -> int:
    import os
    v = os.environ.get("AMG_MASK_MEM_BUDGET_MB")
    if v:
        try:
            return max(128, int(v)) * 1024 * 1024
        except ValueError:
            pass
    return _DEFAULT_MASK_BUDGET_BYTES


def auto_points_per_batch(requested: int, height: int, width: int,
                          multimask: bool = True) -> int:
    """
    画像解像度から安全な points_per_batch を求める (requested を超えない・最小1)。

    points_per_batch を下げてもマスク結果は同一 (推論のタイル化のみ)。VRAM だけ削減。
    """
    req = max(1, int(requested))
    per_point = (3 if multimask else 1) * int(height) * int(width) * 4  # float32 mask logits
    if per_point <= 0:
        return req
    safe = max(1, _mask_budget_bytes() // per_point)
    return int(min(req, safe))


class AmgOom(Exception):
    """points_per_batch 最小値でも OOM だった場合に送出する。"""


@dataclass
class AmgGenerateResult:
    annotations: list
    points_per_batch_used: int
    peak_vram_mb: int = 0
    oom_retries: int = 0
    warnings: list = field(default_factory=list)


def _is_oom(exc: Exception) -> bool:
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


class Sam2AmgManager:
    """共有モデルから AMG を構築し、画像 1 枚を generate する。"""

    def __init__(self, model_manager) -> None:
        self._mm = model_manager
        self._generator = None
        self._gen_key = None

    # ------------------------------------------------------------------ #
    # generator 構築 (バッチ内で 1 回だけ作り再利用する)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _settings_key(settings: dict[str, Any], points_per_batch: int):
        keys = ("points_per_side", "pred_iou_thresh", "stability_score_thresh",
                "box_nms_thresh", "crop_n_layers", "crop_n_points_downscale_factor",
                "min_mask_region_area", "use_m2m", "multimask_output")
        return tuple(settings.get(k) for k in keys) + (int(points_per_batch),)

    def _get_generator(self, settings: dict[str, Any], points_per_batch: int):
        """同一設定なら既存の generator を再利用する (毎画像の再構築を避ける)。"""
        key = self._settings_key(settings, points_per_batch)
        if self._generator is None or self._gen_key != key:
            # 旧 generator (内部 predictor の GPU 特徴) を解放してから作り直す
            self._generator = None
            self._reclaim()
            self._generator = self._build_generator(settings, points_per_batch)
            self._gen_key = key
        return self._generator

    def _build_generator(self, settings: dict[str, Any], points_per_batch: int):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        model = self._mm.model
        if model is None:
            raise RuntimeError("SAM 2 モデルがロードされていません")
        return SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=int(settings["points_per_side"]),
            points_per_batch=int(points_per_batch),
            pred_iou_thresh=float(settings["pred_iou_thresh"]),
            stability_score_thresh=float(settings["stability_score_thresh"]),
            box_nms_thresh=float(settings["box_nms_thresh"]),
            crop_n_layers=int(settings["crop_n_layers"]),
            crop_n_points_downscale_factor=int(settings["crop_n_points_downscale_factor"]),
            min_mask_region_area=int(settings["min_mask_region_area"]),
            output_mode="uncompressed_rle",
            use_m2m=bool(settings["use_m2m"]),
            multimask_output=bool(settings["multimask_output"]),
        )

    # ------------------------------------------------------------------ #
    # generate (OOM 再試行つき)
    # ------------------------------------------------------------------ #

    def generate(self, image_rgb, settings: dict[str, Any], oom_retry: bool = True) -> AmgGenerateResult:
        """
        image_rgb (H,W,3 uint8) を AMG 解析し annotations を返す。

        OOM 時は points_per_batch を半分にして再試行 (64->32->16->8...最小1)。
        各再試行前に一時データ破棄 + gc + empty_cache。最小でも失敗なら AmgOom。
        """
        import torch

        device_index = self._device_index()
        try:
            torch.cuda.reset_peak_memory_stats(device_index)
        except Exception:
            pass

        requested_ppb = int(settings["points_per_batch"])
        h, w = int(image_rgb.shape[0]), int(image_rgb.shape[1])
        ppb = auto_points_per_batch(requested_ppb, h, w,
                                    multimask=bool(settings.get("multimask_output", True)))
        retries = 0
        warnings: list = []
        if ppb < requested_ppb:
            msg = (f"高解像 {w}x{h} のため points_per_batch を "
                   f"{requested_ppb}->{ppb} に自動調整 (結果は不変・VRAM超過/共有メモリ流出を防止)")
            warnings.append(msg)
            _log.info(msg)
        while True:
            try:
                generator = self._get_generator(settings, ppb)
                with torch.inference_mode():
                    annotations = generator.generate(image_rgb)
                peak = self._peak_vram_mb(device_index)
                return AmgGenerateResult(
                    annotations=annotations,
                    points_per_batch_used=ppb,
                    peak_vram_mb=peak,
                    oom_retries=retries,
                    warnings=warnings,
                )
            except Exception as e:  # noqa: BLE001
                if not _is_oom(e):
                    raise
                self._reclaim()
                if not oom_retry or ppb <= 1:
                    raise AmgOom(
                        f"points_per_batch={ppb} でも CUDA OOM。画像をスキップします。"
                    ) from e
                new_ppb = max(1, ppb // 2)
                warnings.append(f"CUDA OOM: points_per_batch {ppb} -> {new_ppb} で再試行")
                _log.warning("AMG OOM: points_per_batch %d -> %d", ppb, new_ppb)
                ppb = new_ppb
                retries += 1

    # ------------------------------------------------------------------ #
    # メモリ
    # ------------------------------------------------------------------ #

    def _reclaim(self) -> None:
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def reclaim(self) -> None:
        """画像間 / OOM 後に呼ぶ VRAM 解放 (generator は再利用のため保持)。"""
        self._reclaim()

    def release(self) -> None:
        """ジョブ終了時に呼ぶ完全解放 (generator も破棄して GPU を返す)。"""
        self._generator = None
        self._gen_key = None
        self._reclaim()

    def _device_index(self) -> int:
        dev = getattr(self._mm, "device", None) or "cuda:0"
        try:
            import torch
            d = torch.device(dev)
            return d.index if d.index is not None else 0
        except Exception:
            return 0

    def _peak_vram_mb(self, device_index: int) -> int:
        try:
            import torch
            return int(torch.cuda.max_memory_allocated(device_index) / (1024 * 1024))
        except Exception:
            return 0
