"""
V0.7 Worker側: SAM 2.1 Video Predictor のライフサイクル管理。

torch / sam2 はこのモジュール内でのみ import (GUI からは import しない)。
固定コミット 2b90b9f5 の検証済み API に従う:
  build_sam2_video_predictor(config_file, ckpt_path, device=..., vos_optimized=False)
  predictor.init_state(video_path=<frame_dir>, offload_video_to_cpu, offload_state_to_cpu, async_loading_frames)
  predictor.add_new_mask(inference_state, frame_idx, obj_id, mask)         # mask は2D bool
  predictor.propagate_in_video(inference_state, start_frame_idx, max_frame_num_to_track, reverse)
      -> ジェネレータ yield (frame_idx, obj_ids, video_res_masks)  # logits (num_obj,1,H,W)

V0.7 標準: vos_optimized=False / torch.compile 不使用 / offload_video_to_cpu=True。
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Iterator

import numpy as np

_log = logging.getLogger("sam_worker.video")

_BYTES_PER_MB = 1024 * 1024


class Sam2VideoManager:
    def __init__(self) -> None:
        self._predictor = None
        self._device: str | None = None
        self._precision: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    @property
    def predictor(self):
        return self._predictor

    # ------------------------------------------------------------------ #

    def _autocast(self):
        import torch
        if self._precision == "bf16":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        if self._precision == "fp16":
            return torch.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    def build(self, config_name: str, checkpoint_path: str, device: str, precision: str) -> dict:
        """Video Predictor を構築する。VRAM割当(MB)を返す。"""
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        self._predictor = build_sam2_video_predictor(
            config_name, checkpoint_path, device=device, vos_optimized=False,
        )
        self._device = device
        self._precision = precision
        _log.info("Video Predictor 構築: %s precision=%s device=%s",
                  type(self._predictor).__name__, precision, device)
        return {"vram_allocated_mb": self.vram_allocated_mb()}

    def init_state(
        self,
        frames_dir: str,
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
    ):
        if self._predictor is None:
            raise RuntimeError("Video Predictor が構築されていません")
        state = self._predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
        )
        return state

    def add_reference_mask(self, state, frame_idx: int, obj_id: int, mask_bool: np.ndarray) -> None:
        """基準マスク (2D bool) を frame_idx へ追加する。元配列は変更しない。"""
        import torch
        if mask_bool.ndim != 2:
            raise ValueError(f"基準マスクは2次元が必要: {mask_bool.shape}")
        m = np.ascontiguousarray(mask_bool.astype(bool))
        with torch.inference_mode(), self._autocast():
            self._predictor.add_new_mask(
                inference_state=state, frame_idx=int(frame_idx), obj_id=int(obj_id),
                mask=torch.from_numpy(m),
            )

    def propagate(
        self, state, start_frame_idx: int, max_frames: int, reverse: bool,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """伝播ジェネレータ。各フレームの (frame_idx, mask uint8 0/255 (H,W)) を yield。

        obj_id=1 (V0.7は1対象) を前提に video_res_masks[0,0] を logits>0 で2値化。
        """
        import torch
        with torch.inference_mode(), self._autocast():
            gen = self._predictor.propagate_in_video(
                inference_state=state,
                start_frame_idx=int(start_frame_idx),
                max_frame_num_to_track=int(max_frames),
                reverse=bool(reverse),
            )
            try:
                for f_idx, _obj_ids, video_res_masks in gen:
                    mask = (video_res_masks[0, 0] > 0.0).to("cpu").numpy().astype(np.uint8) * 255
                    yield int(f_idx), mask
            finally:
                close = getattr(gen, "close", None)
                if close is not None:
                    close()

    def vram_allocated_mb(self) -> int:
        try:
            import torch
            if torch.cuda.is_available():
                return int(torch.cuda.memory_allocated() / _BYTES_PER_MB)
        except Exception:
            pass
        return 0

    def release(self) -> None:
        import gc
        self._predictor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        _log.info("Video Predictor 解放")
