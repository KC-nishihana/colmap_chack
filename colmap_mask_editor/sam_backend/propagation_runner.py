"""
V0.7 Worker側: 伝播ジョブを専用バックグラウンドスレッドで実行する。

propagate_in_video() を Worker メインループで同期実行すると pause/cancel/status を
受信できないため、ジョブは PropagationRunner スレッドで走らせ、メインスレッドは
stdin を読み続けて pause/resume/cancel を Event 経由で伝える。

stdout への JSON 書き込みは worker 側の Lock で保護した send_cb 経由で行う
(1行のJSONが複数スレッドで混在しないようにする)。
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ai import propagation_manifest as manifest
from ai import propagation_staging as staging
from ai.propagation_protocol import (
    PropagationDirection,
    PropagationErrorCode,
    PropagationEvent,
    make_job_error,
    make_job_event,
)
from ai.propagation_quality import QualityThresholds, compute_metrics
from ai.propagation_session import FrameState

_log = logging.getLogger("sam_worker.propagation")

_REFERENCE_OBJ_ID = 1


class _Cancelled(Exception):
    pass


def _is_oom(exc: Exception) -> bool:
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


class PropagationRunner(threading.Thread):
    def __init__(
        self,
        *,
        job_id: str,
        params: dict[str, Any],
        video_manager,
        job_dir: Path,
        send_cb: Callable[[dict], None],
        on_finished: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(name=f"prop-{job_id}", daemon=True)
        self._job_id = job_id
        self._p = params
        self._vm = video_manager
        self._job_dir = Path(job_dir)
        self._send = send_cb
        self._on_finished = on_finished

        self._pause = threading.Event()
        self._cancel = threading.Event()
        self._paused_emitted = False

        self._written: dict[int, dict] = {}

    # ---- 外部制御 (worker メインスレッドから) ---------------------------- #

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def cancel(self) -> None:
        self._cancel.set()
        self._pause.clear()  # 一時停止待ちを解除して速やかにキャンセルへ

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def processed(self) -> int:
        return len(self._written)

    @property
    def total(self) -> int:
        return len(self._p.get("frames", []))

    @property
    def is_paused(self) -> bool:
        return self._pause.is_set()

    # ---- スレッド本体 -------------------------------------------------- #

    def run(self) -> None:
        try:
            self._run_job()
        except _Cancelled:
            self._finalize_results(cancelled=True)
            self._emit(PropagationEvent.CANCELLED,
                       completed_count=len(self._written), reason="ユーザーキャンセル")
        except Exception as e:  # noqa: BLE001
            code = (PropagationErrorCode.GPU_OOM if _is_oom(e)
                    else PropagationErrorCode.PREDICT_FAILED)
            _log.error("[%s] 伝播失敗:\n%s", self._job_id, traceback.format_exc())
            self._finalize_results(cancelled=True)  # 完成済みは保持
            self._send(make_job_error(code, f"伝播に失敗しました: {e}", job_id=self._job_id))
        finally:
            try:
                self._vm.release()
            except Exception:
                pass
            if self._on_finished is not None:
                self._on_finished(self._job_id)

    # ---- 実処理 -------------------------------------------------------- #

    def _run_job(self) -> None:
        p = self._p
        frames_dir = self._job_dir / "frames"
        results_dir = self._job_dir / "results"
        ref_idx = int(p["reference_frame_index"])
        direction = p["direction"]
        max_frames = int(p.get("max_frames", 100))

        # 1. ステージング
        self._check_cancel()
        fm = staging.stage_sequence(
            p["frames"], frames_dir, reference_frame_index=ref_idx,
            jpeg_quality=int(p.get("jpeg_quality", 95)),
        )
        manifest.atomic_write_json(
            self._job_dir / "frame_manifest.json",
            manifest.build_frame_manifest(self._job_id, ref_idx, fm["width"], fm["height"], fm["frames"]),
        )
        self._width, self._height = fm["width"], fm["height"]
        self._entry_by_index = {f["frame_index"]: f["entry_key"] for f in fm["frames"]}

        # 2. Video Predictor 構築 + init_state
        self._check_cancel()
        self._vm.build(p["config_name"], p["checkpoint_path"], p["device"], p["precision"])
        state = self._vm.init_state(
            str(frames_dir),
            offload_video_to_cpu=bool(p.get("offload_video_to_cpu", True)),
            offload_state_to_cpu=bool(p.get("offload_state_to_cpu", False)),
            async_loading_frames=bool(p.get("async_loading_frames", False)),
        )

        # 3. 基準マスク
        self._check_cancel()
        ref_mask = staging.read_mask_png(p["reference_mask_path"]) > 0
        if ref_mask.shape != (self._height, self._width):
            raise RuntimeError(
                f"基準マスクサイズ不一致: {ref_mask.shape} != {(self._height, self._width)}"
            )
        self._vm.add_reference_mask(state, ref_idx, _REFERENCE_OBJ_ID, ref_mask)

        self._thresholds = _thresholds_from(p.get("thresholds"))
        self._results_dir = results_dir

        # 4. 伝播 (方向に応じて前方向/後方向)
        if direction in (PropagationDirection.FORWARD, PropagationDirection.BOTH):
            self._run_pass(state, ref_idx, max_frames, reverse=False)
        if direction in (PropagationDirection.BACKWARD, PropagationDirection.BOTH):
            self._run_pass(state, ref_idx, max_frames, reverse=True)

        # 5. 完了
        self._finalize_results(cancelled=False)
        self._emit(
            PropagationEvent.COMPLETED,
            completed_count=len(self._written),
            warning_count=sum(1 for r in self._written.values() if r["warning_codes"]),
        )

    def _run_pass(self, state, ref_idx: int, max_frames: int, reverse: bool) -> None:
        prev_mask: np.ndarray | None = None
        total = len(self._p["frames"])
        for f_idx, mask in self._vm.propagate(state, ref_idx, max_frames, reverse):
            self._wait_if_paused()  # cancel もここで送出

            if f_idx in self._written:
                # 前後で基準フレームが重複 yield される -> 二重保存しない
                prev_mask = mask
                continue

            entry_key = self._entry_by_index.get(f_idx, str(f_idx))
            met = compute_metrics(mask, prev_mask=prev_mask, thresholds=self._thresholds)

            dest = self._results_dir / f"{f_idx:06d}.png"
            try:
                staging.write_mask_png_atomic(dest, mask)
            except Exception as e:  # noqa: BLE001
                self._send(make_job_error(
                    PropagationErrorCode.RESULT_WRITE_FAILED,
                    f"結果書込失敗 frame {f_idx}: {e}", job_id=self._job_id))
                raise

            is_ref = (f_idx == ref_idx)
            state_str = FrameState.WARNING if met.warning_codes else FrameState.DONE
            self._written[f_idx] = {
                "frame_index": f_idx,
                "entry_key": entry_key,
                "result_mask_path": str(dest),
                "state": state_str,
                "warning_codes": list(met.warning_codes),
                "foreground_pixels": met.foreground_pixels,
                "foreground_ratio": round(met.foreground_ratio, 6),
                "bbox": list(met.bbox) if met.bbox else None,
                "centroid": list(met.centroid) if met.centroid else None,
                "component_count": met.component_count,
                "is_reference": is_ref,
            }
            self._emit(
                PropagationEvent.FRAME_READY,
                frame_index=f_idx, entry_key=entry_key,
                result_mask_path=str(dest),
                foreground_ratio=round(met.foreground_ratio, 6),
                warning_codes=list(met.warning_codes),
                is_reference=is_ref,
            )
            self._emit(
                PropagationEvent.PROGRESS,
                processed=len(self._written), total=total,
                frame_index=f_idx, vram_allocated_mb=self._vm.vram_allocated_mb(),
            )
            prev_mask = mask

    # ---- pause / cancel ------------------------------------------------ #

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise _Cancelled()

    def _wait_if_paused(self) -> None:
        self._check_cancel()
        if self._pause.is_set():
            if not self._paused_emitted:
                self._emit(PropagationEvent.PAUSED)
                self._paused_emitted = True
            while self._pause.is_set():
                if self._cancel.is_set():
                    raise _Cancelled()
                time.sleep(0.05)
            if self._paused_emitted:
                self._emit(PropagationEvent.RESUMED)
                self._paused_emitted = False

    # ---- 結果 manifest ------------------------------------------------- #

    def _finalize_results(self, cancelled: bool) -> None:
        try:
            results = [self._written[k] for k in sorted(self._written)]
            data = manifest.build_result_manifest(
                self._job_id, int(self._p["reference_frame_index"]),
                getattr(self, "_width", 0), getattr(self, "_height", 0), results,
            )
            data["cancelled"] = bool(cancelled)
            manifest.atomic_write_json(self._job_dir / "results" / "result_manifest.json", data)
        except Exception:
            _log.error("[%s] result_manifest 書込失敗:\n%s", self._job_id, traceback.format_exc())

    def _emit(self, event: str, **fields) -> None:
        self._send(make_job_event(event, self._job_id, **fields))


def _thresholds_from(d: dict | None) -> QualityThresholds:
    if not d:
        return QualityThresholds()
    base = QualityThresholds()
    return QualityThresholds(
        too_large_ratio=float(d.get("too_large_ratio", base.too_large_ratio)),
        area_drop_ratio=float(d.get("area_drop_ratio", base.area_drop_ratio)),
        area_growth_ratio=float(d.get("area_growth_ratio", base.area_growth_ratio)),
        component_count=int(d.get("component_count", base.component_count)),
        low_iou=float(d.get("low_iou", base.low_iou)),
    )
