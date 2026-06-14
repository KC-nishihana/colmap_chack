"""
V0.7 GUI側: 伝播オーケストレーション。

AiSession と同じ SamProcessManager (常駐Worker) を共有し、伝播コマンドを送信して
進捗イベントを PropagationSession へ反映する。torch / sam2 は import しない。

古い job_id のイベントは反映せず、関連一時結果を削除する。
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Signal

from ai import runtime_paths
from ai.propagation_protocol import PropagationCommand, PropagationErrorCode, PropagationEvent
from ai.propagation_session import (
    FrameState,
    PropagationFrame,
    PropagationSession,
    PropagationUiState,
)
from ai.propagation_staging import write_mask_png_atomic

_log = logging.getLogger(__name__)


class PropagationController(QObject):
    started = Signal(str)          # job_id
    frame_ready = Signal(int)      # frame_index
    progress = Signal(int, int)    # processed, total
    completed = Signal()
    failed = Signal(str, str)      # error_code, message
    paused = Signal()
    resumed = Signal()
    cancelled = Signal()
    state_changed = Signal(object)  # PropagationUiState

    def __init__(self, process_manager, parent=None) -> None:
        super().__init__(parent)
        self._pm = process_manager
        self._session: Optional[PropagationSession] = None
        self._pending: Optional[PropagationSession] = None
        self._active_job_id: Optional[str] = None
        self._start_rid: Optional[int] = None
        self._pm.event_received.connect(self._on_event)
        self._pm.error_received.connect(self._on_error)

    # ------------------------------------------------------------------ #

    @property
    def session(self) -> Optional[PropagationSession]:
        return self._session

    @property
    def state(self) -> PropagationUiState:
        return self._session.state if self._session else PropagationUiState.IDLE

    @property
    def active_job_id(self) -> Optional[str]:
        return self._active_job_id

    def is_active(self) -> bool:
        return self._session is not None and self._session.state in (
            PropagationUiState.STAGING, PropagationUiState.INITIALIZING,
            PropagationUiState.RUNNING, PropagationUiState.PAUSED,
            PropagationUiState.CANCELLING,
        )

    def has_unapplied_results(self) -> bool:
        return self._session is not None and self._session.state == PropagationUiState.REVIEW

    # ------------------------------------------------------------------ #

    def start(
        self,
        *,
        frames: list[PropagationFrame],
        reference_frame_index: int,
        reference_mask: np.ndarray,
        order_mode,
        direction: str,
        model_id: str,
        checkpoint_path: str,
        precision: str = "bf16",
        device: str = "cuda:0",
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = False,
        max_frames: int = 100,
        jpeg_quality: int = 95,
        thresholds: Optional[dict] = None,
    ) -> None:
        """伝播を開始する。reference_mask は uint8 0/255 (H,W)。"""
        ref = frames[reference_frame_index]
        ref_path = runtime_paths.get_propagation_root() / f"ref_{uuid.uuid4().hex[:8]}.png"
        write_mask_png_atomic(ref_path, reference_mask)

        self._pending = PropagationSession(
            job_id="", state=PropagationUiState.STAGING,
            reference_entry_key=ref.entry_key, reference_frame_index=reference_frame_index,
            reference_mask_path=str(ref_path), order_mode=order_mode, direction=direction,
            frames=list(frames), model_id=model_id, precision=precision, device=device,
        )
        wire = [{"frame_index": f.frame_index, "entry_key": f.entry_key,
                 "source_path": f.source_path} for f in frames]
        self._start_rid = self._pm.send_command(
            PropagationCommand.START,
            frames=wire, reference_frame_index=reference_frame_index,
            reference_mask_path=str(ref_path), model_id=model_id,
            checkpoint_path=checkpoint_path, precision=precision, device=device,
            direction=direction, offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu, max_frames=max_frames,
            jpeg_quality=jpeg_quality, thresholds=thresholds,
        )

    def pause(self) -> None:
        if self._active_job_id:
            self._pm.send_command(PropagationCommand.PAUSE, job_id=self._active_job_id)

    def resume(self) -> None:
        if self._active_job_id:
            self._pm.send_command(PropagationCommand.RESUME, job_id=self._active_job_id)

    def cancel(self) -> None:
        if self._active_job_id:
            self._pm.send_command(PropagationCommand.CANCEL, job_id=self._active_job_id)

    def release(self) -> None:
        if self._active_job_id:
            self._pm.send_command(PropagationCommand.RELEASE, job_id=self._active_job_id)
        self._discard_session_temp()
        self._session = None
        self._active_job_id = None

    def discard_session(self) -> None:
        """レビュー結果を破棄しセッションを終了する (一時結果を削除)。"""
        self._discard_session_temp()
        self._session = None
        self._active_job_id = None
        self._set_state(PropagationUiState.IDLE)

    # ------------------------------------------------------------------ #

    def _set_state(self, state: PropagationUiState) -> None:
        if self._session is not None:
            self._session.state = state
        self.state_changed.emit(state)

    def _on_event(self, msg: dict) -> None:
        event = msg.get("event")
        if event == PropagationEvent.STARTED and msg.get("request_id") == self._start_rid:
            self._active_job_id = msg.get("job_id")
            self._session = self._pending
            self._pending = None
            if self._session is not None:
                self._session.job_id = self._active_job_id
            self._set_state(PropagationUiState.RUNNING)
            self.started.emit(self._active_job_id or "")
            return

        job_id = msg.get("job_id")
        if job_id is None or event not in PropagationEvent.ALL:
            return
        if job_id != self._active_job_id:
            self._discard_foreign(msg)
            return

        if event == PropagationEvent.FRAME_READY:
            self._apply_frame(msg)
        elif event == PropagationEvent.PROGRESS:
            self.progress.emit(int(msg.get("processed", 0)), int(msg.get("total", 0)))
        elif event == PropagationEvent.PAUSED:
            self._set_state(PropagationUiState.PAUSED)
            self.paused.emit()
        elif event == PropagationEvent.RESUMED:
            self._set_state(PropagationUiState.RUNNING)
            self.resumed.emit()
        elif event == PropagationEvent.CANCELLING:
            self._set_state(PropagationUiState.CANCELLING)
        elif event == PropagationEvent.CANCELLED:
            self._finalize_review()
            self.cancelled.emit()
        elif event == PropagationEvent.COMPLETED:
            self._finalize_review()
            self.completed.emit()

    def _on_error(self, msg: dict) -> None:
        code = msg.get("error_code", "")
        if code not in PropagationErrorCode.ALL:
            return  # 単一画像AIのエラーは AiSession 側で処理
        job_id = msg.get("job_id")
        if self._active_job_id and job_id and job_id != self._active_job_id:
            return
        self._set_state(PropagationUiState.ERROR)
        self.failed.emit(code, msg.get("message", ""))

    def _apply_frame(self, msg: dict) -> None:
        if self._session is None:
            return
        idx = int(msg.get("frame_index"))
        frame = self._session.frame_by_index(idx)
        if frame is None:
            return
        frame.result_mask_path = msg.get("result_mask_path")
        frame.foreground_ratio = float(msg.get("foreground_ratio", 0.0))
        frame.warning_codes = list(msg.get("warning_codes", []))
        frame.state = FrameState.WARNING if frame.warning_codes else FrameState.DONE
        frame.accepted = True
        self.frame_ready.emit(idx)

    def _finalize_review(self) -> None:
        if self._session is None:
            return
        self._session.recompute_counts()
        self._set_state(PropagationUiState.REVIEW)

    def _discard_foreign(self, msg: dict) -> None:
        rp = msg.get("result_mask_path")
        if rp:
            runtime_paths.delete_result_file(rp)
        _log.info("古い job_id のイベントを破棄: %s", msg.get("job_id"))

    def _discard_session_temp(self) -> None:
        if self._session is None:
            return
        try:
            job_dir = runtime_paths.get_propagation_job_dir(self._session.job_id, create=False)
            ref = Path(self._session.reference_mask_path)
            if ref.exists():
                ref.unlink()
            _ = job_dir  # ジョブディレクトリ自体は履歴管理で扱う (即時削除しない)
        except OSError:
            pass
