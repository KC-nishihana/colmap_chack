"""
V0.8 GUI側: 全画像自動分割オーケストレーション。

AiSession と同じ SamProcessManager (常駐Worker) を共有し、AMG コマンドを送信して
進捗イベントを Qt シグナルへ橋渡しする。torch / sam2 は import しない。
古い job_id のイベントは反映しない。
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QObject, Signal

from ai.amg_protocol import AmgCommand, AmgErrorCode, AmgEvent

_log = logging.getLogger(__name__)


class AmgController(QObject):
    started = Signal(str)            # job_id
    image_started = Signal(dict)
    image_completed = Signal(dict)
    image_skipped = Signal(dict)
    image_failed = Signal(dict)
    progress = Signal(dict)
    paused = Signal()
    resumed = Signal()
    cancelling = Signal()
    cancelled = Signal(dict)
    completed = Signal(dict)
    failed = Signal(str, str)        # error_code, message

    def __init__(self, process_manager, parent=None) -> None:
        super().__init__(parent)
        self._pm = process_manager
        self._active_job_id: Optional[str] = None
        self._start_rid: Optional[int] = None
        self._running = False
        self._paused = False
        self._pm.event_received.connect(self._on_event)
        self._pm.error_received.connect(self._on_error)

    # ------------------------------------------------------------------ #

    @property
    def active_job_id(self) -> Optional[str]:
        return self._active_job_id

    def is_running(self) -> bool:
        return self._running

    def is_paused(self) -> bool:
        return self._paused

    # ------------------------------------------------------------------ #

    def start(
        self,
        *,
        project_root: str,
        images: list[dict],
        settings: dict,
        preset: str,
        model: dict,
        force: bool = False,
        oom_retry: bool = True,
        retry_single: bool = False,
    ) -> int:
        """AMG バッチを開始する。images = [{'image_key','source_path'}, ...]。"""
        command = AmgCommand.RETRY_IMAGE if retry_single else AmgCommand.BATCH_START
        self._start_rid = self._pm.send_command(
            command, project_root=str(project_root), images=images,
            settings=settings, preset=preset, model=model,
            force=force, oom_retry=oom_retry,
        )
        return self._start_rid

    def pause(self) -> None:
        if self._active_job_id:
            self._pm.send_command(AmgCommand.BATCH_PAUSE, job_id=self._active_job_id)

    def resume(self) -> None:
        if self._active_job_id:
            self._pm.send_command(AmgCommand.BATCH_RESUME, job_id=self._active_job_id)

    def cancel(self) -> None:
        if self._active_job_id:
            self._pm.send_command(AmgCommand.BATCH_CANCEL, job_id=self._active_job_id)

    def release(self) -> None:
        if self._active_job_id:
            self._pm.send_command(AmgCommand.RELEASE, job_id=self._active_job_id)
        self._reset()

    # ------------------------------------------------------------------ #

    def _reset(self) -> None:
        self._active_job_id = None
        self._running = False
        self._paused = False

    def _on_event(self, msg: dict) -> None:
        event = msg.get("event")
        if event == AmgEvent.BATCH_STARTED and msg.get("request_id") == self._start_rid:
            self._active_job_id = msg.get("job_id")
            self._running = True
            self._paused = False
            self.started.emit(self._active_job_id or "")
            return

        job_id = msg.get("job_id")
        if job_id is None or event not in AmgEvent.ALL:
            return
        if job_id != self._active_job_id:
            _log.info("古い job_id のイベントを破棄: %s", job_id)
            return

        if event == AmgEvent.IMAGE_STARTED:
            self.image_started.emit(msg)
        elif event == AmgEvent.IMAGE_COMPLETED:
            self.image_completed.emit(msg)
        elif event == AmgEvent.IMAGE_SKIPPED:
            self.image_skipped.emit(msg)
        elif event == AmgEvent.IMAGE_FAILED:
            self.image_failed.emit(msg)
        elif event == AmgEvent.BATCH_PROGRESS:
            self.progress.emit(msg)
        elif event == AmgEvent.BATCH_PAUSED:
            self._paused = True
            self.paused.emit()
        elif event == AmgEvent.BATCH_RESUMED:
            self._paused = False
            self.resumed.emit()
        elif event == AmgEvent.BATCH_CANCELLING:
            self.cancelling.emit()
        elif event == AmgEvent.BATCH_CANCELLED:
            self._running = False
            self._paused = False
            self.cancelled.emit(msg)
        elif event == AmgEvent.BATCH_COMPLETED:
            self._running = False
            self._paused = False
            self.completed.emit(msg)
        elif event == AmgEvent.BATCH_FAILED:
            self._running = False
            self._paused = False
            self.failed.emit("AMG_BATCH_FAILED", "バッチ処理に失敗しました")

    def _on_error(self, msg: dict) -> None:
        code = msg.get("error_code", "")
        if code not in AmgErrorCode.ALL:
            return  # 他系統のエラーは無視
        job_id = msg.get("job_id")
        # 開始前 (BUSY/MODEL_NOT_LOADED 等) は active_job_id 未設定でも通す
        if self._active_job_id and job_id and job_id != self._active_job_id:
            return
        # image_key 付きエラーは「画像 1 枚の失敗」。Worker は次画像を続けるので
        # バッチ全体を停止させてはならない。件数は後続の IMAGE_FAILED / BATCH_PROGRESS
        # イベントで更新されるため、ここでは image_failed に流すだけにする。
        if msg.get("image_key"):
            self.image_failed.emit(msg)
            return
        # image_key の無いエラー (開始失敗 / BUSY / バッチ致命) のみバッチ停止扱い。
        self.failed.emit(code, msg.get("message", ""))
