"""
V0.9: 完全被覆 partition 生成の CPU 専用 QProcess を管理するコントローラ (PySide6)。

GUI スレッドで重い分割・統合を実行しない。専用の CPU Worker
(partition_backend.partition_worker_main) を QProcess で起動し、JSON Lines で通信する。
このコントローラは torch / sam2 を import しない。
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcessEnvironment, Signal
from PySide6.QtCore import QProcess

from ai import protocol
from ai.partition_protocol import PartitionCommand, PartitionEvent


class PartitionController(QObject):
    build_started = Signal(str)             # job_id
    stage_changed = Signal(str, str)        # job_id, stage
    progress = Signal(str, float)           # job_id, fraction
    build_completed = Signal(str, dict)     # job_id, info
    build_failed = Signal(str, str, str)    # job_id, error_code, message
    build_cancelled = Signal(str)           # job_id
    worker_error = Signal(str)

    def __init__(self, python_executable: str | None = None, parent=None):
        super().__init__(parent)
        self._py = python_executable or sys.executable
        self._proc: QProcess | None = None
        self._decoder = protocol.JsonLineDecoder()
        self._rid = 0
        self._active_job: str | None = None

    # ---- プロセス管理 ---- #
    def start(self) -> None:
        if self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning:
            return
        proc = QProcess(self)
        pkg_root = Path(__file__).resolve().parent.parent  # colmap_mask_editor/
        env = QProcessEnvironment.systemEnvironment()
        existing = env.value("PYTHONPATH", "")
        env.insert("PYTHONPATH", str(pkg_root) + (";" + existing if existing else ""))
        env.insert("PYTHONIOENCODING", "utf-8")
        proc.setProcessEnvironment(env)
        proc.setProgram(self._py)
        proc.setArguments(["-m", "partition_backend.partition_worker_main"])
        proc.readyReadStandardOutput.connect(self._on_stdout)
        proc.errorOccurred.connect(lambda e: self.worker_error.emit(f"QProcess error: {e}"))
        proc.start()
        self._proc = proc

    def is_running(self) -> bool:
        return (self._proc is not None
                and self._proc.state() != QProcess.ProcessState.NotRunning)

    def shutdown(self) -> None:
        if self._proc is None:
            return
        try:
            self._send(PartitionCommand.SHUTDOWN)
            self._proc.waitForFinished(3000)
        finally:
            if self._proc.state() != QProcess.ProcessState.NotRunning:
                self._proc.kill()
            self._proc = None

    # ---- コマンド ---- #
    def build(self, *, job_id: str, image_path: str, image_key: str,
              output_dir: str, settings: dict, segments_path: str | None = None) -> None:
        self.start()
        self._active_job = job_id
        self._send(PartitionCommand.BUILD_START, job_id=job_id,
                   image_path=image_path, image_key=image_key,
                   output_dir=output_dir, settings=settings,
                   segments_path=segments_path)

    def cancel(self) -> None:
        if self.is_running():
            self._send(PartitionCommand.BUILD_CANCEL)

    # ---- 内部 ---- #
    def _send(self, command: str, **fields) -> None:
        if self._proc is None:
            return
        self._rid += 1
        msg = protocol.make_request(command, self._rid, **fields)
        self._proc.write(protocol.encode_message(msg))

    def _on_stdout(self) -> None:
        if self._proc is None:
            return
        data = bytes(self._proc.readAllStandardOutput())
        for parsed in self._decoder.feed(data):
            if not parsed.ok:
                continue
            self._dispatch(parsed.obj)

    def _dispatch(self, msg: dict) -> None:
        event = msg.get("event")
        job = msg.get("job_id", self._active_job or "")
        if event == PartitionEvent.BUILD_STARTED:
            self.build_started.emit(job)
        elif event == PartitionEvent.STAGE_CHANGED:
            self.stage_changed.emit(job, msg.get("stage", ""))
        elif event == PartitionEvent.PROGRESS:
            self.progress.emit(job, float(msg.get("fraction", 0.0)))
        elif event == PartitionEvent.BUILD_COMPLETED:
            self._active_job = None
            self.build_completed.emit(job, msg)
        elif event == PartitionEvent.BUILD_CANCELLED:
            self._active_job = None
            self.build_cancelled.emit(job)
        elif event == protocol.Event.ERROR or msg.get("status") == protocol.Status.ERROR:
            self._active_job = None
            self.build_failed.emit(job, msg.get("error_code", ""), msg.get("message", ""))
