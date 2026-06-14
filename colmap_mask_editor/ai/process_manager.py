"""
QProcess による SAM Worker (常駐子プロセス) の管理。

役割:
  - sys.executable (既定) で worker_main を非バッファリング起動する
  - stdout を JSON Lines として行単位デコードし、イベントを Signal で配る
  - stderr は別系統でログとして配る (stdout とマージしない)
  - request_id ごとにタイムアウトを管理する
  - shutdown -> terminate -> kill の三段階で安全に終了する

GUI スレッドから使う想定。torch / sam2 は一切 import しない。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from ai import protocol
from ai.protocol import JsonLineDecoder

_log = logging.getLogger(__name__)


# コマンド種別ごとの既定タイムアウト (ms)。QSettings で上書き可能 (詳細設定)。
DEFAULT_TIMEOUTS_MS: dict[str, int] = {
    protocol.Command.HELLO: 30_000,
    protocol.Command.HEALTH: 30_000,
    protocol.Command.LOAD_MODEL: 180_000,
    protocol.Command.SET_IMAGE: 120_000,
    protocol.Command.PREDICT: 30_000,
    protocol.Command.UNLOAD_MODEL: 30_000,
    protocol.Command.RELEASE_IMAGE: 30_000,
    protocol.Command.CLEAR_CUDA_CACHE: 30_000,
    protocol.Command.SHUTDOWN: 5_000,
}

WORKER_START_TIMEOUT_MS = 30_000
GRACEFUL_SHUTDOWN_WAIT_MS = 5_000
TERMINATE_WAIT_MS = 3_000


class SamProcessManager(QObject):
    """SAM Worker プロセスのライフサイクルと JSON Lines 通信を司る。"""

    # Worker -> GUI
    ready = Signal(dict)              # hello 応答 (event=ready)
    event_received = Signal(dict)    # 成功イベント全般 (readyを除く)
    error_received = Signal(dict)    # error イベント
    log_line = Signal(str)           # stderr 1行
    request_timed_out = Signal(int, str)  # request_id, command

    # プロセス状態
    worker_started = Signal()
    worker_stopped = Signal(int, str)   # exit_code, 説明
    worker_crashed = Signal(int, str)   # exit_code, 説明 (異常終了)
    start_failed = Signal(str)          # 起動失敗 (errorOccurred=FailedToStart)

    def __init__(
        self,
        worker_main_path: Optional[Path] = None,
        python_executable: Optional[str] = None,
        timeouts_ms: Optional[dict[str, int]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)

        if worker_main_path is None:
            worker_main_path = (
                Path(__file__).resolve().parent.parent / "sam_backend" / "worker_main.py"
            )
        self._worker_main_path = Path(worker_main_path)
        self._python_executable = python_executable or sys.executable

        self._timeouts = dict(DEFAULT_TIMEOUTS_MS)
        if timeouts_ms:
            self._timeouts.update(timeouts_ms)

        self._proc: Optional[QProcess] = None
        self._decoder = JsonLineDecoder()
        self._next_request_id = 0
        self._intentional_stop = False

        # request_id -> QTimer (タイムアウト監視中)
        self._pending: dict[int, QTimer] = {}
        # 再利用するタイマープール
        self._free_timers: list[QTimer] = []

    # ------------------------------------------------------------------ #
    # 設定
    # ------------------------------------------------------------------ #

    def set_python_executable(self, path: str) -> None:
        self._python_executable = path or sys.executable

    def set_timeout(self, command: str, ms: int) -> None:
        self._timeouts[command] = int(ms)

    @property
    def python_executable(self) -> str:
        return self._python_executable

    @property
    def worker_main_path(self) -> Path:
        return self._worker_main_path

    # ------------------------------------------------------------------ #
    # 起動 / 状態
    # ------------------------------------------------------------------ #

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.state() == QProcess.ProcessState.Running

    def process_id(self) -> Optional[int]:
        """起動中 Worker の OS プロセス ID。未起動なら None。"""
        if self._proc is None:
            return None
        pid = int(self._proc.processId())
        return pid if pid > 0 else None

    def start(self) -> bool:
        """Worker を起動する。既に起動済みなら False。"""
        if self.is_running():
            _log.warning("Worker は既に起動しています")
            return False

        if not self._worker_main_path.exists():
            self.start_failed.emit(f"worker_main が見つかりません: {self._worker_main_path}")
            return False

        self._intentional_stop = False
        self._decoder.reset()

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        proc.setProgram(self._python_executable)
        proc.setArguments(["-u", str(self._worker_main_path)])
        proc.setProcessEnvironment(self._build_environment())

        proc.started.connect(self._on_started)
        proc.readyReadStandardOutput.connect(self._on_stdout)
        proc.readyReadStandardError.connect(self._on_stderr)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error_occurred)
        proc.stateChanged.connect(self._on_state_changed)

        self._proc = proc
        _log.info("Worker起動: %s -u %s", self._python_executable, self._worker_main_path)
        proc.start()
        return True

    def _build_environment(self) -> QProcessEnvironment:
        env = QProcessEnvironment.systemEnvironment()
        # 非バッファリング・UTF-8 を強制 (日本語パス対策)
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")
        # Worker が ai / sam_backend を import できるようパッケージルートを通す
        package_root = str(self._worker_main_path.resolve().parent.parent)
        existing = env.value("PYTHONPATH", "")
        joined = package_root if not existing else package_root + ";" + existing
        env.insert("PYTHONPATH", joined)
        return env

    # ------------------------------------------------------------------ #
    # コマンド送信
    # ------------------------------------------------------------------ #

    def next_request_id(self) -> int:
        self._next_request_id += 1
        return self._next_request_id

    def send_command(
        self,
        command: str,
        request_id: Optional[int] = None,
        timeout_ms: Optional[int] = None,
        **fields: Any,
    ) -> int:
        """
        コマンドを Worker へ送る。request_id を省略すると自動採番する。
        送信した request_id を返す。タイムアウト監視を開始する。
        """
        if not self.is_running():
            raise RuntimeError("Worker が起動していません")

        if request_id is None:
            request_id = self.next_request_id()

        msg = protocol.make_request(command, request_id, **fields)
        data = protocol.encode_message(msg)
        assert self._proc is not None
        self._proc.write(data)

        self._arm_timeout(request_id, command, timeout_ms)
        _log.debug("送信: command=%s request_id=%d", command, request_id)
        return request_id

    def _acquire_timer(self) -> QTimer:
        """タイマーを使い回す (per-request の QTimer 生成/deleteLater は
        ネストしたイベントループ内でクラッシュを招くため、プールで再利用する)。"""
        if self._free_timers:
            return self._free_timers.pop()
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda t=timer: self._on_timer_fired(t))
        return timer

    def _arm_timeout(self, request_id: int, command: str, timeout_ms: Optional[int]) -> None:
        ms = timeout_ms if timeout_ms is not None else self._timeouts.get(command, 30_000)
        if ms <= 0:
            return
        timer = self._acquire_timer()
        timer.setProperty("request_id", int(request_id))
        timer.setProperty("command", str(command))
        self._pending[request_id] = timer
        timer.start(ms)

    def _resolve_pending(self, request_id: int) -> None:
        timer = self._pending.pop(request_id, None)
        if timer is not None:
            timer.stop()
            self._free_timers.append(timer)

    def _on_timer_fired(self, timer: QTimer) -> None:
        request_id = timer.property("request_id")
        command = timer.property("command")
        # 既に解決済みならタイマーがプールに戻っている。二重発火を防ぐ。
        if request_id not in self._pending or self._pending.get(request_id) is not timer:
            return
        del self._pending[request_id]
        self._free_timers.append(timer)
        _log.warning("応答タイムアウト: command=%s request_id=%s", command, request_id)
        self.request_timed_out.emit(int(request_id), str(command))

    # ------------------------------------------------------------------ #
    # stdout / stderr 受信
    # ------------------------------------------------------------------ #

    def _on_stdout(self) -> None:
        assert self._proc is not None
        data = bytes(self._proc.readAllStandardOutput())
        for parsed in self._decoder.feed(data):
            if not parsed.ok:
                # JSON 以外が stdout に来た -> 破棄しログへ (Worker のバグ/汚染)
                _log.warning("stdout 非JSON行を破棄: %r (%s)", parsed.raw[:200], parsed.error)
                continue
            self._dispatch(parsed.obj)

    def _on_stderr(self) -> None:
        assert self._proc is not None
        data = bytes(self._proc.readAllStandardError())
        text = data.decode("utf-8", errors="replace")
        for line in text.splitlines():
            if line.strip():
                self.log_line.emit(line)
                _log.debug("[worker stderr] %s", line)

    def _dispatch(self, msg: dict) -> None:
        request_id = msg.get("request_id")
        if isinstance(request_id, int):
            self._resolve_pending(request_id)

        event = msg.get("event")
        if protocol.is_error(msg):
            self.error_received.emit(msg)
        elif event == protocol.Event.READY:
            self.ready.emit(msg)
        else:
            self.event_received.emit(msg)

    # ------------------------------------------------------------------ #
    # プロセスシグナル
    # ------------------------------------------------------------------ #

    def _on_started(self) -> None:
        _log.info("Worker プロセス開始 (pid=%s)", self._proc.processId() if self._proc else "?")
        self.worker_started.emit()

    def _on_state_changed(self, state) -> None:
        _log.debug("Worker state changed: %s", state)

    def _on_error_occurred(self, error) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            msg = (
                f"Worker を起動できませんでした。Python 実行ファイルを確認してください:\n"
                f"{self._python_executable}"
            )
            _log.error(msg)
            self.start_failed.emit(msg)

    def _on_finished(self, exit_code: int, exit_status) -> None:
        # 全 pending タイムアウトを止める
        for _rid, timer in list(self._pending.items()):
            timer.stop()
            self._free_timers.append(timer)
        self._pending.clear()

        crashed = exit_status == QProcess.ExitStatus.CrashExit
        _log.info("Worker終了: exit_code=%d, crashed=%s, intentional=%s",
                  exit_code, crashed, self._intentional_stop)

        if self._intentional_stop:
            self.worker_stopped.emit(exit_code, "正常終了")
        elif crashed or exit_code != 0:
            self.worker_crashed.emit(
                exit_code,
                f"Worker が予期せず終了しました (exit_code={exit_code}, crashed={crashed})",
            )
        else:
            self.worker_stopped.emit(exit_code, "終了")

    # ------------------------------------------------------------------ #
    # 終了処理 (shutdown -> terminate -> kill)
    # ------------------------------------------------------------------ #

    def request_shutdown(self) -> None:
        """shutdown コマンドを送る (非同期)。応答後にプロセスが自走終了する想定。"""
        if self.is_running():
            try:
                self.send_command(protocol.Command.SHUTDOWN)
            except RuntimeError:
                pass

    def stop(self, graceful_wait_ms: int = GRACEFUL_SHUTDOWN_WAIT_MS) -> None:
        """
        Worker を確実に終了させる (アプリ終了時に呼ぶ・ブロッキング)。
        1. shutdown 送信 → waitForFinished
        2. terminate → waitForFinished
        3. kill (最終手段)
        """
        proc = self._proc
        if proc is None:
            return
        if proc.state() == QProcess.ProcessState.NotRunning:
            self._proc = None
            return

        self._intentional_stop = True

        # 1. graceful
        try:
            proc.write(protocol.encode_message(
                protocol.make_request(protocol.Command.SHUTDOWN, self.next_request_id())
            ))
            proc.closeWriteChannel()
        except Exception:
            pass

        if proc.waitForFinished(graceful_wait_ms):
            _log.info("Worker は shutdown で正常終了しました")
            self._proc = None
            return

        # 2. terminate
        _log.warning("shutdown 応答なし -> terminate")
        proc.terminate()
        if proc.waitForFinished(TERMINATE_WAIT_MS):
            self._proc = None
            return

        # 3. kill (最終手段)
        _log.error("terminate 失敗 -> kill (最終手段)")
        proc.kill()
        proc.waitForFinished(TERMINATE_WAIT_MS)
        self._proc = None
