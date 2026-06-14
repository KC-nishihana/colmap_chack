"""
AIセグメンテーションの高レベルオーケストレーション。

SamProcessManager (QProcess通信) の上に状態機械 (AiUiState) を載せ、
GUI (MainWindow / AISegmentPanel) からは「Worker起動」「モデル読込」
「画像設定」「推論」「候補選択」だけを呼べばよいようにする。

責務:
  - 状態 (AiUiState) を1か所で管理し、変化を state_changed で通知
  - hello で cuda_extension_loaded=False の場合は AI を無効化
  - 画像切替で image_key を更新し、古い request_id / image_key の結果を破棄
  - 推論結果NPZの読込と候補保持 (最大3) ・選択
  - Workerクラッシュ時に通常マスクへ影響を与えず ERROR へ遷移

torch / sam2 は import しない。マスク本体は ai_mask_ops 経由でNPZから読む。
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal

from ai import protocol, runtime_paths
from ai.ai_mask_ops import (
    AiPredictionResult,
    NpzCorruptError,
    load_prediction_npz,
)
from ai.ai_prompt import AiPromptSession
from ai.process_manager import SamProcessManager

_log = logging.getLogger(__name__)


class AiUiState(Enum):
    DISABLED = auto()          # AI機能無効 (未起動 / CUDA拡張なし)
    WORKER_STARTING = auto()
    WORKER_READY = auto()      # hello完了・モデル未ロード
    MODEL_LOADING = auto()
    MODEL_READY = auto()       # モデルロード済み・画像未設定
    IMAGE_ENCODING = auto()
    PROMPT_EDITING = auto()    # 画像Embedding済み・プロンプト編集中
    PREDICTING = auto()
    PREVIEW = auto()           # 候補マスク表示中
    ERROR = auto()


class AiSession(QObject):
    """AIセグメンテーションのセッション/状態を司る。"""

    state_changed = Signal(object)        # AiUiState
    worker_info = Signal(dict)            # hello 応答 (GPU名等)
    model_ready = Signal(dict)            # model_loaded 応答
    image_ready = Signal(dict)            # image_ready 応答
    prediction_ready = Signal(object)     # AiPredictionResult
    candidate_changed = Signal(int)       # 選択候補インデックス
    error = Signal(str, str)              # error_code, message
    cuda_extension_unavailable = Signal(str)
    worker_unavailable = Signal(str)      # 起動失敗 / クラッシュ
    log = Signal(str)

    def __init__(
        self,
        worker_main_path: Optional[Path] = None,
        python_executable: Optional[str] = None,
        timeouts_ms: Optional[dict[str, int]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)

        self._proc = SamProcessManager(
            worker_main_path=worker_main_path,
            python_executable=python_executable,
            timeouts_ms=timeouts_ms,
            parent=self,
        )
        self._wire_process_signals()

        self._state = AiUiState.DISABLED

        # Worker 能力 (hello で確定)
        self._hello: dict = {}
        self._cuda_extension_loaded = False

        # モデル/画像状態
        self._model_id: Optional[str] = None
        self._precision: Optional[str] = None
        self._device: Optional[str] = None
        self._image_path: Optional[str] = None
        self._image_key: Optional[str] = None
        self._image_size: Optional[tuple[int, int]] = None  # (w, h)

        # プロンプト・推論
        self.prompts = AiPromptSession()
        self._result: Optional[AiPredictionResult] = None
        self._selected_candidate = -1
        self._active_predict_id = -1

        # 起動時に取り残しNPZを掃除
        try:
            runtime_paths.cleanup_old_files()
        except Exception as e:  # pragma: no cover - 掃除失敗は致命ではない
            _log.debug("起動時の一時ファイル掃除に失敗: %s", e)

    # ------------------------------------------------------------------ #
    # 状態
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> AiUiState:
        return self._state

    def _set_state(self, state: AiUiState) -> None:
        if self._state != state:
            self._state = state
            _log.debug("AI状態遷移: %s", state.name)
            self.state_changed.emit(state)

    @property
    def cuda_extension_loaded(self) -> bool:
        return self._cuda_extension_loaded

    @property
    def hello_info(self) -> dict:
        return dict(self._hello)

    @property
    def image_key(self) -> Optional[str]:
        return self._image_key

    @property
    def model_id(self) -> Optional[str]:
        return self._model_id

    @property
    def result(self) -> Optional[AiPredictionResult]:
        return self._result

    @property
    def selected_candidate_index(self) -> int:
        return self._selected_candidate

    def is_worker_running(self) -> bool:
        return self._proc.is_running()

    @property
    def process_manager(self):
        """共有 SamProcessManager (V0.7 伝播コントローラが同一Workerを使うため)。"""
        return self._proc

    def has_preview(self) -> bool:
        return self._state == AiUiState.PREVIEW and self._result is not None

    def has_pending(self) -> bool:
        """未確定 (プレビュー or プロンプトが残っている) かどうか。"""
        if self._state in (AiUiState.PREVIEW, AiUiState.PREDICTING):
            return True
        return self.prompts.has_any() and self._state in (
            AiUiState.PROMPT_EDITING, AiUiState.PREVIEW
        )

    # ------------------------------------------------------------------ #
    # Worker ライフサイクル
    # ------------------------------------------------------------------ #

    def set_python_executable(self, path: str) -> None:
        self._proc.set_python_executable(path)

    def set_timeout(self, command: str, ms: int) -> None:
        self._proc.set_timeout(command, ms)

    def start_worker(self) -> bool:
        if self._proc.is_running():
            return False
        self._set_state(AiUiState.WORKER_STARTING)
        ok = self._proc.start()
        if not ok:
            self._set_state(AiUiState.ERROR)
        return ok

    def restart_worker(self) -> None:
        """Worker を停止し再起動する。通常マスクには影響しない。"""
        self._discard_prediction()
        self._proc.stop(graceful_wait_ms=2000)
        self._model_id = None
        self._image_key = None
        self._image_size = None
        self.start_worker()

    def send_hello(self) -> None:
        if self._proc.is_running():
            self._proc.send_command(protocol.Command.HELLO)

    def send_health(self) -> None:
        if self._proc.is_running():
            self._proc.send_command(protocol.Command.HEALTH)

    def shutdown(self) -> None:
        """アプリ終了時に呼ぶ。ブロッキングで子プロセスを確実に終了。"""
        self._discard_prediction()
        self._proc.stop()
        self._set_state(AiUiState.DISABLED)

    # ------------------------------------------------------------------ #
    # モデル
    # ------------------------------------------------------------------ #

    def load_model(self, model_id: str, checkpoint_path: str,
                   precision: str = "bf16", device: str = "cuda:0") -> None:
        if not self._proc.is_running():
            self.error.emit(protocol.ErrorCode.INTERNAL, "Workerが起動していません")
            return
        if not self._cuda_extension_loaded:
            self.cuda_extension_unavailable.emit(
                "SAM 2 CUDA拡張が利用できないため、モデルを読み込めません。"
            )
            return
        self._model_id = model_id
        self._precision = precision
        self._device = device
        self._set_state(AiUiState.MODEL_LOADING)
        self._proc.send_command(
            protocol.Command.LOAD_MODEL,
            model_id=model_id,
            checkpoint_path=str(checkpoint_path),
            precision=precision,
            device=device,
        )

    def unload_model(self) -> None:
        if not self._proc.is_running():
            return
        self._discard_prediction()
        self._image_key = None
        self._proc.send_command(protocol.Command.UNLOAD_MODEL)
        self._model_id = None
        self._set_state(AiUiState.WORKER_READY)

    # ------------------------------------------------------------------ #
    # 画像
    # ------------------------------------------------------------------ #

    def set_image(self, image_path: str) -> None:
        if not self._proc.is_running() or self._model_id is None:
            return
        # 画像が変わるので古い結果・プロンプトを捨てる
        self._discard_prediction()
        self.prompts.reset()
        self._image_path = str(image_path)
        self._image_key = None  # image_ready まで未確定
        self._set_state(AiUiState.IMAGE_ENCODING)
        self._proc.send_command(
            protocol.Command.SET_IMAGE,
            image_path=str(image_path),
        )

    def release_image(self) -> None:
        if self._proc.is_running() and self._image_key is not None:
            self._proc.send_command(
                protocol.Command.RELEASE_IMAGE, image_key=self._image_key
            )
        self._image_key = None

    def invalidate_image(self) -> None:
        """
        画像切替時に呼ぶ。現在の Embedding/プロンプト/プレビューを無効化する。
        Worker への再エンコードは次に AI を使うときまで遅延する。
        通常マスクには影響しない。
        """
        self._discard_prediction()
        self.prompts.reset()
        self._image_key = None
        if self._model_id is not None and self._state in (
            AiUiState.PROMPT_EDITING, AiUiState.PREVIEW, AiUiState.IMAGE_ENCODING,
            AiUiState.PREDICTING,
        ):
            self._set_state(AiUiState.MODEL_READY)

    def needs_image_encoding(self) -> bool:
        """モデルはあるが現在画像の Embedding が無い状態か。"""
        return self._model_id is not None and self._image_key is None

    # ------------------------------------------------------------------ #
    # 推論
    # ------------------------------------------------------------------ #

    def predict(self, multimask_output: bool = True) -> None:
        if self._image_key is None:
            self.error.emit(protocol.ErrorCode.BAD_REQUEST, "画像が設定されていません")
            return
        if self.prompts.is_empty():
            self.error.emit(protocol.ErrorCode.BAD_REQUEST, "プロンプトがありません")
            return
        fields = self.prompts.to_predict_fields()
        self._set_state(AiUiState.PREDICTING)
        self._active_predict_id = self._proc.send_command(
            protocol.Command.PREDICT,
            image_key=self._image_key,
            multimask_output=multimask_output,
            **fields,
        )

    def select_candidate(self, index: int) -> None:
        """候補を切り替える (再推論しない)。"""
        if self._result is None:
            return
        if 0 <= index < self._result.mask_count:
            self._selected_candidate = index
            self.candidate_changed.emit(index)

    def selected_mask(self):
        """現在選択中の候補マスク (H,W uint8 0/255) を返す。無ければ None。"""
        if self._result is None or self._selected_candidate < 0:
            return None
        if self._selected_candidate >= self._result.mask_count:
            return None
        return self._result.candidates[self._selected_candidate].mask

    # ------------------------------------------------------------------ #
    # プレビュー破棄 / 適用後リセット
    # ------------------------------------------------------------------ #

    def discard_preview(self) -> None:
        """プレビューと候補を破棄する。プロンプトは保持し再推論可能。"""
        self._discard_prediction()
        if self._image_key is not None:
            self._set_state(AiUiState.PROMPT_EDITING)
        elif self._model_id is not None:
            self._set_state(AiUiState.MODEL_READY)

    def reset_after_apply(self) -> None:
        """適用後: プレビュー・プロンプト・候補をすべて破棄する。"""
        self._discard_prediction()
        self.prompts.reset()
        if self._image_key is not None:
            self._set_state(AiUiState.PROMPT_EDITING)
        elif self._model_id is not None:
            self._set_state(AiUiState.MODEL_READY)

    def _discard_prediction(self) -> None:
        """結果オブジェクトを破棄し、結果NPZを削除する。"""
        if self._result is not None:
            # NPZ は load 時点で読み切っているので、参照していた一時ファイルを掃除
            pass
        self._result = None
        self._selected_candidate = -1
        self._active_predict_id = -1

    # ------------------------------------------------------------------ #
    # SamProcessManager シグナルの受信
    # ------------------------------------------------------------------ #

    def _wire_process_signals(self) -> None:
        self._proc.ready.connect(self._on_ready)
        self._proc.event_received.connect(self._on_event)
        self._proc.error_received.connect(self._on_error)
        self._proc.log_line.connect(self.log)
        self._proc.request_timed_out.connect(self._on_timeout)
        self._proc.worker_started.connect(self._on_worker_started)
        self._proc.worker_crashed.connect(self._on_worker_crashed)
        self._proc.worker_stopped.connect(self._on_worker_stopped)
        self._proc.start_failed.connect(self._on_start_failed)

    def _on_worker_started(self) -> None:
        # プロセスは起動した。hello を送って能力確認。
        self.send_hello()

    def _on_ready(self, msg: dict) -> None:
        self._hello = dict(msg)
        self._cuda_extension_loaded = bool(msg.get("cuda_extension_loaded", False))
        self.worker_info.emit(msg)
        if not self._cuda_extension_loaded:
            self._set_state(AiUiState.ERROR)
            self.cuda_extension_unavailable.emit(
                msg.get("message")
                or "SAM 2 CUDA拡張を読み込めませんでした。AIセグメンテーションは使用できません。"
            )
            return
        self._set_state(AiUiState.WORKER_READY)

    def _on_event(self, msg: dict) -> None:
        event = msg.get("event")
        if event == protocol.Event.MODEL_LOADED:
            self._set_state(AiUiState.MODEL_READY)
            self.model_ready.emit(msg)
        elif event == protocol.Event.IMAGE_READY:
            self._image_key = msg.get("image_key")
            self._image_size = (int(msg.get("width", 0)), int(msg.get("height", 0)))
            self._set_state(AiUiState.PROMPT_EDITING)
            self.image_ready.emit(msg)
        elif event == protocol.Event.PREDICTION_READY:
            self._on_prediction_ready(msg)
        elif event == protocol.Event.MODEL_UNLOADED:
            pass
        elif event == protocol.Event.HEALTH_RESULT:
            pass
        # その他のイベントは無視 (image_released, cuda_cache_cleared 等)

    def _on_prediction_ready(self, msg: dict) -> None:
        request_id = msg.get("request_id", -1)
        result_path = msg.get("result_path")
        msg_image_key = msg.get("image_key")

        # 古い request_id / 別画像の結果は適用せず NPZ を削除する
        if request_id != self._active_predict_id:
            _log.info("古い予測結果を破棄: request_id=%s (active=%s)",
                      request_id, self._active_predict_id)
            runtime_paths.delete_result_file(result_path)
            return
        if msg_image_key != self._image_key:
            _log.info("別画像の予測結果を破棄: image_key=%s (current=%s)",
                      msg_image_key, self._image_key)
            runtime_paths.delete_result_file(result_path)
            return

        try:
            result = load_prediction_npz(
                result_path,
                expected_request_id=request_id,
                expected_image_key=self._image_key,
            )
        except NpzCorruptError as e:
            _log.error("予測結果NPZが不正: %s", e)
            runtime_paths.delete_result_file(result_path)
            self._set_state(AiUiState.PROMPT_EDITING)
            self.error.emit(protocol.ErrorCode.PREDICT_FAILED,
                            f"推論結果ファイルを読み込めませんでした:\n{e}")
            return

        # 読み込み完了したら結果ファイルは不要
        runtime_paths.delete_result_file(result_path)

        self._result = result
        self._selected_candidate = result.best_index()
        self._set_state(AiUiState.PREVIEW)
        self.prediction_ready.emit(result)

    def _on_error(self, msg: dict) -> None:
        code = msg.get("error_code", protocol.ErrorCode.INTERNAL)
        message = msg.get("message", "不明なエラー")
        _log.warning("Workerエラー: %s - %s", code, message)

        if code == protocol.ErrorCode.CUDA_EXTENSION_UNAVAILABLE:
            self._cuda_extension_loaded = False
            self._set_state(AiUiState.ERROR)
            self.cuda_extension_unavailable.emit(message)
            return

        if code == protocol.ErrorCode.CUDA_OOM:
            # Worker は維持。プレビューが出る前なら PROMPT_EDITING に戻す。
            self._discard_prediction()
            if self._image_key is not None:
                self._set_state(AiUiState.PROMPT_EDITING)
            else:
                self._set_state(AiUiState.MODEL_READY)
            self.error.emit(code, message)
            return

        # その他のエラーは状態を巻き戻す (通常マスクは触らない)
        self._discard_prediction()
        if self._state in (AiUiState.PREDICTING, AiUiState.PREVIEW, AiUiState.IMAGE_ENCODING):
            if self._image_key is not None:
                self._set_state(AiUiState.PROMPT_EDITING)
            elif self._model_id is not None:
                self._set_state(AiUiState.MODEL_READY)
            else:
                self._set_state(AiUiState.WORKER_READY)
        elif self._state == AiUiState.MODEL_LOADING:
            self._model_id = None
            self._set_state(AiUiState.WORKER_READY)
        self.error.emit(code, message)

    def _on_timeout(self, request_id: int, command: str) -> None:
        _log.warning("コマンドタイムアウト: %s (request_id=%d)", command, request_id)
        self._discard_prediction()
        if command == protocol.Command.LOAD_MODEL:
            self._model_id = None
            self._set_state(AiUiState.WORKER_READY)
        elif command in (protocol.Command.SET_IMAGE, protocol.Command.PREDICT):
            if self._image_key is not None:
                self._set_state(AiUiState.PROMPT_EDITING)
            elif self._model_id is not None:
                self._set_state(AiUiState.MODEL_READY)
        self.error.emit(
            protocol.ErrorCode.INTERNAL,
            f"{command} が時間内に完了しませんでした。Workerの状態を確認してください。",
        )

    def _on_worker_crashed(self, exit_code: int, message: str) -> None:
        _log.error("Workerクラッシュ: exit_code=%d", exit_code)
        self._discard_prediction()
        self._model_id = None
        self._image_key = None
        self._cuda_extension_loaded = False
        self._set_state(AiUiState.ERROR)
        self.worker_unavailable.emit(message)

    def _on_worker_stopped(self, exit_code: int, message: str) -> None:
        _log.info("Worker停止: %s", message)
        if self._state != AiUiState.DISABLED:
            self._set_state(AiUiState.DISABLED)

    def _on_start_failed(self, message: str) -> None:
        self._set_state(AiUiState.ERROR)
        self.worker_unavailable.emit(message)
