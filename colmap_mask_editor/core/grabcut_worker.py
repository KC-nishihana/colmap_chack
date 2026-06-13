"""
GrabCut処理をGUIスレッドから分離するQObjectベースWorker (v0.4B)

使用方法 (INITIAL):
  worker = GrabCutWorker(image, rect, options, request_id)
  thread = QThread()
  worker.moveToThread(thread)
  thread.started.connect(worker.run)
  worker.finished.connect(on_finished)
  thread.start()

使用方法 (REFINE):
  worker = GrabCutWorker(
      task_type=GrabCutTaskType.REFINE,
      session=session_copy,
      hint_strokes=strokes_copy,
      options=options,
      request_id=request_id,
  )
"""

import logging
import traceback
from enum import Enum, auto

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from core.grabcut_tool import (
    GrabCutOptions,
    GrabCutResult,
    GrabCutSession,
    create_grabcut_session,
    refine_grabcut_session,
)

_log = logging.getLogger(__name__)


class GrabCutTaskType(Enum):
    INITIAL = auto()   # 矩形から初回GrabCut → GrabCutSession生成
    REFINE = auto()    # ヒントを使ってGC_INIT_WITH_MASKで再推定


class GrabCutWorker(QObject):
    """
    GrabCut処理をバックグラウンドスレッドで実行するWorker。
    Qtウィジェットを直接操作せず、Signalのみで通知する。

    NOTE: finished は引数なし Signal() で定義する。
    Signal(object) をスレッド境界で渡すと PySide6 のバージョンによっては
    queued delivery が不安定なため、結果は self.result 属性に格納して渡す。

    INITIAL タスク:
      self.result  → GrabCutResult (後方互換)
      self.session → GrabCutSession (新規)

    REFINE タスク:
      self.result  → GrabCutSession (新しいSession)
      self.session → GrabCutSession (same as result)
    """

    finished = Signal(int)      # request_id — int はスレッド境界を安全に渡せる
    failed = Signal(str, int)   # message, request_id
    progress = Signal(str)      # 進捗メッセージ
    cancelled = Signal(int)     # request_id

    def __init__(
        self,
        image_bgr: np.ndarray = None,
        rect: tuple = None,
        options: GrabCutOptions = None,
        request_id: int = 0,
        *,
        task_type: GrabCutTaskType = None,
        session: GrabCutSession = None,
        hint_strokes: list = None,
        current_mask: np.ndarray = None,
        parent: QObject = None,
    ) -> None:
        super().__init__(parent)

        # task_type の自動判定 (後方互換)
        if task_type is None:
            task_type = GrabCutTaskType.REFINE if session is not None else GrabCutTaskType.INITIAL

        self._task_type = task_type
        self._image = image_bgr
        self._rect = rect
        self._options = options if options is not None else GrabCutOptions()
        self._request_id = request_id
        self._session = session
        self._hint_strokes = hint_strokes if hint_strokes is not None else []
        self._current_mask = current_mask
        self._cancel_requested = False

        self.result: object = None    # GrabCutResult (INITIAL) / GrabCutSession (REFINE)
        self.session: object = None   # GrabCutSession | None

    @property
    def request_id(self) -> int:
        return self._request_id

    def request_cancel(self) -> None:
        """キャンセルフラグを立てる。cv2.grabCut() 実行前後で確認する。"""
        _log.info("キャンセル要求 (request_id=%d)", self._request_id)
        self._cancel_requested = True

    def run(self) -> None:
        """QThread の start() から呼ばれるエントリポイント。"""
        _log.info("GrabCutWorker 開始 (request_id=%d, task=%s)",
                  self._request_id, self._task_type.name)
        try:
            if self._task_type == GrabCutTaskType.INITIAL:
                self._run_initial()
            elif self._task_type == GrabCutTaskType.REFINE:
                self._run_refine()
            else:
                raise ValueError(f"不明なタスク種別: {self._task_type}")

        except ValueError as e:
            _log.warning("GrabCut ValueError (request_id=%d): %s", self._request_id, e)
            self.failed.emit(str(e), self._request_id)
        except cv2.error as e:
            _log.error("GrabCut cv2.error (request_id=%d): %s", self._request_id, e)
            self.failed.emit(
                "OpenCV GrabCut処理に失敗しました。\n"
                "反復回数を下げるか、矩形範囲を変更してください。",
                self._request_id,
            )
        except MemoryError:
            _log.error("GrabCut MemoryError (request_id=%d)", self._request_id)
            self.failed.emit(
                "画像サイズが大きく、メモリを確保できませんでした。\n"
                "最大処理サイズを小さくしてください。",
                self._request_id,
            )
        except RuntimeError as e:
            _log.error("GrabCut RuntimeError (request_id=%d): %s", self._request_id, e)
            self.failed.emit(f"実行時エラーが発生しました: {e}", self._request_id)
        except Exception as e:
            _log.error(
                "GrabCut 予期しないエラー (request_id=%d):\n%s",
                self._request_id, traceback.format_exc(),
            )
            self.failed.emit(f"予期しないエラーが発生しました: {e}", self._request_id)

    # ------------------------------------------------------------------ #
    # 内部処理
    # ------------------------------------------------------------------ #

    def _run_initial(self) -> None:
        """初回GrabCut処理 (GC_INIT_WITH_RECT)。"""
        self.progress.emit("ROIを準備しています")
        if self._cancel_requested:
            _log.info("GrabCutキャンセル (ROI前, request_id=%d)", self._request_id)
            self.cancelled.emit(self._request_id)
            return

        if self._options.use_downscale:
            self.progress.emit(f"{self._options.max_processing_size}pxへ縮小しています")
        else:
            self.progress.emit("前景・背景を推定しています")

        if self._cancel_requested:
            _log.info("GrabCutキャンセル (GrabCut前, request_id=%d)", self._request_id)
            self.cancelled.emit(self._request_id)
            return

        gc_session: GrabCutSession = create_grabcut_session(
            self._image, self._rect, self._options, self._current_mask
        )

        if self._cancel_requested:
            _log.info("GrabCutキャンセル (GrabCut後, request_id=%d)", self._request_id)
            self.cancelled.emit(self._request_id)
            return

        # 後方互換のため GrabCutResult も生成
        gc_result = GrabCutResult(
            mask=gc_session.preview_mask,
            original_size=gc_session.original_size,
            roi=gc_session.roi,
            processing_size=gc_session.processing_size,
            scale=gc_session.scale,
            processing_time_sec=gc_session.processing_time_sec,
            was_downscaled=gc_session.was_downscaled,
        )

        self.result = gc_result
        self.session = gc_session

        self.progress.emit("プレビューを準備しています")
        _log.info(
            "GrabCutWorker 完了 (request_id=%d): 処理時間 %.3f秒, 縮小率 %.4f",
            self._request_id, gc_result.processing_time_sec, gc_result.scale,
        )
        self.finished.emit(self._request_id)

    def _run_refine(self) -> None:
        """ヒントを使ったGrabCut再推定 (GC_INIT_WITH_MASK)。"""
        if self._session is None:
            raise ValueError("再推定にはGrabCutSessionが必要です")

        self.progress.emit("ヒントを処理しています")
        if self._cancel_requested:
            self.cancelled.emit(self._request_id)
            return

        _log.info(
            "再推定開始: ストローク数=%d, 再推定回数=%d (request_id=%d)",
            len(self._hint_strokes), self._session.refine_count + 1, self._request_id,
        )

        self.progress.emit("GrabCut再推定を実行しています")
        if self._cancel_requested:
            self.cancelled.emit(self._request_id)
            return

        iter_count = max(1, min(20, self._options.iter_count))
        new_session: GrabCutSession = refine_grabcut_session(
            self._session, self._hint_strokes, iter_count
        )

        if self._cancel_requested:
            self.cancelled.emit(self._request_id)
            return

        self.result = new_session
        self.session = new_session

        self.progress.emit("再推定完了")
        _log.info(
            "GrabCut再推定完了 (request_id=%d): 再推定回数=%d, 処理時間=%.3f秒",
            self._request_id, new_session.refine_count, new_session.processing_time_sec,
        )
        self.finished.emit(self._request_id)
