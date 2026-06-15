"""
V0.9: 完全被覆 partition 生成の CPU 専用 Worker (QProcess 子プロセス)。

torch / sam2 / PySide6 を import しない。Python 標準ライブラリ + NumPy + OpenCV のみ。
stdin から JSON Lines のコマンドを受け、stdout へ JSON Lines のイベントを返す。
ログ・トレースバックは stderr へ (stdout は JSON 専用)。

コマンド: partition_build_start / partition_build_cancel / partition_build_status /
          partition_validate / partition_release / shutdown
イベント: partition_build_started / partition_stage_changed / partition_progress /
          partition_build_completed / partition_build_cancelled /
          partition_build_failed / partition_validated / partition_released

重い分割・統合は GUI スレッドではなくこの CPU プロセスで実行する。キャンセルは
処理ステージのチェックポイントで受け付け、正常な既存 partition や V0.8 の
segments.npz、レビュー状態を変更しない (保存前に中断する)。
"""

from __future__ import annotations

import queue
import sys
import threading
import traceback

# 重要: torch / sam2 / PySide6 は import しない。
# cv2/numpy/builder は reader スレッド起動前に「メインスレッドで」import しておく。
# (起動後に別スレッドが stdin を読みながら初回 import すると import lock と
#  cv2 の初期化が干渉してデッドロックし得る)。
from ai import protocol, partition_npz
from ai.partition_protocol import (
    PartitionCommand,
    PartitionErrorCode,
    PartitionEvent,
    make_job_error,
    make_job_event,
)
from partition_backend import partition_builder as builder
from partition_backend.slic_backend import SlicUnavailableError


class _Out:
    """stdout への JSON Lines 出力 (1 スレッドからのみ呼ぶ)。"""

    def __init__(self, stream=None):
        self._s = stream if stream is not None else sys.stdout

    def send(self, msg: dict) -> None:
        self._s.write(protocol.encode_line(msg) + "\n")
        self._s.flush()


def _log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


class PartitionWorker:
    """単一画像ずつ partition を生成する CPU Worker。"""

    def __init__(self, out: _Out):
        self.out = out
        self._cancel = threading.Event()
        self._active_job: str | None = None

    # ---- コマンド処理 ---- #
    def handle(self, msg: dict) -> bool:
        """1 コマンドを処理する。継続するなら True、shutdown なら False。"""
        command = msg.get("command")
        rid = msg.get("request_id")
        if command == PartitionCommand.SHUTDOWN:
            return False
        if command == PartitionCommand.BUILD_CANCEL:
            self._cancel.set()
            return True
        if command == PartitionCommand.BUILD_STATUS:
            self.out.send(protocol.make_event(
                PartitionEvent.STAGE_CHANGED, rid,
                job_id=self._active_job,
                busy=self._active_job is not None))
            return True
        if command == PartitionCommand.VALIDATE:
            self._do_validate(msg, rid)
            return True
        if command == PartitionCommand.RELEASE:
            self._active_job = None
            self.out.send(protocol.make_event(PartitionEvent.RELEASED, rid))
            return True
        if command == PartitionCommand.BUILD_START:
            self._do_build(msg, rid)
            return True
        self.out.send(protocol.make_error(
            PartitionErrorCode.INTERNAL, f"不明なコマンド: {command!r}", rid))
        return True

    def _do_validate(self, msg: dict, rid) -> None:
        path = msg.get("partition_path")
        try:
            partition_npz.verify_partition_npz(path)
            self.out.send(protocol.make_event(
                PartitionEvent.VALIDATED, rid, partition_path=path, valid=True))
        except Exception as e:  # noqa: BLE001
            self.out.send(make_job_error(
                PartitionErrorCode.CACHE_CORRUPT, str(e), request_id=rid,
                partition_path=path, valid=False))

    def _do_build(self, msg: dict, rid) -> None:
        job_id = str(msg.get("job_id", ""))
        if self._active_job is not None:
            self.out.send(make_job_error(
                PartitionErrorCode.BUSY, "別の partition 生成が進行中です",
                job_id=job_id, request_id=rid))
            return
        self._active_job = job_id
        self._cancel.clear()

        image_path = msg.get("image_path")
        image_key = msg.get("image_key", "")
        output_dir = msg.get("output_dir")
        settings = msg.get("settings", {})
        segments_path = msg.get("segments_path")

        self.out.send(make_job_event(
            PartitionEvent.BUILD_STARTED, job_id, rid, image_key=image_key))

        last_stage = {"name": None}

        def on_progress(stage, frac, info):
            if stage != last_stage["name"]:
                last_stage["name"] = stage
                self.out.send(make_job_event(
                    PartitionEvent.STAGE_CHANGED, job_id, stage=stage))
            self.out.send(make_job_event(
                PartitionEvent.PROGRESS, job_id, stage=stage,
                fraction=float(frac), **info))

        try:
            manifest = builder.build_partition(
                image_path, image_key=image_key, output_dir=output_dir,
                settings=settings, segments_path=segments_path,
                progress=on_progress, should_cancel=self._cancel.is_set)
            self.out.send(make_job_event(
                PartitionEvent.BUILD_COMPLETED, job_id, image_key=image_key,
                leaf_count=manifest["leaf_count"], node_count=manifest["node_count"],
                root_id=manifest["root_id"],
                coverage_ratio=manifest["coverage"]["coverage_ratio"],
                backend_used=manifest["backend_used"]))
        except builder.PartitionCancelled:
            self.out.send(make_job_event(
                PartitionEvent.BUILD_CANCELLED, job_id, image_key=image_key))
        except SlicUnavailableError as e:
            self.out.send(make_job_error(
                PartitionErrorCode.SLIC_UNAVAILABLE, str(e),
                job_id=job_id, request_id=rid))
        except FileNotFoundError as e:
            self.out.send(make_job_error(
                PartitionErrorCode.IMAGE_LOAD_FAILED, str(e),
                job_id=job_id, request_id=rid))
        except Exception as e:  # noqa: BLE001
            _log("partition build failed:\n" + traceback.format_exc())
            self.out.send(make_job_error(
                PartitionErrorCode.INTERNAL, str(e), job_id=job_id, request_id=rid))
        finally:
            self._active_job = None


def run(stdin=None, stdout=None) -> int:
    """Worker のメインループ。reader スレッドで cancel を並行受信する。"""
    stdin = stdin if stdin is not None else sys.stdin
    # stdin/stdout を utf-8 に固定 (日本語パスを含むコマンドを確実に扱う)。
    for stream in (stdin, stdout if stdout is not None else sys.stdout):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    out = _Out(stdout)
    worker = PartitionWorker(out)

    cmd_queue: "queue.Queue[dict | None]" = queue.Queue()
    decoder = protocol.JsonLineDecoder()

    def reader():
        # 注意: `for line in stdin` は read-ahead バッファで 1 行ごとに yield せず
        # デッドロックする。readline を直接回して 1 行ずつ即時処理する。
        for line in iter(stdin.readline, ""):
            for parsed in decoder.feed(line):
                if not parsed.ok:
                    _log(f"不正な行を無視: {parsed.error}")
                    continue
                obj = parsed.obj
                # cancel は即座に反映 (build 実行中でも reader スレッドが set する)
                if obj.get("command") == PartitionCommand.BUILD_CANCEL:
                    worker._cancel.set()
                cmd_queue.put(obj)
        cmd_queue.put(None)  # EOF

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    out.send(protocol.make_event(PartitionEvent.RELEASED, None, ready=True))

    while True:
        obj = cmd_queue.get()
        if obj is None:
            break
        try:
            if not worker.handle(obj):
                break
        except Exception:  # noqa: BLE001
            _log("worker handle error:\n" + traceback.format_exc())
    return 0


if __name__ == "__main__":
    sys.exit(run())
