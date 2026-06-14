"""
テスト用の偽 SAM Worker。

本物の worker_main と同じ JSON Lines プロトコルを話すが、torch / sam2 / CUDA を
一切使わない。QProcess 統合テスト (process_manager / ai_session) で使用する。

挙動は環境変数 FAKE_SAM_MODE で切り替える:
  normal             : 正常系 (3候補NPZを書き出す)
  slow               : 各応答前に少し sleep
  crash              : predict で異常終了 (exit 1)
  invalid_json       : stdout に非JSON行を混ぜる
  stderr_noise       : stderr に大量出力 (stdout はJSONのみ)
  timeout            : predict に応答しない
  missing_result     : prediction_ready が存在しない result_path を指す
  corrupt_npz        : 壊れたNPZを書いて参照する
  cuda_extension_false: hello で cuda_extension_loaded=False
  oom                : predict で CUDA_OOM エラー応答

  伝播 (V0.7):
  propagation_normal      : 全フレーム frame_ready + completed
  propagation_slow        : 各フレーム前に sleep
  propagation_warning     : 一部フレームに warning_codes を付与
  propagation_frame_failure: 1フレーム後に PREDICT_FAILED で失敗 (Workerは維持)
  propagation_cancel      : cancel を受けたら CANCELLED (完成済み保持)
  propagation_crash       : 伝播中に異常終了 (exit 1)
  propagation_invalid_event: stdout に非JSON行を混ぜる
  propagation_missing_result: frame_ready が存在しない result を指す
  propagation_corrupt_mask: 壊れたPNGを書いて参照する
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import numpy as np  # noqa: E402

from ai import protocol, runtime_paths  # noqa: E402
from ai.propagation_protocol import (  # noqa: E402
    PropagationCommand,
    PropagationErrorCode,
    PropagationEvent,
    make_job_error,
    make_job_event,
)
from sam_backend import result_writer  # noqa: E402

MODE = os.environ.get("FAKE_SAM_MODE", "normal")

_REAL_STDOUT = sys.stdout
_STDOUT_LOCK = threading.Lock()


def send(msg: dict) -> None:
    with _STDOUT_LOCK:
        _REAL_STDOUT.write(protocol.encode_line(msg) + "\n")
        _REAL_STDOUT.flush()


def err(code: str, message: str, request_id=None, **fields) -> None:
    send(protocol.make_error(code, message, request_id, **fields))


def maybe_sleep(sec: float = 0.3) -> None:
    if MODE == "slow":
        time.sleep(sec)


def write_fake_npz(request_id: int, image_key: str, h: int = 64, w: int = 80) -> str:
    masks = np.zeros((3, h, w), dtype=np.uint8)
    masks[0, 10:50, 10:60] = 255
    masks[1, 5:55, 5:70] = 255
    masks[2, 20:40, 20:50] = 255
    scores = np.array([0.95, 0.88, 0.77], dtype=np.float32)
    return result_writer.write_result_npz(masks, scores, request_id, image_key)


def write_corrupt_npz(request_id: int) -> str:
    d = runtime_paths.get_runtime_dir(create=True)
    p = d / f"sam_result_{request_id}.npz"
    p.write_bytes(b"this is not a valid npz file")
    return str(p)


_IMAGE_KEY = None
_MODEL_LOADED = False


class _FakeProp(threading.Thread):
    """偽の伝播ジョブ。frame_ready を逐次送り、pause/cancel を尊重する。"""

    def __init__(self, job_id, frames, job_dir):
        super().__init__(daemon=True)
        self.job_id = job_id
        self._frames = frames
        self._job_dir = Path(job_dir)
        self._pause = threading.Event()
        self._cancel = threading.Event()
        self.processed = 0
        self.total = len(frames)

    def pause(self):
        self._pause.set()

    def resume(self):
        self._pause.clear()

    def cancel(self):
        self._cancel.set()
        self._pause.clear()

    @property
    def is_paused(self):
        return self._pause.is_set()

    def _write_result(self, frame_index: int) -> str:
        results = self._job_dir / "results"
        results.mkdir(parents=True, exist_ok=True)
        dest = results / f"{frame_index:06d}.png"
        if MODE == "propagation_missing_result":
            return str(results / f"missing_{frame_index:06d}.png")
        if MODE == "propagation_corrupt_mask":
            dest.write_bytes(b"not a png")
            return str(dest)
        import cv2
        m = np.zeros((48, 64), np.uint8)
        m[5:30, 5 + frame_index:25 + frame_index] = 255
        ok, buf = cv2.imencode(".png", m)
        buf.tofile(str(dest))
        return str(dest)

    def run(self):
        try:
            for i, fr in enumerate(self._frames):
                if self._cancel.is_set():
                    send(make_job_event(PropagationEvent.CANCELLED, self.job_id,
                                        completed_count=self.processed))
                    return
                if self._pause.is_set():
                    send(make_job_event(PropagationEvent.PAUSED, self.job_id))
                    while self._pause.is_set():
                        if self._cancel.is_set():
                            send(make_job_event(PropagationEvent.CANCELLED, self.job_id,
                                                completed_count=self.processed))
                            return
                        time.sleep(0.02)
                    send(make_job_event(PropagationEvent.RESUMED, self.job_id))

                if MODE == "propagation_slow":
                    time.sleep(0.1)
                if MODE == "propagation_crash":
                    os._exit(1)

                fidx = int(fr["frame_index"])
                rp = self._write_result(fidx)
                warns = ["LOW_IOU"] if (MODE == "propagation_warning" and i == 1) else []
                send(make_job_event(PropagationEvent.FRAME_READY, self.job_id,
                                    frame_index=fidx, entry_key=fr.get("entry_key", str(fidx)),
                                    result_mask_path=rp, foreground_ratio=0.1,
                                    warning_codes=warns, is_reference=(i == 0)))
                self.processed += 1
                send(make_job_event(PropagationEvent.PROGRESS, self.job_id,
                                    processed=self.processed, total=self.total))

                if MODE == "propagation_frame_failure" and i == 0:
                    send(make_job_error(PropagationErrorCode.PREDICT_FAILED,
                                        "frame failure (fake)", job_id=self.job_id))
                    return
            send(make_job_event(PropagationEvent.COMPLETED, self.job_id,
                                completed_count=self.processed,
                                warning_count=(1 if MODE == "propagation_warning" else 0)))
        except Exception as e:  # noqa: BLE001
            send(make_job_error(PropagationErrorCode.PREDICT_FAILED, str(e), job_id=self.job_id))


_PROP: _FakeProp | None = None


def _handle_propagation(command, msg, rid) -> None:
    global _PROP
    if command == PropagationCommand.START:
        if _PROP is not None and _PROP.is_alive():
            send(make_job_error(PropagationErrorCode.BUSY, "実行中です", request_id=rid))
            return
        frames = msg.get("frames") or []
        if len(frames) < 2:
            send(make_job_error(PropagationErrorCode.INVALID_SEQUENCE, "2枚未満", request_id=rid))
            return
        if MODE == "propagation_invalid_event":
            _REAL_STDOUT.write("NOT JSON PROP EVENT\n")
            _REAL_STDOUT.flush()
        job_id = "prop-" + uuid.uuid4().hex[:8]
        job_dir = runtime_paths.get_propagation_job_dir(job_id)
        _PROP = _FakeProp(job_id, frames, job_dir)
        send(make_job_event(PropagationEvent.STARTED, job_id, request_id=rid,
                            frame_count=len(frames)))
        _PROP.start()
        return

    r = _PROP
    job_id = msg.get("job_id")
    if r is None or not r.is_alive() or (job_id and job_id != r.job_id):
        send(make_job_error(PropagationErrorCode.NOT_FOUND, "ジョブなし",
                            job_id=job_id, request_id=rid))
        return
    if command == PropagationCommand.PAUSE:
        r.pause(); send(make_job_event(PropagationEvent.PAUSED, r.job_id, request_id=rid))
    elif command == PropagationCommand.RESUME:
        r.resume(); send(make_job_event(PropagationEvent.RESUMED, r.job_id, request_id=rid))
    elif command == PropagationCommand.CANCEL:
        r.cancel(); send(make_job_event(PropagationEvent.CANCELLING, r.job_id, request_id=rid))
    elif command == PropagationCommand.STATUS:
        send(make_job_event(PropagationEvent.PROGRESS, r.job_id, request_id=rid,
                            processed=r.processed, total=r.total, paused=r.is_paused))
    elif command == PropagationCommand.RELEASE:
        r.cancel(); send(make_job_event(PropagationEvent.RELEASED, r.job_id, request_id=rid))


def handle(msg: dict) -> bool:
    """1コマンド処理。継続なら True, shutdown なら False。"""
    global _IMAGE_KEY, _MODEL_LOADED
    command = msg.get("command")
    rid = msg.get("request_id")

    if command == protocol.Command.HELLO:
        maybe_sleep()
        cuda_ext = MODE != "cuda_extension_false"
        payload = dict(
            python_executable=sys.executable,
            python_version=sys.version.split()[0],
            torch_version="fake-2.0",
            torch_cuda_version="12.1",
            cuda_available=True,
            cuda_extension_loaded=cuda_ext,
            gpu_name="Fake RTX 4090",
            compute_capability="8.9",
            sam2_commit="fake-commit",
        )
        if not cuda_ext:
            payload["message"] = "SAM 2 CUDA拡張を読み込めませんでした。"
        send(protocol.make_event(protocol.Event.READY, rid, **payload))

    elif command == protocol.Command.HEALTH:
        if MODE == "cuda_extension_false":
            err(protocol.ErrorCode.CUDA_EXTENSION_UNAVAILABLE,
                "SAM 2 CUDA拡張を読み込めませんでした。", rid)
        else:
            send(protocol.make_event(protocol.Event.HEALTH_RESULT, rid,
                                     cuda_available=True, cuda_extension_loaded=True,
                                     model_loaded=_MODEL_LOADED, vram_allocated_mb=100))

    elif command == protocol.Command.LOAD_MODEL:
        maybe_sleep()
        _MODEL_LOADED = True
        send(protocol.make_event(protocol.Event.MODEL_LOADED, rid,
                                 model_id=msg.get("model_id"), device=msg.get("device"),
                                 precision=msg.get("precision"), vram_allocated_mb=1234))

    elif command == protocol.Command.UNLOAD_MODEL:
        _MODEL_LOADED = False
        send(protocol.make_event(protocol.Event.MODEL_UNLOADED, rid))

    elif command == protocol.Command.SET_IMAGE:
        maybe_sleep()
        _IMAGE_KEY = uuid.uuid4().hex
        send(protocol.make_event(protocol.Event.IMAGE_READY, rid,
                                 width=80, height=64, embedding_time_sec=0.5,
                                 image_key=_IMAGE_KEY))

    elif command == protocol.Command.PREDICT:
        if MODE == "crash":
            os._exit(1)
        if MODE == "timeout":
            return True  # 応答しない
        if MODE == "oom":
            err(protocol.ErrorCode.CUDA_OOM, "CUDA メモリ不足で推論に失敗しました", rid)
            return True
        if MODE == "invalid_json":
            _REAL_STDOUT.write("THIS IS NOT JSON\n")
            _REAL_STDOUT.flush()
        maybe_sleep()
        image_key = msg.get("image_key", _IMAGE_KEY)
        if MODE == "missing_result":
            result_path = str(runtime_paths.get_runtime_dir() / f"nonexistent_{rid}.npz")
        elif MODE == "corrupt_npz":
            result_path = write_corrupt_npz(int(rid))
        else:
            result_path = write_fake_npz(int(rid), image_key)
        send(protocol.make_event(protocol.Event.PREDICTION_READY, rid,
                                 image_key=image_key, result_path=result_path,
                                 mask_count=3, scores=[0.95, 0.88, 0.77],
                                 width=80, height=64, prediction_time_sec=0.08,
                                 vram_allocated_mb=2000))

    elif command == protocol.Command.RELEASE_IMAGE:
        _IMAGE_KEY = None
        send(protocol.make_event(protocol.Event.IMAGE_RELEASED, rid))

    elif command == protocol.Command.CLEAR_CUDA_CACHE:
        send(protocol.make_event(protocol.Event.CUDA_CACHE_CLEARED, rid))

    elif command == protocol.Command.SHUTDOWN:
        if _PROP is not None and _PROP.is_alive():
            _PROP.cancel()
        send(protocol.make_event(protocol.Event.SHUTTING_DOWN, rid))
        return False

    elif command in PropagationCommand.ALL:
        _handle_propagation(command, msg, rid)

    else:
        err(protocol.ErrorCode.BAD_REQUEST, f"不明なコマンド: {command}", rid)

    return True


def main() -> int:
    if MODE == "stderr_noise":
        for i in range(50):
            sys.stderr.write(f"[fake worker noise] line {i}\n")
        sys.stderr.flush()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            err(protocol.ErrorCode.BAD_REQUEST, "不正なJSON")
            continue
        if not handle(msg):
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
