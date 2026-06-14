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
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import numpy as np  # noqa: E402

from ai import protocol, runtime_paths  # noqa: E402
from sam_backend import result_writer  # noqa: E402

MODE = os.environ.get("FAKE_SAM_MODE", "normal")

_REAL_STDOUT = sys.stdout


def send(msg: dict) -> None:
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
        send(protocol.make_event(protocol.Event.SHUTTING_DOWN, rid))
        return False

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
