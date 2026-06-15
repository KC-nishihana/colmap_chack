"""V0.8: AMG バッチランナーのテスト (torch 不要・Fake generator)。"""

import threading

import numpy as np
import pytest

from ai import amg_manifest as M, amg_npz
from ai import amg_rle
from ai.amg_protocol import AmgEvent
from sam_backend.amg_batch_runner import AmgBatchRunner


MODEL = {"model_id": "sam2.1_hiera_small", "sam2_commit": "abc", "checkpoint_fingerprint": "fp"}


class _Result:
    def __init__(self, annotations, ppb=64, retries=0, peak=123):
        self.annotations = annotations
        self.points_per_batch_used = ppb
        self.oom_retries = retries
        self.peak_vram_mb = peak
        self.warnings = []


def _ann(m):
    h, w = m.shape
    ys, xs = np.where(m > 0)
    return {
        "segmentation": {"size": [h, w], "counts": amg_rle.encode_mask(m)},
        "area": int((m > 0).sum()),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
        "predicted_iou": 0.9, "stability_score": 0.95,
        "point_coords": [[float(xs.mean()), float(ys.mean())]], "crop_box": [0, 0, w, h],
    }


def _fake_load(_path):
    h, w = 16, 20
    return np.zeros((h, w, 3), np.uint8), w, h


def _fake_generate(_rgb, _settings):
    h, w = 16, 20
    a = np.zeros((h, w), np.uint8); a[1:8, 1:10] = 1
    b = np.zeros((h, w), np.uint8); b[9:15, 11:19] = 1
    return _Result([_ann(a), _ann(b)])


class _Collector:
    def __init__(self):
        self.events = []
        self._lock = threading.Lock()

    def __call__(self, msg):
        with self._lock:
            self.events.append(msg)

    def of(self, event):
        return [e for e in self.events if e.get("event") == event]


def _make_images(tmp_path, n):
    images = []
    for i in range(n):
        p = tmp_path / "src" / f"IMG_{i:03d}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\xff\xd8\xff" + bytes([i]) * 300)
        images.append({"image_key": f"IMG_{i:03d}.jpg", "source_path": str(p)})
    return images


def _run(tmp_path, images, *, generate_fn=_fake_generate, force=False, collector=None):
    col = collector or _Collector()
    runner = AmgBatchRunner(
        job_id="amg-test", project_root=str(tmp_path), images=images,
        settings=M.preset_settings("fast"), preset="fast", model=MODEL,
        generate_fn=generate_fn, load_image_fn=_fake_load, send_cb=col, force=force,
    )
    runner.start()
    runner.join(timeout=20)
    assert not runner.is_alive()
    return col, runner


def test_batch_normal(tmp_path):
    images = _make_images(tmp_path, 3)
    col, runner = _run(tmp_path, images)
    assert len(col.of(AmgEvent.IMAGE_COMPLETED)) == 3
    assert len(col.of(AmgEvent.BATCH_COMPLETED)) == 1
    assert runner.status_snapshot()["succeeded"] == 3
    # NPZ + manifest が各画像に存在し検証可能
    for item in images:
        cdir = M.cache_dir_for(tmp_path, item["image_key"])
        amg_npz.verify_segments_npz(cdir / "segments.npz")
        man = M.read_json(cdir / "manifest.json")
        assert man["segment_count"] == 2
        assert man["status"] == "ready"
    # batch_manifest
    batch = M.read_json(tmp_path / "segmentation_cache" / "batch_manifest.json")
    assert all(v["status"] == "ready" for v in batch["images"].values())


def test_skip_on_reuse(tmp_path):
    images = _make_images(tmp_path, 2)
    _run(tmp_path, images)
    # 2 回目は全 skip (再利用)
    col2, runner2 = _run(tmp_path, images)
    assert len(col2.of(AmgEvent.IMAGE_SKIPPED)) == 2
    assert len(col2.of(AmgEvent.IMAGE_COMPLETED)) == 0
    assert runner2.status_snapshot()["reused"] == 2


def test_force_reanalyze(tmp_path):
    images = _make_images(tmp_path, 1)
    _run(tmp_path, images)
    col2 = _Collector()
    runner = AmgBatchRunner(
        job_id="amg-f", project_root=str(tmp_path), images=images,
        settings=M.preset_settings("fast"), preset="fast", model=MODEL,
        generate_fn=_fake_generate, load_image_fn=_fake_load, send_cb=col2, force=True,
    )
    runner.start(); runner.join(timeout=20)
    assert len(col2.of(AmgEvent.IMAGE_COMPLETED)) == 1  # force で再解析
    assert len(col2.of(AmgEvent.IMAGE_SKIPPED)) == 0


def test_failure_isolated(tmp_path):
    images = _make_images(tmp_path, 3)
    calls = {"n": 0}

    def flaky(rgb, settings):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom on image 2")
        return _fake_generate(rgb, settings)

    col, runner = _run(tmp_path, images, generate_fn=flaky)
    assert len(col.of(AmgEvent.IMAGE_FAILED)) == 1
    assert len(col.of(AmgEvent.IMAGE_COMPLETED)) == 2
    assert len(col.of(AmgEvent.BATCH_COMPLETED)) == 1  # バッチ自体は完了
    snap = runner.status_snapshot()
    assert snap["failed"] == 1 and snap["succeeded"] == 2
    # 失敗画像は batch_manifest で failed
    batch = M.read_json(tmp_path / "segmentation_cache" / "batch_manifest.json")
    statuses = sorted(v["status"] for v in batch["images"].values())
    assert statuses == ["failed", "ready", "ready"]


def test_auto_points_per_batch_caps_high_res(monkeypatch):
    """高解像では points_per_batch を自動縮小し、低解像では維持する (結果は不変)。"""
    from sam_backend.sam2_amg_manager import auto_points_per_batch
    # FHD: 予算内なので requested を維持
    assert auto_points_per_batch(64, 1080, 1920) == 64
    # 8K相当: 大幅縮小される
    ppb_8k = auto_points_per_batch(64, 4320, 7680)
    assert 1 <= ppb_8k < 64
    # requested を超えない・最小1
    assert auto_points_per_batch(8, 4320, 7680) <= 8
    assert auto_points_per_batch(64, 100000, 100000) == 1


def test_auto_points_per_batch_env_budget(monkeypatch):
    from sam_backend import sam2_amg_manager as M2
    monkeypatch.setenv("AMG_MASK_MEM_BUDGET_MB", "8000")  # 予算を増やす
    big = M2.auto_points_per_batch(64, 4320, 7680)
    monkeypatch.setenv("AMG_MASK_MEM_BUDGET_MB", "500")   # 予算を減らす
    small = M2.auto_points_per_batch(64, 4320, 7680)
    assert big > small


def test_periodic_reclaim_fires_during_batch(tmp_path):
    """進むほど遅くなる対策: 一定枚数ごとに reclaim_fn が呼ばれる (終了時だけではない)。"""
    images = _make_images(tmp_path, 5)
    calls = {"n": 0}

    def reclaim():
        calls["n"] += 1

    col = _Collector()
    runner = AmgBatchRunner(
        job_id="amg-r", project_root=str(tmp_path), images=images,
        settings=M.preset_settings("fast"), preset="fast", model=MODEL,
        generate_fn=_fake_generate, load_image_fn=_fake_load, send_cb=col,
        reclaim_fn=reclaim, reclaim_interval=2,
    )
    runner.start(); runner.join(timeout=20)
    # interval=2 で 5 枚 -> 周期回収 2 回 (2枚目,4枚目) + 終了時 1 回 = 3 回以上
    assert calls["n"] >= 3, calls["n"]


def test_no_reclaim_when_fn_none(tmp_path):
    """reclaim_fn 未指定 (テスト用) でも例外なく完走する。"""
    images = _make_images(tmp_path, 3)
    col, runner = _run(tmp_path, images)
    assert runner.status_snapshot()["succeeded"] == 3


def test_oom_retry_recorded(tmp_path):
    images = _make_images(tmp_path, 1)

    def gen_oom(rgb, settings):
        h, w = 16, 20
        a = np.zeros((h, w), np.uint8); a[1:8, 1:10] = 1
        return _Result([_ann(a)], ppb=16, retries=2)

    col, runner = _run(tmp_path, images, generate_fn=gen_oom)
    cdir = M.cache_dir_for(tmp_path, images[0]["image_key"])
    man = M.read_json(cdir / "manifest.json")
    # generator は要求値のまま (キャッシュ判定の基準)。実効値は generator_effective へ分離。
    assert man["generator"]["points_per_batch"] == 64        # 要求値
    assert man["generator_effective"]["points_per_batch"] == 16   # 自動縮小された実効値
    assert man["generator_effective"]["oom_retries"] == 2
    # settings_hash は要求値で計算され、同設定の再解析でキャッシュ再利用できる
    assert man["settings_hash"] == M.settings_hash(M.preset_settings("fast"), MODEL)
    assert any("OOM" in w for w in man["warnings"])


def test_downscaled_run_is_reusable(tmp_path):
    """高解像度で points_per_batch を自動縮小して解析した画像も、同じ要求設定なら
    次回起動で REUSABLE 判定になる (effective を記録して stale 化させない)。"""
    from ai import amg_cache

    images = _make_images(tmp_path, 1)

    def gen_downscaled(rgb, settings):
        h, w = 16, 20
        a = np.zeros((h, w), np.uint8); a[1:8, 1:10] = 1
        return _Result([_ann(a)], ppb=4)  # 要求 64 -> 実効 4 に縮小

    _run(tmp_path, images, generate_fn=gen_downscaled)
    cdir = M.cache_dir_for(tmp_path, images[0]["image_key"])
    chk = amg_cache.evaluate_cache(
        cdir, source_path=images[0]["source_path"],
        model=MODEL, generator=M.preset_settings("fast"),
    )
    assert chk.state == amg_cache.REUSABLE, chk.reason


def test_cancel_keeps_completed(tmp_path):
    images = _make_images(tmp_path, 6)
    col = _Collector()
    started = threading.Event()

    def slow(rgb, settings):
        started.set()
        import time as _t
        _t.sleep(0.15)
        return _fake_generate(rgb, settings)

    runner = AmgBatchRunner(
        job_id="amg-c", project_root=str(tmp_path), images=images,
        settings=M.preset_settings("fast"), preset="fast", model=MODEL,
        generate_fn=slow, load_image_fn=_fake_load, send_cb=col,
    )
    runner.start()
    started.wait(timeout=5)
    runner.cancel()
    runner.join(timeout=20)
    assert len(col.of(AmgEvent.BATCH_CANCELLED)) == 1
    # 完成済みの結果は保持される (途中までの IMAGE_COMPLETED は NPZ が残る)
    completed = col.of(AmgEvent.IMAGE_COMPLETED)
    for ev in completed:
        cdir = M.cache_dir_for(tmp_path, ev["image_key"])
        amg_npz.verify_segments_npz(cdir / "segments.npz")
    assert len(completed) < 6  # 全部は終わっていない
