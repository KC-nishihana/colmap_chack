"""
sam2_manifest.json の整合性テスト (torch / sam2 / CUDA 不要)。

verified=true は実機フル検証済みを意味する。その場合 commit が固定 SHA であり、
verification 情報 (kernel 実行・model 推論) が揃っていることを要求する。
開発環境で実機未検証の場合は verified=false を正常状態として許可する。
"""

import json
import re
from pathlib import Path

import pytest

_MANIFEST = (
    Path(__file__).resolve().parent.parent / "sam_backend" / "sam2_manifest.json"
)

_SHA40 = re.compile(r"^[0-9a-f]{40}$")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))


def test_manifest_exists():
    assert _MANIFEST.exists(), f"マニフェストがありません: {_MANIFEST}"


def test_protocol_version_is_1(manifest):
    assert manifest.get("protocol_version") == 1


def test_verified_is_bool(manifest):
    assert isinstance(manifest.get("verified"), bool)


def test_default_model_registered(manifest):
    from ai import model_registry
    assert model_registry.has_model(manifest["default_model"])


def test_verified_true_requires_full_evidence(manifest):
    """verified=true のときのみ厳格条件を課す。false は開発状態として許可。"""
    if manifest.get("verified") is not True:
        pytest.skip("verified=false (実機未検証) は正常状態として許可")

    commit = manifest.get("commit", "")
    assert _SHA40.match(commit), f"verified=true なら commit は 40 桁 SHA: {commit!r}"

    ver = manifest.get("verification")
    assert isinstance(ver, dict), "verified=true なら verification 情報が必要"
    assert ver.get("cuda_extension_imported") is True
    assert ver.get("cuda_extension_kernel_executed") is True
    assert ver.get("model_inference") is True
