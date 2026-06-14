"""
CUDA 検証の成功判定ロジック (純粋関数) のテスト。

torch / sam2 / CUDA を import せずに evaluate_verification() の振る舞いを検証する。
"""

from ai.cuda_verification import REQUIRED_VERIFICATION_KEYS, evaluate_verification


def _all_true() -> dict:
    return {key: True for key in REQUIRED_VERIFICATION_KEYS}


def test_all_checks_true_is_success():
    assert evaluate_verification(_all_true()) is True


def test_extension_import_only_is_failure():
    # sam2._C を import できた「だけ」では成功にしない
    result = {"cuda_extension_imported": True}
    assert evaluate_verification(result) is False


def test_kernel_not_executed_is_failure():
    result = _all_true()
    result["cuda_extension_kernel_executed"] = False
    assert evaluate_verification(result) is False


def test_model_inference_false_is_failure():
    result = _all_true()
    result["model_loaded"] = False
    assert evaluate_verification(result) is False


def test_fill_holes_false_is_failure():
    result = _all_true()
    result["fill_holes_test"] = False
    assert evaluate_verification(result) is False


def test_multimask_false_is_failure():
    result = _all_true()
    result["multimask_output"] = False
    assert evaluate_verification(result) is False


def test_missing_key_is_failure():
    # キー欠落 (None) も成功にしない
    result = _all_true()
    del result["embedding_ok"]
    assert evaluate_verification(result) is False


def test_truthy_but_not_true_is_failure():
    # is True を厳密に要求する (1 や "yes" は不可)
    result = _all_true()
    result["positive_click_ok"] = 1
    assert evaluate_verification(result) is False
