"""
Phase 1: Windows CUDA 環境診断 (環境を変更せず診断のみ)。

使い方:
    & "C:\\ProgramData\\Anaconda3\\Scripts\\conda.exe" run -p "C:\\conda-envs\\colmap_mask_editor" ^
        python colmap_mask_editor/scripts/check_sam2_cuda_environment.py

結果はコンソールへ表示し、logs/sam2_environment_report.json へも保存する。

終了コード:
    0: CUDA拡張をビルド・実行できる可能性が高い
    1: 必須条件不足 (PyTorch CUDA 不可 / CUDA_HOME 不明 / nvcc / cl.exe 無し 等)
    2: バージョン不整合 (torch.version.cuda と nvcc のメジャー/マイナー不一致)
    3: CUDA拡張ロード失敗 (sam2 はあるが sam2._C を import できない)

特に区別する3つの「CUDAバージョン」:
    - nvidia-smi が表示するドライバー対応 CUDA Version
    - torch.version.cuda  (PyTorch がビルドされた CUDA)
    - nvcc --version      (ローカル CUDA Toolkit)
CUDA拡張のコンパイル判断には torch.version.cuda と nvcc を使う (ドライバー表示は参考)。
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return -1, "", str(e)


def _parse_cuda_major_minor(version: str | None) -> tuple[int, int] | None:
    if not version:
        return None
    m = re.search(r"(\d+)\.(\d+)", version)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def collect() -> dict:
    report: dict = {}

    report["python_executable"] = sys.executable
    report["python_version"] = sys.version.split()[0]
    report["os"] = f"{platform.system()} {platform.release()} ({platform.version()})"
    report["is_64bit"] = (struct.calcsize("P") * 8 == 64)

    # --- PyTorch ---
    torch_info: dict = {"installed": False}
    try:
        import torch
        torch_info["installed"] = True
        torch_info["version"] = torch.__version__
        torch_info["cuda_available"] = bool(torch.cuda.is_available())
        torch_info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        try:
            import torchvision
            torch_info["torchvision_version"] = torchvision.__version__
        except Exception:
            torch_info["torchvision_version"] = None
        if torch.cuda.is_available():
            torch_info["gpu_name"] = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            torch_info["compute_capability"] = f"{cc[0]}.{cc[1]}"
    except Exception as e:
        torch_info["error"] = repr(e)
    report["torch"] = torch_info

    # --- CUDA Toolkit / nvcc ---
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    report["cuda_home"] = cuda_home
    nvcc = shutil.which("nvcc")
    if not nvcc and cuda_home:
        cand = Path(cuda_home) / "bin" / "nvcc.exe"
        if cand.exists():
            nvcc = str(cand)
    report["nvcc_path"] = nvcc
    nvcc_version = None
    if nvcc:
        rc, out, _err = _run([nvcc, "--version"])
        if rc == 0:
            m = re.search(r"release (\d+\.\d+)", out)
            nvcc_version = m.group(1) if m else out
    report["nvcc_version"] = nvcc_version

    # --- nvidia-smi (ドライバー対応 CUDA) ---
    smi = shutil.which("nvidia-smi")
    smi_cuda = None
    if smi:
        rc, out, _err = _run([smi])
        if rc == 0:
            m = re.search(r"CUDA Version:\s*(\d+\.\d+)", out)
            smi_cuda = m.group(1) if m else None
    report["nvidia_smi_cuda_version"] = smi_cuda

    # --- MSVC cl.exe ---
    cl = shutil.which("cl")
    report["cl_path"] = cl
    msvc_version = None
    if cl:
        rc, out, err = _run([cl])
        text = (out + "\n" + err)
        m = re.search(r"Version (\d+\.\d+\.\d+)", text)
        msvc_version = m.group(1) if m else None
    report["msvc_version"] = msvc_version

    # --- ninja ---
    report["ninja_path"] = shutil.which("ninja")

    # --- Visual Studio 2022 Build Tools ---
    vs_found = False
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / \
        "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.exists():
        rc, out, _err = _run([str(vswhere), "-products", "*", "-property", "displayName"])
        if rc == 0 and "2022" in out:
            vs_found = True
        report["visual_studio"] = out if rc == 0 else None
    report["visual_studio_2022_buildtools"] = vs_found

    # --- SAM 2 ---
    sam2_info: dict = {"installed": False, "cuda_extension": False}
    try:
        import sam2
        sam2_info["installed"] = True
        sam2_info["path"] = getattr(sam2, "__file__", None)
        try:
            import sam2._C  # noqa: F401
            sam2_info["cuda_extension"] = True
        except Exception as e:
            sam2_info["cuda_extension_error"] = repr(e)
    except Exception as e:
        sam2_info["error"] = repr(e)
    report["sam2"] = sam2_info

    return report


def evaluate(report: dict) -> tuple[int, list[str]]:
    """終了コードと理由メッセージを返す。"""
    msgs: list[str] = []

    torch_info = report.get("torch", {})
    if not torch_info.get("installed"):
        msgs.append("PyTorch が見つかりません。CUDA版PyTorchを先に導入してください。")
        return 1, msgs
    if not torch_info.get("cuda_available"):
        msgs.append("torch.cuda.is_available() が False です。CUDA版PyTorchを確認してください。")
        return 1, msgs
    if not report.get("cuda_home"):
        msgs.append("CUDA_HOME / CUDA_PATH が設定されていません。CUDA Toolkit を確認してください。")
        return 1, msgs
    if not report.get("nvcc_path"):
        msgs.append("nvcc が見つかりません。CUDA Toolkit (nvcc) を導入してください。")
        return 1, msgs
    if not report.get("cl_path"):
        msgs.append("cl.exe が見つかりません。VS2022 Build Tools の x64 開発者シェルで実行してください。")
        return 1, msgs

    # バージョン整合 (torch.version.cuda vs nvcc)
    torch_cuda = _parse_cuda_major_minor(torch_info.get("torch_cuda_version"))
    nvcc_cuda = _parse_cuda_major_minor(report.get("nvcc_version"))
    if torch_cuda and nvcc_cuda and torch_cuda != nvcc_cuda:
        msgs.append(
            f"バージョン不整合: torch.version.cuda={torch_info.get('torch_cuda_version')} と "
            f"nvcc={report.get('nvcc_version')} のメジャー/マイナーが一致しません。"
            "勝手に別バージョンを入れず、両者を一致させてください。"
        )
        return 2, msgs

    sam2_info = report.get("sam2", {})
    if sam2_info.get("installed") and not sam2_info.get("cuda_extension"):
        msgs.append(
            "sam2 はインストール済みですが sam2._C を import できません (CUDA拡張ロード失敗)。"
            "setup_sam2_cuda_windows.ps1 で再ビルドしてください。"
        )
        return 3, msgs

    if not sam2_info.get("installed"):
        msgs.append("SAM 2 は未インストールです (環境は整っています)。setup スクリプトで導入してください。")

    msgs.append("CUDA拡張をビルド・実行できる可能性が高い環境です。")
    return 0, msgs


def main() -> int:
    report = collect()
    code, msgs = evaluate(report)
    report["exit_code"] = code
    report["messages"] = msgs

    # コンソール表示
    print("=" * 70)
    print(" SAM 2 CUDA 環境診断")
    print("=" * 70)
    for k, v in report.items():
        if k in ("messages",):
            continue
        print(f"{k:32}: {v}")
    print("-" * 70)
    for m in msgs:
        print(f"  * {m}")
    print(f"exit_code = {code}")

    # JSON 保存
    repo_root = Path(__file__).resolve().parent.parent.parent
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / "sam2_environment_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {out_path}")

    return code


if __name__ == "__main__":
    sys.exit(main())
