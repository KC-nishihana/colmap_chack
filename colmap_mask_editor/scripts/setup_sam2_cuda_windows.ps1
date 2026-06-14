<#
.SYNOPSIS
    Phase 2: Windows ネイティブで CUDA 拡張付き SAM 2 を現在の Python 環境へ導入する。

.DESCRIPTION
    PyTorch (CUDA版) が既に正しく入っている前提で、SAM 2 公式リポジトリを
    external/sam2 へ clone し、マニフェスト指定コミットへ checkout して
    CUDA 拡張を必須ビルドする。

    成功条件 (すべて満たした時のみ成功扱い):
      - torch.cuda.is_available() == True
      - CUDA_HOME が None でない
      - nvcc 実行可能
      - cl.exe 実行可能
      - sam2 import 可能
      - sam2._C import 可能
      - GPU Compute Capability 取得可能
      - SAM 2 画像モデルを CUDA へロード可能
      - 画像推論が成功
      - CUDA 拡張を使う後処理が実行可能

.NOTES
    禁止事項 (CLAUDE.md V0.6):
      - SAM2_BUILD_CUDA=0 を使わない
      - CUDA拡張の失敗を警告だけで無視しない
      - 上流ソースを自動書き換えしない
      - -allow-unsupported-compiler を無条件付与しない
      - VS と CUDA Toolkit が非互換な状態で強制ビルドしない
      - PyTorch CPU 版へフォールバックしない
      - GUI から pip install しない

    必ず「VS2022 x64 Native Tools Command Prompt」由来のシェル
    (cl.exe / nvcc が PATH にある) から、対象 conda 環境を有効化して実行すること。
#>

[CmdletBinding()]
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")),
    [string]$Sam2Commit = ""   # 省略時はマニフェストの commit を使用
)

$ErrorActionPreference = "Stop"

function Fail($msg) {
    Write-Host "[FAIL] $msg" -ForegroundColor Red
    exit 1
}
function Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "[ OK ] $msg" -ForegroundColor Green }

$logsDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$externalDir = Join-Path $RepoRoot "external"
$sam2Dir = Join-Path $externalDir "sam2"
$manifestPath = Join-Path $RepoRoot "colmap_mask_editor\sam_backend\sam2_manifest.json"

# 0. 実行前スナップショット
Info "実行前の pip freeze / python 実行ファイルを保存"
python -m pip freeze > (Join-Path $logsDir "requirements_before_sam2.txt")
python -c "import sys; print(sys.executable)" > (Join-Path $logsDir "python_executable.txt")

# 1. Python 実行ファイル
$pyExe = (python -c "import sys; print(sys.executable)")
Info "Python: $pyExe"

# 2-3. PyTorch CUDA + torch.version.cuda
$torchCheck = python -c @"
import sys, json
try:
    import torch
    info = dict(
        torch=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        torch_cuda=getattr(torch.version,'cuda',None),
    )
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        info['cc'] = f'{cc[0]}.{cc[1]}'
        info['gpu'] = torch.cuda.get_device_name(0)
    print(json.dumps(info))
except Exception as e:
    print(json.dumps(dict(error=repr(e))))
"@
$torch = $torchCheck | ConvertFrom-Json
if ($torch.error) { Fail "PyTorch を import できません: $($torch.error)" }
if (-not $torch.cuda_available) { Fail "torch.cuda.is_available() == False。CUDA版PyTorchを確認してください (CPU版へフォールバックしません)。" }
Ok "PyTorch $($torch.torch) / torch.version.cuda=$($torch.torch_cuda) / GPU=$($torch.gpu) / CC=$($torch.cc)"

# 4. CUDA_HOME
$cudaHome = $env:CUDA_HOME
if (-not $cudaHome) { $cudaHome = $env:CUDA_PATH }
if (-not $cudaHome) { Fail "CUDA_HOME / CUDA_PATH が未設定です。CUDA Toolkit を確認してください。" }
Ok "CUDA_HOME = $cudaHome"

# 5. nvcc
$nvcc = (Get-Command nvcc -ErrorAction SilentlyContinue)
if (-not $nvcc) { Fail "nvcc が PATH にありません。CUDA Toolkit を導入してください。" }
$nvccVer = (nvcc --version | Select-String "release (\d+\.\d+)").Matches.Groups[1].Value
Ok "nvcc release $nvccVer"

# バージョン整合チェック (torch.version.cuda と nvcc)
$torchMajorMinor = ($torch.torch_cuda -split "\." | Select-Object -First 2) -join "."
if ($torchMajorMinor -and $nvccVer -and ($torchMajorMinor -ne $nvccVer)) {
    Fail "バージョン不整合: torch.version.cuda=$($torch.torch_cuda) と nvcc=$nvccVer のメジャー/マイナーが一致しません。両者を一致させてください (自動で別バージョンを入れません)。"
}

# 6. cl.exe
$cl = (Get-Command cl -ErrorAction SilentlyContinue)
if (-not $cl) { Fail "cl.exe が見つかりません。VS2022 x64 Native Tools 環境で実行してください。" }
Ok "cl.exe = $($cl.Source)"

# 7. Compute Capability (torch から取得した実値を使用)
$arch = $torch.cc
if (-not $arch) { Fail "Compute Capability を取得できませんでした。" }
Ok "TORCH_CUDA_ARCH_LIST = $arch (実機 GPU の値を使用)"

# 8-9. SAM 2 公式リポジトリ取得 + commit checkout
if (-not (Test-Path $manifestPath)) { Fail "マニフェストがありません: $manifestPath" }
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
if (-not $Sam2Commit) { $Sam2Commit = $manifest.commit }
$repoUrl = $manifest.repository

New-Item -ItemType Directory -Force -Path $externalDir | Out-Null
if (-not (Test-Path (Join-Path $sam2Dir ".git"))) {
    Info "SAM 2 を clone: $repoUrl"
    git clone $repoUrl $sam2Dir
} else {
    Info "SAM 2 は clone 済み: $sam2Dir"
    git -C $sam2Dir fetch --all
}

if ($Sam2Commit) {
    Info "checkout commit: $Sam2Commit"
    git -C $sam2Dir checkout $Sam2Commit
} else {
    Write-Host "[WARN] マニフェストに commit が未設定です。現在の既定ブランチHEADで進めます。" -ForegroundColor Yellow
    Write-Host "       検証成功後、このスクリプトが実コミットSHAをマニフェストへ記録します。" -ForegroundColor Yellow
}
$resolvedCommit = (git -C $sam2Dir rev-parse HEAD)
Ok "SAM 2 commit = $resolvedCommit"

# 10. CUDA 拡張を必須としてビルド (フォールバック禁止)
$env:SAM2_BUILD_CUDA = "1"
$env:SAM2_BUILD_ALLOW_ERRORS = "0"
$env:MAX_JOBS = "4"
$env:TORCH_CUDA_ARCH_LIST = $arch

Info "SAM 2 を no-build-isolation でビルド/インストール"
python -m pip install -v --no-build-isolation -e $sam2Dir
if ($LASTEXITCODE -ne 0) { Fail "SAM 2 のビルド/インストールに失敗しました (ビルド失敗を無視せず停止)。" }

# 11. 拡張 import テスト
$extCheck = python -c @"
import json
try:
    import sam2, sam2._C
    print(json.dumps(dict(ok=True)))
except Exception as e:
    print(json.dumps(dict(ok=False, error=repr(e))))
"@
$ext = $extCheck | ConvertFrom-Json
if (-not $ext.ok) { Fail "sam2._C を import できません (CUDA拡張ロード失敗): $($ext.error)" }
Ok "sam2._C import 成功"

# 12. 推論スモークテスト (CUDA拡張後処理を含む) - checkpoint があれば実行
$ckpt = Join-Path $RepoRoot "models\sam2\sam2.1_hiera_small.pt"
if (Test-Path $ckpt) {
    Info "推論スモークテスト (verify_sam2_cuda_extension.py)"
    python (Join-Path $PSScriptRoot "verify_sam2_cuda_extension.py") --checkpoint $ckpt --model "sam2.1_hiera_small"
    if ($LASTEXITCODE -ne 0) { Fail "CUDA拡張を含む推論スモークテストに失敗しました。" }
    Ok "推論スモークテスト成功"

    # 検証成功 → マニフェストへ commit を記録
    $manifest.commit = $resolvedCommit
    $manifest.verified = $true
    $manifest | ConvertTo-Json -Depth 10 | Set-Content -Path $manifestPath -Encoding utf8
    Ok "マニフェストへ検証済みコミットを記録: $resolvedCommit"
} else {
    Write-Host "[WARN] チェックポイントが見つかりません: $ckpt" -ForegroundColor Yellow
    Write-Host "       models/sam2/ へ sam2.1_hiera_small.pt を配置し、verify スクリプトで最終確認してください。" -ForegroundColor Yellow
}

# 13. 環境情報を保存
python (Join-Path $PSScriptRoot "check_sam2_cuda_environment.py") | Out-Null
Ok "セットアップ完了。logs/ の各レポートを確認してください。"
