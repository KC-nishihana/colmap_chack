# COLMAP Mask Editor v0.10

COLMAP形式のプロジェクトに対応した、マスク画像の手動確認・修正GUIツールです。  
v0.4A では **GrabCutによる半自動マスク生成機能** を追加しました。  
v0.4A.1 では **大画像対応・GUIスレッド分離・例外処理強化・テスト追加** を行いました。  
v0.4B・v0.5 では **GrabCut再推定・ヒント描画・補正UI** を追加しました。  
v0.5.1 では **UI整理（3タブ化）・QSettings設定保存・未確定GrabCut保護・安定性向上** を行いました。  
v0.6 では **Meta SAM 2.1 によるAIセグメンテーション（WindowsネイティブCUDA・CUDA拡張必須・QProcess常駐Worker）** を追加しました。  
v0.6.1 では **CUDA拡張カーネルの直接実行検証・実QProcess統合テスト・終了コード厳格化** を行いました。  
v0.7 では **SAM 2.1 Video Predictor による複数画像へのマスク伝播・レビュー・一括適用** を追加しました。  
v0.8 では **SAM 2.1 Automatic Mask Generator による全画像自動分割（各画像を独立解析・RLE圧縮NPZ保存・必要/不要レビュー・最終マスク生成）** を追加しました。  
v0.9 では **完全被覆・階層型リージョン分割（全画素100%所属・重複なし・粗い階層から判断・局所細分化）** を追加しました。  
v0.10 では **REMOVE_ONLY「不要領域だけ選択（推奨）」レビュー方式（全画素を暗黙KEEP・不要候補だけREMOVE・重複候補グループ化・代表候補表示・累積除外プレビュー）** を追加しました。

## v0.10 REMOVE_ONLY「不要領域だけ選択」レビュー方式

V0.8 の従来方式は、SAM が生成した候補ごとに KEEP / REMOVE / 未確認 を判断するため、候補数が多い画像では操作回数が
増えがちでした。**v0.10 の REMOVE_ONLY** は、画像全体または既存マスクを初期状態（暗黙 KEEP）として採用し、ユーザーは
**不要と判断した SAM 候補だけ** を REMOVE します。KEEP を個別に設定する必要はありません。

- **レビュー方式の選択**: 「不要領域だけ選択（推奨）」(`remove_only`) と「必要・不要を個別設定（従来方式）」(`standard`)。
  従来方式は削除せず維持しています。
- **判断状態の解釈**: 未確認 = 暗黙 KEEP / REMOVE = 明示除外 / KEEP = 互換用。**未確認が残っていても最終マスク生成・
  レビュー完了が可能**です（未確認は最終マスクへ REMOVE として反映しません）。
- **基準マスク**: 現在の通常マスク（既定）か画像全体（全面 255）。既存マスクのサイズが画像と一致しない場合は中止します
  （全面 255 へ黙って置換しません）。
- **最終マスク合成**: 既存 `amg_mask_composer.compose_final_mask` の `MODE_EXCLUDE_REMOVE` を再利用します
  （`base[REMOVE和集合] = 0` → uint8 0/255）。REMOVE_ONLY 専用の重複処理は作りません。
- **重複候補の削減**: RLE 同士の IoU / containment を dense 復号なしで計算し、ほぼ同じ対象の候補を 1 グループへまとめ、
  代表候補だけ表示します。REMOVE 済み領域に 98% 以上包含された候補は表示抑制します（判断値は改変しません）。
- **確認順**: 大きい候補・画像端に接する候補・品質スコア・確認順スコア・SAM 順で並べ替えられます（意味分類ではありません）。
- **操作**: 左クリック=不要 / 右クリック・Ctrl+左=解除、`R`/`U`/`Enter`/`N`/`P`/`Space`、複数選択の一括 REMOVE/解除、
  判断専用 Undo（最大 100・通常マスク Undo とは分離）、判断後の次候補移動・完了後の次画像移動。
- **キャッシュ**: 重複グループ結果を `review_index.npz`（`allow_pickle=False`・dense 禁止・原子保存）へ保存し、
  `segments.npz` の SHA-256 やグループしきい値が変わると stale 化します。重複計算は GUI スレッド外の CPU Worker で実行します。

## v0.9 完全被覆・階層型リージョン分割

V0.8 の Automatic Mask Generator は SAM 候補同士が重複し、未検出画素が残り、1 つの対象に複数の部分候補が
生成され、判断回数が多くなりがちでした。**v0.9 の「完全被覆リージョン」** は、画像の全画素を **重複なく** リージョンへ
割り当て、最初に **20〜40 程度の大きなリージョンだけ** を判断します。境界が不十分な場所だけを子リージョンへ展開して
詳細に判断できます。これが v0.9 の **主レビュー方式** です（V0.8 の AMG 候補レビューは詳細確認用として残しています）。

- **完全被覆リージョンとは**: 全画素を 1 つの葉リージョンへ必ず割り当てる階層型の分割。`region_id 0`（未所属）や
  負値を残しません。**全画素 100% 所属・重複所属 0・coverage_ratio = 1.0** を保証します。
- **SAM 候補との違い**: SAM 候補（V0.8 `segments.npz`）は **統合のヒントとしてのみ** 使用し、そのまま最終領域には
  しません。SAM 候補が無くても色・テクスチャ・境界だけで完全被覆を生成します。
- **基礎分割バックエンド**:
  - **SLICO**（`cv2.ximgproc.createSuperpixelSLIC`／OpenCV contrib）
  - **Grid Watershed**（OpenCV 標準機能のみで動作する必須代替。境界画素 `-1` を隣接領域へ割当）
  - **AUTO**: SLICO が使えれば SLICO、無ければ Grid Watershed へ **自動フォールバック**。使用バックエンドは
    manifest へ記録します。
- **粒度**: 粗い／標準／詳細／カスタム。最初は粗い階層（20〜40 リージョン）を表示します。
- **階層型リージョン**: 隣接領域のみを **色・テクスチャ・境界・SAM 情報** で階層統合（Region Adjacency Graph +
  優先度付きキュー）。統合後の各リージョンは原則として連結です。
- **局所細分化**: レビュー画面で領域をダブルクリックすると、その領域だけを子へ 1 段階分割。Backspace で親へ戻す。
  画像全体の階層レベルは変えません。
- **親判断の継承**: 葉の実効判断は葉から root 方向へ最初に見つかった明示判断。子で別判断を設定すれば上書きできます。
  粒度を変えても判断は失われません（判断は階層ノードへ保存し表示カットと独立）。
- **KEEP／REMOVE 確定**: 左クリック=KEEP、右クリック=REMOVE、Ctrl+左=未確認、N/Shift+N で次/前の未確認へ移動。
- **未確認画素率**: 画素所属率・KEEP/REMOVE/未確認率・表示/葉リージョン数を常時表示（葉面積から集計、全解像度
  マスクは再生成しません）。
- **最終確定**: 未確認が残る場合は「未確認へ移動／KEEP にする／REMOVE にする／中止」を選択。確定後は未確認画素 0、
  KEEP 画素 + REMOVE 画素 = 全画素を保証します。
- **partition.npz**: 階層ツリー・region map の C-order run-length・SAM シグネチャを **固定 dtype 配列**（`allow_pickle=False`・
  dense H×W ラベルマップ禁止・原子保存・SHA-256・再読込検証）で保存。
- **partition_manifest.json / partition_review.json**: 管理情報（不変）と判断/UI 状態（小さな JSON）を分離。判断変更は
  partition_review.json だけを原子更新し、partition.npz を書き換えません。
- **キャッシュ再利用 / stale / corrupt**: 元画像 fingerprint・画像サイズ・`segments.npz` SHA・partition settings_hash・
  partition.npz SHA がすべて一致する場合だけ再利用。V0.8 `segments.npz` 再生成や設定変更で stale 判定。stale 時は
  古い review を自動移行せずバックアップし新規レビューを開始します。
- **CPU 専用 QProcess**: 重い分割・統合は GUI スレッドではなく CPU 専用 Worker
  （`partition_backend.partition_worker_main`）で実行。Worker は torch / sam2 / PySide6 を import しません。GUI も
  torch / sam2 を import しません。
- **一括適用 / ロールバック / 取り消し**: 複数画像のレビュー完了後、最終マスクを QThread で一括生成。共通トランザクション
  基盤で原子適用・失敗時ロールバック・最後のバッチ取り消しに対応。
- **作業解像度**: 8K 画像を全解像度の float 配列で大量保持しません。長辺が `working_max_side`（初期 2048）を超える
  場合は縮小して基礎分割を生成し、元解像度へは **最近傍補間** で戻します。
- **ベンチマーク**: `python -m tools.benchmark_partition_builder` で FHD/4K/8K × 粗い/標準を計測し
  `logs/partition_benchmark.{json,csv}` へ保存します。
- **日本語・全角スペースパス対応**。

## v0.8 全画像自動分割 (SAM 2.1 Automatic Mask Generator)

プロジェクト内の画像を **1枚ずつ独立** して SAM 2.1 Automatic Mask Generator (AMG) で自動セグメンテーションし、
セグメント候補を **RLE 形式へ変換して画像1枚につき1つの圧縮NPZ** へ保存します。解析後、元画像上の候補を
クリックして **必要 / 不要 / 未確認** を手動分類し、そこから従来形式の最終マスクPNGを生成します。

画像間伝播は使用しません（伝播は V0.7 の**実験的機能**として残しています）。AMG も **QProcess 常駐 Worker 側だけ**
で実行し、GUI は torch / sam2 を import しません。

- **解析方式**: 各画像を独立して AMG 処理。1枚の解析失敗が他画像へ影響しません。
- **プリセット**: 高速 / 標準 / 詳細（各項目は詳細設定で変更可。変更後は「カスタム」表示）。初期値は高速。
- **解析対象**: すべて / 選択画像 / 未処理 / 古い結果(stale) / 失敗 / 現在画像。初期値は未処理。
- **RLE とは**: マスクを「背景長・前景長・…」の連長で表す圧縮表現。SAM 2 公式 `uncompressed_rle`
  （Fortran order=列優先、背景開始、`counts` 交互配列、`size=[h,w]`）と一致。GUI 側は NumPy のみで復号します。
- **NPZ 保存形式**: `segments.npz`（不変・`allow_pickle=False` で読込可・dense マスクや pickle を含まない）。
  判断状態は NPZ へ書かず、`manifest.json` の小さな JSON のみを原子的に更新します（レビューで NPZ を書き直しません）。
- **セグメント PNG は保存しません**。最終マスクだけを PNG として書き出します。
- **manifest.json**: 元 `image_key` / `source_path` / fingerprint / モデル / generator設定 / `settings_hash` /
  `segments_npz_sha256` / `decisions` などを管理。`batch_manifest.json` がプロジェクト全体の処理状況を持ちます。
- **キャッシュ再利用 / stale / corrupt**: 元画像（file_size, mtime_ns）・サイズ・モデル・SAM2コミット・
  checkpoint・generator設定・schema・SHA-256 がすべて一致したときだけ再利用。変われば stale（自動削除せず再解析対象）。
  NPZ 破損は corrupt。
- **途中停止・再開**: 一時停止 / 再開 / キャンセル（現画像完了後に停止・完成済み結果は保持）。アプリ再起動後も
  処理済み画像を再利用し、`processing` のまま残った状態は次回 `unprocessed` へ回復します。
- **必要 / 不要 / 未確認**: 左クリック=KEEP、右クリック=REMOVE、Ctrl+左=未確認へ戻す。重複候補は Tab で切替。
  クリック判定は全マスクを復号せず `rle_contains_point()` で行います。**REMOVE が KEEP より優先**、未確認は自動反映しません。
- **最終マスク生成方式**: 不要領域を除外（既定）/ 必要領域のみ / 現在マスクへ追加・除外。
- **一括適用 / ロールバック / バッチ取り消し**: 既存マスクをバックアップ→一時生成→全成功後 `os.replace`、失敗時はロールバック。
- **キャッシュ検証 / 削除**: 整合検証（SHA-256・dtype・shape・offsets・RLE合計・decision整合等）と、
  現在/選択/stale/失敗/全キャッシュの削除（最終マスクPNGは削除対象外）。
- 解析キャッシュは `project/segmentation_cache/`（`batch_manifest.json` と `images/<cache_id>/{segments.npz, manifest.json}`）。
  `cache_id` は `image_key` の SHA-256 先頭16文字（日本語・同名・サブフォルダ・禁止文字・パス長・大小衝突を回避）。

実機 AMG テスト:
```powershell
$env:RUN_SAM2_CUDA_TESTS = "1"
$env:SAM2_CHECKPOINT = "C:\...\sam2.1_hiera_small.pt"
python -m pytest -m sam2_cuda colmap_mask_editor/tests/test_sam2_amg_qprocess_cuda_integration.py -v
```

ベンチマーク（FHD/4K/8K × 高速/標準）:
```powershell
python colmap_mask_editor/tools/benchmark_sam2_amg.py
# -> logs/sam2_amg_benchmark.json / logs/sam2_amg_benchmark.csv
```

## v0.7 画像伝播 (SAM 2.1 Video Predictor, 実験的機能)

1枚の画像で確定した1対象のマスクを基準に、前後の連続画像へマスクを伝播します。
連続撮影画像から人物・車両・三脚・空・植生などを複数画像でまとめて除外/有効化する用途を想定しています。
伝播 (Video Predictor) も既存の **QProcess 常駐 Worker 側だけ** で実行し、GUI は torch/sam2 を import しません。

- **画像順序を明示**: 現在の一覧順 / COLMAP images.txt 優先 / ファイル名(自然順) / 撮影日時。
  COLMAP順を撮影時系列と決めつけず、開始前に順序を確認できます。
- **基準マスク**: 現在のAI候補、または現在の通常マスク。
- **伝播方向/範囲**: 前方向 / 後方向 / 前後、前後N枚または一覧で選択した範囲。
- **同一画像サイズ制限**: V0.7 は同一サイズの画像のみ伝播します (自動リサイズしません)。
- **ステージング**: SAM 2 Video Predictor 用に連番JPEG (`000000.jpg`...) へ変換 (元画像は変更しません。EXIF/色空間は単一画像推論と統一)。日本語・全角スペースを含むパスに対応します。
- **非同期ジョブ**: 伝播は Worker 内の専用スレッドで実行し、一時停止 / 再開 / キャンセルを受け付けます (キャンセルは現フレーム完了後・完成済み結果は保持)。
- **品質警告**: 前景率・連結成分数・前フレームとの面積比/IoU・境界接触などから警告を表示します (自動破棄はしません。カメラ移動が大きいと正しい対象でも値が変動するため)。
- **結果レビュー**: フレームごとに採用/除外し、採用した画像だけへ **追加 / 除外 / 置換** で一括適用します。
- **トランザクション適用**: 既存マスクをバックアップし、全マスクを一時生成→全成功後に `os.replace` で確定。途中失敗時はロールバックします。
- **バッチ取り消し**: 最後の一括適用をまとめて取り消せます (既存はバックアップから復元、新規作成は削除)。
- 一時ファイルは `%LOCALAPPDATA%/COLMAPMaskEditor/propagation_runtime/<job_id>/` (frames/results/backup)。

実機 Video Predictor テスト:
```powershell
$env:RUN_SAM2_CUDA_TESTS = "1"
$env:SAM2_CHECKPOINT = "C:\...\sam2.1_hiera_small.pt"
python -m pytest -m sam2_cuda colmap_mask_editor/tests/test_sam2_video_qprocess_cuda_integration.py -v
```

## 動作環境

- OS: Windows 11 (ネイティブ。WSL / Docker は使用しません)
- Python: 3.12系
- 依存ライブラリ: PySide6, OpenCV, NumPy, Pillow
- AI機能 (任意): NVIDIA GPU (RTX 4090 検証) + CUDA Toolkit + PyTorch(CUDA版) + SAM 2 (CUDA拡張)

---

## v0.6 AIセグメンテーション (SAM 2.1)

SAM 2.1 を使った点・矩形プロンプトによる半自動マスク生成です。**SAM 2 関連の重い処理
(PyTorch / SAM 2 / CUDA拡張) はすべて別プロセス (QProcess 常駐 Worker) で実行され、
GUI プロセスは `torch` / `sam2` / `sam2._C` を一切 import しません。** そのため Worker が
クラッシュ・CUDAエラー・Out of Memory で落ちても、本体 (ブラシ・GrabCut・保存) は動作し続けます。

### 構成

| プロセス | 役割 | 依存 |
|---------|------|------|
| GUIプロセス | PySide6 / OpenCV / 既存編集機能 | torch非依存 |
| AIサブプロセス (Worker) | PyTorch / SAM 2.1 / CUDA / SAM 2 CUDA拡張 | `sam_backend/worker_main.py` |

- 通信: **JSON Lines** (stdout=JSON専用 / stderr=ログ)。マスク本体は **NPZ一時ファイル**で受け渡し。
- Worker は推論ごとに起動せず、アプリ使用中は**常駐**します。
- **CUDA拡張 (`sam2._C`) は必須**です。読み込めない場合 AI 機能は無効化され、CPUや拡張なし処理へ**フォールバックしません**。

### セットアップ手順 (実機が必要)

1. **環境診断** (環境を変更しません):
   ```powershell
   python colmap_mask_editor/scripts/check_sam2_cuda_environment.py
   ```
   `logs/sam2_environment_report.json` に結果が出ます。終了コード 0 = ビルド可能性が高い。

2. **SAM 2 + CUDA拡張の導入** (PyTorch CUDA版が入っている前提・VS2022 x64 Native Tools 環境で実行):
   ```powershell
   $env:SAM2_BUILD_CUDA = "1"; $env:SAM2_BUILD_ALLOW_ERRORS = "0"
   colmap_mask_editor\scripts\setup_sam2_cuda_windows.ps1
   # ビルド+import のみ確認 (実機検証なし) なら:
   colmap_mask_editor\scripts\setup_sam2_cuda_windows.ps1 -BuildOnly
   ```
   SAM 2 は `external/sam2/` へ clone され、`sam_backend/sam2_manifest.json` の検証済みコミットへ
   checkout されます (リポジトリにはコミットしません)。チェックポイントがあれば実機検証まで行い、
   成功時のみマニフェストの `verified=true` と検証情報を書き込みます。

   セットアップ/検証スクリプトの**終了コード**:

   | コード | 意味 |
   |--------|------|
   | 0 | 完全検証成功 (`-BuildOnly` ではビルド+import 成功) |
   | 1 | 環境または入力不足 |
   | 2 | PyTorch CUDA と CUDA Toolkit のバージョン不整合 |
   | 3 | CUDA拡張・SAM推論のビルド/ロード/実行失敗 |
   | 4 | チェックポイント不足により実機検証未完了 |

3. **モデル配置**: `models/sam2/sam2.1_hiera_small.pt` を配置 (リポジトリにはコミットしません)。

4. **CUDA拡張検証**:
   ```powershell
   python colmap_mask_editor/scripts/verify_sam2_cuda_extension.py --checkpoint models/sam2/sam2.1_hiera_small.pt
   ```
   `sam2._C` の import「だけ」では成功にしません。次をすべて実機で確認します:
   - `sam2._C` の import
   - `get_connected_components()` による **CUDA カーネルの直接実行** (連結成分の面積 100/300 を検証)
   - `fill_holes_in_mask_scores()` の後処理
   - SAM 2 モデルロード・画像Embedding・正/負クリック・矩形・multimask 推論

   結果は `logs/sam2_cuda_verification.json` に保存され、成功条件を満たさない場合は終了コードを 0 にしません。

### 使い方

1. 「AIセグメント」タブで **Worker起動 → モデル読込**。
2. **正クリック (左)** で対象点、**負クリック (右)** で背景点、**左ドラッグ**で矩形を指定。
3. **推論実行** で最大3候補とスコアを表示。**候補1/2/3** で切替 (再推論なし)。
4. **追加 / 除外 / 置換** で通常マスクへ適用 (Undo 可能)。**キャンセル**で破棄。
5. 画像切替・終了時に未確定があれば **適用 / 破棄 / キャンセル** を確認します。

### Workerクラッシュ時の復旧

Worker がクラッシュしても本体は維持されます。「AIセグメント」タブの **Worker再起動** で再開できます。
通常マスクは変更されず、ブラシ・GrabCut・保存はそのまま使えます。

### テスト

```powershell
# 通常テスト (torch/GPU不要・Fake Worker使用)
& "C:\ProgramData\Anaconda3\Scripts\conda.exe" run -p "C:\conda-envs\colmap_mask_editor" python -m pytest colmap_mask_editor/tests/ -q

# 実機CUDAテスト (要 GPU/SAM2/チェックポイント)
$env:RUN_SAM2_CUDA_TESTS = "1"
python -m pytest -m sam2_cuda -v

# ベンチマーク (FHD/4K/8K)
python colmap_mask_editor/tools/benchmark_sam2.py --checkpoint models/sam2/sam2.1_hiera_small.pt
```

## インストール

```
cd colmap_mask_editor
pip install -r requirements.txt
```

## 起動方法

```
python app.py
```

コマンドライン引数でプロジェクトフォルダを指定することもできます:

```
python app.py C:\path\to\project
```

---

## プロジェクトフォルダの構成

```
project/
├─ images/                 ← 必須。全画像を格納
│  ├─ IMG_0001.jpg
│  └─ sub/
│     └─ IMG_0003.jpg
├─ sparse/                 ← オプション
│  └─ 0/
│     └─ images.txt        ← COLMAP登録情報（画像の順序付けに使用）
├─ masks/                  ← オプション。既存マスクを格納
│  └─ IMG_0001.png
└─ masks_colmap/           ← COLMAP互換出力先（ツールが自動生成）
   └─ IMG_0001.jpg.png
```

### images.txt の扱い

`sparse/0/images.txt` が存在する場合:

- `images/` フォルダ内の**全画像**を表示します（`images.txt` に関わらず）
- `images.txt` に登録されている画像を先頭に、未登録の画像をその後に並べます
- 未登録画像は一覧で `[未登録]` タグ付きで表示されます

> **補足:** `images.txt` の POINTS2D 行が空行（特徴点マッチングなし）でも、全画像が正しく認識されます。

---

## 基本的な使い方

1. `python app.py` で起動
2. **ファイル > プロジェクトを開く** でプロジェクトフォルダを選択
3. 左ペインで編集したい画像を選択
4. 中央キャンバスに画像とマスク（赤い半透明オーバーレイ）が表示される
5. 左クリックドラッグでマスク追加、右クリックドラッグで削除
6. **S** または **Ctrl+S** で保存（`masks/` フォルダを上書き保存または新規作成）

---

## キーボードショートカット

| キー | 操作 |
|------|------|
| S / Ctrl+S | 保存 |
| A | 前の画像 |
| D | 次の画像 |
| Z / Ctrl+Z | Undo / **GrabCutヒントUndo（ヒント描画中）** |
| Ctrl+Y | Redo / **GrabCutヒントRedo（ヒント描画中）** |
| **Ctrl+Shift+Z** | **全ヒントを消去（ヒント描画中）** |
| M | マスク表示 ON/OFF |
| F | 差分表示 ON/OFF |
| B | ブラシモード |
| R | 矩形追加モード |
| Shift+R | 矩形削除モード |
| P | ポリゴン追加モード |
| Shift+P | ポリゴン削除モード |
| **G** | **GrabCut有効化モード** |
| **Shift+G** | **GrabCut除外モード** |
| **Ctrl+G** | **GrabCut置換モード** |
| Enter | ポリゴン確定 / GrabCutプレビュー適用 |
| **Ctrl+Enter** | **GrabCut再推定（ヒントを使って再実行）** |
| Esc | ポリゴンキャンセル / GrabCutプレビューキャンセル |
| Backspace | ポリゴンの最後の頂点を削除 |
| + / = | ブラシサイズを大きくする |
| - | ブラシサイズを小さくする |
| ホイール | ズーム（カーソル位置を中心に） |
| 中ボタンドラッグ | パン |

---

## 右パネルの構成（v0.6）

右パネルは **4つのタブ** で構成されています。タブ下部のナビゲーションボタン（前の画像・次の画像・保存・Undo/Redo）は常に表示されます。

| タブ | 内容 |
|------|------|
| **編集** | 編集モード選択、ブラシ設定、マスク表示、差分表示、モルフォロジー処理、小領域除去 |
| **GrabCut** | GrabCut設定、補正（ヒント描画・再推定）、GCステータス表示 |
| **AIセグメント** | SAM 2.1 状態（Worker/CUDA/CUDA拡張/GPU/モデル/VRAM）、モデル設定、プロンプト（正/負/矩形）、候補マスク、追加・除外・置換 |
| **保存・確認** | 保存設定、品質チェック、COLMAP互換出力、統計表示 |

GrabCut系モード（G / Shift+G / Ctrl+G）を選択すると **GrabCutタブ** へ、通常編集モードを選択すると **編集タブ** へ自動切り替わります。

---

## 設定の自動保存（v0.5.1）

アプリ終了時に以下の設定が自動保存され、次回起動時に復元されます。

| 設定項目 | 内容 |
|----------|------|
| ウィンドウ位置・サイズ | 起動時に前回の位置を復元 |
| スプリッタ位置 | 左ペイン・中央・右パネルの幅比率 |
| 右パネルタブ番号 | 前回選択していたタブ |
| ブラシサイズ | 前回のブラシサイズ |
| GrabCut設定 | 反復回数・大画像縮小・最大処理サイズ・後処理オプション |
| 最後に開いたフォルダ | ファイルダイアログの初期フォルダ |

**設定 > 設定を初期化** ですべての設定を工場出荷状態に戻せます。

---

## 未確定状態の保護（v0.5.1）

### GrabCut未確定セッションの保護

GrabCutプレビュー表示中・ヒント編集中に他の画像へ移動しようとすると、3択ダイアログが表示されます。

| 選択肢 | 動作 |
|--------|------|
| 適用 | 現在のGrabCutプレビューをマスクに確定してから移動 |
| 破棄 | GrabCutプレビューを破棄して移動 |
| キャンセル | 移動を中止してGrabCutを継続 |

### 未保存マスクの確認

未保存のマスク編集がある状態で他の画像に移動しようとすると、3択ダイアログが表示されます。

| 選択肢 | 動作 |
|--------|------|
| 保存 | マスクを保存してから移動 |
| 破棄 | 編集を破棄して移動 |
| キャンセル | 移動を中止して編集を継続 |

---

## 右パネルの機能詳細

### 編集モード

右パネルの「編集」タブでモードを選択します。現在のモードはステータスバーにも表示されます。

| モード | ショートカット | 操作 |
|--------|--------------|------|
| ブラシ追加/削除 | B | 左クリック=追加、右クリック=削除 |
| 矩形追加 | R | ドラッグで矩形範囲を指定→マスク追加 |
| 矩形削除 | Shift+R | ドラッグで矩形範囲を指定→マスク削除 |
| ポリゴン追加 | P | クリックで頂点追加→Enter確定 |
| ポリゴン削除 | Shift+P | クリックで頂点追加→Enter確定 |
| **GrabCut有効化** | **G** | **矩形ドラッグ→候補をプレビュー→Enter適用** |
| **GrabCut除外** | **Shift+G** | **矩形ドラッグ→候補をプレビュー→Enter適用** |
| **GrabCut置換** | **Ctrl+G** | **矩形ドラッグ→候補をプレビュー→Enter適用** |
| パン操作 | — | 左クリックドラッグでパン |

---

## GrabCut機能

### 概要

GrabCutは、矩形範囲を指定するだけで、OpenCVの機械学習アルゴリズムが前景（被写体）と背景を自動分離する半自動マスク生成機能です。

> **注意:** GrabCutは完全自動ではありません。  
> 候補領域を素早く生成したあと、人間がブラシ・矩形・ポリゴン・モルフォロジーで仕上げる運用を想定しています。

### GrabCutの3つのモード

| モード | キー | 動作 |
|--------|------|------|
| GrabCut有効化 | G | GrabCutで抽出した領域を **有効領域（255）に追加** する |
| GrabCut除外 | Shift+G | GrabCutで抽出した領域を **除外領域（0）に変更** する |
| GrabCut置換 | Ctrl+G | GrabCut結果で現在マスクを **まるごと置換** する |

### 操作手順

1. **G / Shift+G / Ctrl+G** でGrabCutモードに切り替える
2. キャンバス上で抽出したい領域を**左クリックドラッグ**して矩形を指定する
3. マウスを離すと、GrabCutが実行されて**候補領域がプレビュー表示**される
   - 有効化モード: 黄色オーバーレイ
   - 除外モード: 青系オーバーレイ
   - 置換モード: 緑系オーバーレイ
4. ステータスバーに `GrabCutプレビュー中: Enter=適用 / Esc=キャンセル` と表示される
5. 結果が良ければ **Enter** で確定、やり直すなら **Esc** でキャンセル
6. 確定後は Undo/Redo で元に戻せる

### GrabCut設定

右パネルの「GrabCut設定」グループで調整できます。

| 設定 | 内容 | 初期値 |
|------|------|--------|
| 反復回数 | 大きいほど精度向上（処理は遅くなる） | 5 |
| 適用後に膨張 | 確定時にGrabCut結果を膨張処理してからマスクへ合成 | OFF |
| 適用後に収縮 | 確定時にGrabCut結果を収縮処理してからマスクへ合成 | OFF |
| 後処理カーネルサイズ | 膨張・収縮のカーネルサイズ | 3 |
| 大画像を縮小して処理する | ROI切り出し＋縮小でGrabCutを実行 | ON |
| GrabCut最大処理サイズ | ROI長辺の上限（px）。超えた場合に縮小する | 2048 |
| **既存の除外領域を背景制約として使用** | **現在マスクが0の領域をGC_BGD制約として使用（v0.4B）** | **OFF** |

### 大画像処理の仕組み（v0.4A.1）

4K・8K等の大画像をそのままGrabCutに渡すと処理時間とメモリ消費が大きくなります。  
v0.4A.1 では以下の最適化処理を行います:

```
元画像
  ↓ 指定矩形の周囲に余白を追加してROIを切り出す
  ↓ ROIの長辺が最大処理サイズを超えた場合、INTER_AREAで縮小
  ↓ 縮小座標系でGrabCut実行
  ↓ 結果をINTER_NEARESTで元ROIサイズへ復元（2値を維持）
  ↓ 元画像サイズのゼロマスクへ配置
最終マスク（元画像と同じ解像度・uint8・0/255のみ）
```

完了後、ステータスバーに元画像サイズ・ROIサイズ・処理解像度・縮小率・処理時間を表示します。

### GrabCut処理中の動作（v0.4A.1）

- GrabCutは**別スレッド**で実行されます。処理中もGUIは応答します
- 処理中はプログレスダイアログが表示されます
- **キャンセルボタン** または **Esc** でキャンセルできます（`cv2.grabCut()` の実行直前/直後に停止します）
- 処理中は以下の操作が一時的に無効になります:
  - GrabCut編集モードへの切り替え
  - GrabCut設定の変更
  - 前後画像移動
  - 保存・Undo/Redo
  - プロジェクトを開く

### エラー発生時の対処

| エラーメッセージ | 対処方法 |
|---|---|
| 矩形が小さすぎます | 対象物の周囲を広めに矩形指定してください |
| 矩形が画像全体に近すぎます | 背景が含まれるよう矩形を小さくしてください |
| メモリを確保できませんでした | 「GrabCut最大処理サイズ」を小さくしてください |
| 前景候補が見つかりません | 別の矩形範囲を指定してください |
| OpenCV GrabCut処理に失敗 | 反復回数を下げるか、矩形範囲を変更してください |

### GrabCut補正（v0.4B 新機能）

初回GrabCutの結果が不十分な場合、**対象ヒント（前景）** または **背景ヒント** をキャンバス上に描き込み、GrabCutを再実行して精度を改善できます。

#### 補正の流れ

1. 初回GrabCutを実行してプレビューを確認する
2. 右パネルの「GrabCut補正」グループが有効になる
3. ヒント種別を選択:
   - **対象ヒント**: 必ず抽出したい領域を緑で描く（`GC_FGD`制約）
   - **背景ヒント**: 必ず背景にしたい領域を赤で描く（`GC_BGD`制約）
   - **ヒント消去**: 描いたヒントを消去して初回GrabCut状態に戻す
4. ブラシサイズを調整してキャンバス上にヒントを描く
5. **再推定 [Ctrl+Enter]** を押すとヒントを反映した再実行が行われる
6. 満足いく結果になったら **適用 [Enter]** でマスクに確定する
7. やり直したい場合は **キャンセル [Esc]** で初期状態に戻る

#### ヒントのUndo/Redo

| 操作 | 動作 |
|------|------|
| Ctrl+Z | 最後のヒントストロークを取り消す |
| Ctrl+Y | 取り消したヒントストロークをやり直す |
| Ctrl+Shift+Z | 全ヒントを消去する |

> **注意:** ヒント描画中はCtrl+Z/YがヒントのUndo/Redoに使われます。マスクのUndo/Redoは適用後に使用してください。

#### 再推定の仕組み

```
初回GrabCut(GC_INIT_WITH_RECT)
  ↓ プレビュー表示
  ↓ ユーザーがヒントを描く
  ↓ ヒントをGrabCut内部ラベルに反映
再推定(GC_INIT_WITH_MASK)
  ↓ プレビュー更新
  ↓ ヒント追加・再推定を繰り返し
Enter → マスクへ確定
```

- **再推定は何回でも繰り返せます**
- 各再推定は `base_label_mask`（初回GrabCut結果）とすべてのヒントストロークから開始します
- 再推定もGUIスレッド外で実行されます（処理中もGUIは応答します）

### 注意点

- プレビュー表示はGrabCutの生の結果です。後処理（膨張・収縮）は **Enter** で確定した時点で適用されます
- モードを切り替えると、未確定のGrabCutプレビューは自動的にキャンセルされます
- GrabCut適用後は通常の編集（ブラシ・矩形・ポリゴン・モルフォロジー）で仕上げることを推奨します

---

### 差分表示

「差分表示 [F]」チェックボックスまたは **F** キーでON/OFFを切り替えます。

保存直後（またはロード直後）のマスクと、現在の編集状態を比較して表示します:

| 色 | 意味 |
|----|------|
| 緑 | 追加された領域 |
| 青 | 削除された領域 |
| 赤半透明 | 変化なしのマスク領域 |

### モルフォロジー処理

| ボタン | 動作 |
|--------|------|
| 膨張 +1 / +3 | 楕円形カーネルで 1 または 3px 膨張 |
| 収縮 -1 / -3 | 楕円形カーネルで 1 または 3px 収縮 |
| 穴埋め | MORPH_CLOSE でマスク内の穴を埋める（カーネルサイズ変更可、初期値5） |

すべてUndo/Redo対応です。

### 小領域除去

指定した面積（px）未満の白い連結領域を除去します。  
面積閾値はスピンボックスで変更可能（初期値: 100px）。Undo/Redo対応。

### ブラシ設定
ブラシサイズをスピンボックスまたはスライダーで変更（1〜300）。

### マスク表示
- マスク表示のON/OFFを切り替えます（キー: M）
- 透明度スライダーでオーバーレイの濃さを調整します

### 保存設定
「保存時にCOLMAP互換マスクも出力する」をONにすると、  
**S** で保存するたびに `masks_colmap/` にも同時出力します。

### 操作ボタン
| ボタン | 動作 |
|--------|------|
| 前の画像 / 次の画像 | 画像を切り替える |
| 保存 | 現在のマスクを保存 |
| 元に戻す / やり直し | Undo / Redo |
| 画像サイズに合わせてリサイズ | マスクサイズが画像と異なる場合に修正 |

### 品質チェック
| ボタン | 動作 |
|--------|------|
| 一括チェック | 全画像のマスク品質をチェックして一覧を更新。ステータスごとの枚数サマリを表示 |
| COLMAP互換出力 | `masks/` の全マスクを `masks_colmap/` に一括出力 |
| ログCSV出力 | チェック結果を `mask_check_log.csv` に出力 |

### マスク統計
現在表示中の画像の情報（サイズ、マスク率、ステータス、パス）をリアルタイムで表示。

---

## 保存先

### 通常保存（S / Ctrl+S）

既存マスクがある場合:
```
masks/<元のマスクパス> ← 上書き
```

マスクが存在しない場合:
```
masks/<画像名>.png ← 新規作成
```

### COLMAP互換マスク（オプション）

「保存時にCOLMAP互換マスクも出力する」を有効にした場合、または「COLMAP互換出力」ボタンを押した場合:

```
masks_colmap/<画像名>.jpg.png  ← 元ファイル名 + ".png"
```

例:
```
images/IMG_0001.jpg
  → masks_colmap/IMG_0001.jpg.png

images/sub/IMG_0003.jpg
  → masks_colmap/sub/IMG_0003.jpg.png
```

### 編集ログ

保存のたびに以下のCSVが追記されます:

```
project/mask_edit_log.csv
```

列: `image_path, input_mask_path, saved_mask_path, status, width, height, mask_width, mask_height, timestamp`

---

## マスク画像の仕様

| 値 | 意味 | 表示 |
|----|------|------|
| 255 (白) | 有効領域（復元対象） | 赤い半透明オーバーレイ |
| 0 (黒) | 除外領域（復元対象外） | オーバーレイなし |

- 形式: グレースケールPNG
- 左クリック: 有効領域を追加（255に塗る）
- 右クリック: 除外領域に変更（0に塗る）
- 読み込み時に0/255以外の中間値が含まれる場合は自動で2値化します（128以上→255、未満→0）

---

## 画像一覧のステータス表示

### ステータスラベル

| 表示 | 意味 |
|------|------|
| [OK] | マスクが正常 |
| [マスクなし] | マスクファイルが存在しない |
| [未保存] | 編集済みで未保存 |
| [サイズ不一致] | マスクサイズが画像と異なる |
| [空マスク] | マスクが全て255（除外ピクセルなし） |
| [全面マスク] | 除外ピクセルが95%以上 |
| [中間値あり] | 0/255以外のピクセルを含む（自動2値化済み） |
| [画像エラー] | 画像ファイルを読み込めない |
| [マスクエラー] | マスクファイルを読み込めない |
| [要確認] | 一括チェック未実行 |
| [未登録] | `images.txt` に登録されていない画像 |

### フィルタ

一覧上部のプルダウンで絞り込みができます:

```
すべて / 正常 / 要確認 / 未保存 / マスクなし /
サイズ不一致 / 空マスク / 全面マスク / 中間値あり / 読み込みエラー / COLMAP未登録
```

---

## チェックログ CSV（mask_check_log.csv）

「ログCSV出力」ボタンで出力。文字コードは UTF-8 BOM付き（Windows Excelで文字化けしません）。

| 列名 | 内容 |
|------|------|
| image_path | 画像の相対パス |
| input_mask_path | 入力マスクの相対パス |
| edited_mask_path | 編集済みマスクのパス（現バージョンでは空） |
| colmap_mask_path | COLMAP互換マスクの相対パス |
| status | ステータス文字列 |
| width / height | 画像サイズ |
| mask_width / mask_height | マスクサイズ |
| mask_ratio | 除外ピクセル率（0.0〜1.0） |
| has_intermediate_values | 中間値が含まれていたか |
| is_empty_mask | 除外ピクセルがゼロか |
| is_full_mask | 除外ピクセルが95%以上か |
| note | 補足メモ |

---

## ファイル構成

```
colmap_mask_editor/
├─ app.py                         エントリーポイント・ログ設定
├─ requirements.txt               実行用依存ライブラリ
├─ requirements-dev.txt           テスト用依存ライブラリ（pytest含む）
├─ pytest.ini                     pytest設定
├─ logs/                          アプリケーションログ（自動生成）
│  └─ colmap_mask_editor.log
├─ core/
│  ├─ version.py                  バージョン定数（APP_VERSION等）（v0.5.1追加）
│  ├─ app_settings.py             QSettings設定保存・読み込み（v0.5.1追加）
│  ├─ project_loader.py           プロジェクト構造解析・全画像スキャン
│  ├─ colmap_images_txt.py        images.txt パーサ
│  ├─ mask_io.py                  マスク読み書き・パス解決
│  ├─ mask_ops.py                 ブラシ編集・Undo/Redo・リサイズ
│  ├─ mask_checker.py             品質チェック・ステータス判定
│  ├─ colmap_export.py            COLMAP互換マスク一括出力
│  ├─ check_log.py                チェックログCSV出力
│  ├─ mask_morphology.py          膨張・収縮・穴埋め
│  ├─ mask_components.py          小領域除去
│  ├─ grabcut_tool.py             GrabCutSession・HintStroke・create/apply/refine
│  └─ grabcut_worker.py           GrabCutWorker INITIAL/REFINE両対応
├─ ui/
│  ├─ main_window.py              メインウィンドウ・3タブUI・設定保存・保護ダイアログ（v0.5.1更新）
│  ├─ image_canvas.py             キャンバス・GrabCutUiState・ヒント描画
│  └─ image_list_panel.py         画像一覧・フィルタ・ステータス表示
├─ tests/                         自動テスト（155テスト）
│  ├─ test_grabcut_tool.py        GrabCutツール単体テスト
│  ├─ test_grabcut_large_image.py 大画像処理テスト
│  ├─ test_grabcut_worker.py      WorkerシグナルテストINITIAL/REFINE
│  ├─ test_grabcut_session.py     GrabCutSession生成テスト
│  ├─ test_grabcut_hints.py       ヒントストローク座標変換テスト
│  ├─ test_grabcut_refine_worker.py REFINEタスクテスト
│  ├─ test_grabcut_refine_gui.py  補正UIスモークテスト
│  ├─ test_main_window_smoke.py   GUIスモークテスト
│  ├─ test_version.py             バージョン整合テスト（v0.5.1追加）
│  ├─ test_app_settings.py        設定保存テスト（v0.5.1追加）
│  ├─ test_right_panel_tabs.py    タブUI・自動切替テスト（v0.5.1追加）
│  ├─ test_pending_state_guard.py 未確定状態確認テスト（v0.5.1追加）
│  └─ test_grabcut_thread_integration.py QThread統合テスト（v0.5.1追加）
└─ (tools/)
   └─ benchmark_grabcut.py        4K/8K GrabCutベンチマーク（v0.5.1追加、tools/配下）
```

---

## テスト実行

```bash
cd colmap_mask_editor
pip install -r requirements-dev.txt
pytest
```

実行結果例（全155テスト）:

```
tests/test_grabcut_tool.py               - apply_grabcut_result / 入力検証 / 戻り値仕様
tests/test_grabcut_large_image.py        - ROI計算 / 縮小・復元 / 大画像統合
tests/test_grabcut_worker.py             - INITIAL/REFINEシグナル / キャンセル / 例外処理
tests/test_grabcut_session.py            - GrabCutSession構造 / ROI計算 / ラベル値検証
tests/test_grabcut_hints.py              - ヒント座標変換 / FG/BG/消去 / ROI境界外無視
tests/test_grabcut_refine_worker.py      - GC_INIT_WITH_MASK / rect=None / 出力形状検証
tests/test_grabcut_refine_gui.py         - 補正UI存在 / 有効無効切替 / ブラシサイズ同期
tests/test_main_window_smoke.py          - ウィンドウ生成 / 初期値確認
tests/test_version.py                    - バージョン文字列整合 / README確認（v0.5.1）
tests/test_app_settings.py               - デフォルト値 / 保存・復元 / クランプ / リセット（v0.5.1）
tests/test_right_panel_tabs.py           - タブ存在 / ラベル / 自動切替 / 設定復元（v0.5.1）
tests/test_pending_state_guard.py        - GrabCut保護 / 未保存保護 / 3択動作（v0.5.1）
tests/test_grabcut_thread_integration.py - QThread統合 / シグナル伝達 / 参照解放（v0.5.1）
```

### 4K/8K ベンチマーク

```bash
python tools/benchmark_grabcut.py
python tools/benchmark_grabcut.py --only 4k --iter 2
python tools/benchmark_grabcut.py --only 8k --no-downscale
```

---

## 注意事項

- 日本語パス・全角スペースを含むパスに対応しています
- 大きな画像でも操作が重くならないよう、表示はスケールされた合成画像を使用しています
- Undo は最大50ステップ保持されます
- `images.txt` に POINTS2D 行が空（特徴点マッチングなし）の場合も全画像を正しく認識します
- GrabCutは完全自動マスクではありません。候補生成後にブラシ・矩形・ポリゴン・モルフォロジーで仕上げる運用を推奨します

---

## バージョン履歴

| バージョン | 内容 |
|-----------|------|
| v0.5.1 | UI整理（右パネル3タブ化）、QSettings設定自動保存・復元、GrabCut未確定セッション保護（3択ダイアログ）、未保存マスク確認改善、Worker終了処理強化（deferred close）、バージョン管理一元化（core/version.py）、新規テスト5本追加（合計155テスト）、4K/8Kベンチマークスクリプト追加 |
| v0.5 | GrabCut補正UIのヒントボタンactive表示、GCステータスラベル追加、保存失敗通知、画像切替確認ダイアログ、統計表示デバウンス等9件のUI/UX改善 |
| v0.4B | GrabCut再推定機能追加（対象/背景ヒント描画・GC_INIT_WITH_MASKによる繰り返し改善・ヒントUndo/Redo・既存マスクの背景制約オプション）、補正UI追加、テスト4本追加（合計139テスト） |
| v0.4A.1 | 大画像対応（ROI切り出し・自動縮小・元解像度復元）、GrabCutのGUIスレッド分離、プログレス表示、キャンセル機能、処理中UI無効化、例外処理強化、自動テスト追加 |
| v0.4A | GrabCutによる半自動マスク生成機能を追加（有効化・除外・置換の3モード、プレビュー表示、後処理オプション） |
| v0.3 | ポリゴン追加/削除、差分表示、膨張/収縮/穴埋め、小領域除去、品質チェック、CSVログ、COLMAP互換出力 |
| v0.2 | 矩形追加/削除モード追加 |
| v0.1 | 初版（ブラシ編集・Undo/Redo・ズーム・パン） |
