# CHANGELOG

> このファイルがリリースノートの正本です。
> アプリ同梱の `colmap_mask_editor/CHANGELOG.md` はこのファイルへのポインタです。

## v0.10 (2026-06-16) — REMOVE_ONLY「不要領域だけ選択」レビュー方式

- V0.8 AMG を主軸に「例外レビュー方式」(`remove_only`) を追加。全画素を暗黙 KEEP とし、不要候補だけを REMOVE する
- レビュー方式を選択可能に: 「不要領域だけ選択（推奨）」と「必要・不要を個別設定（従来方式）」。従来 standard 方式は維持
- 判断状態の解釈: UNREVIEWED=暗黙 KEEP / REMOVE=明示除外 / KEEP=互換用。未確認が残っていても最終マスク生成・レビュー完了が可能
- 基準マスク: 現在の通常マスク or 画像全体（全面 255）。既存マスクのサイズ不一致は中止（全面へ黙って置換しない）
- 最終マスク合成は既存 `amg_mask_composer.compose_final_mask(MODE_EXCLUDE_REMOVE)` を再利用（専用の重複処理を作らない）
- 累積 REMOVE プレビュー（半透明赤）・現在候補（水色）・基準マスク外（暗いグレー）。表示切替と透明度スライダー
- 進捗表示を REMOVE 指定数・除外画素/率・有効画素/率・候補総数・確認対象候補へ刷新
- RLE 同士の intersection / union / IoU / containment を dense 復号なしで計算 (`amg_rle_overlap`)
- 重複候補のグループ化と代表候補表示、`review_index.npz`（`allow_pickle=False`・dense 禁止・原子保存・SHA-256/しきい値で stale）
- REMOVE 済み領域に 98% 以上包含された候補の表示抑制（判断値は改変しない）
- 判断後の次候補自動移動、レビュー完了後の次画像自動移動、確認順の並べ替え（面積/端接触/品質/確認順スコア/SAM 順）
- 複数候補の一括 REMOVE / 一括解除、候補判断専用の Undo（通常マスク Undo と分離・最大 100）
- 重複グループ計算は GUI スレッド外の CPU Worker (`AmgReviewIndexWorker`)。GUI は torch / sam2 を import しない
- manifest の review ブロックを後方互換拡張（`workflow` 無しは standard 扱い）。REMOVE_ONLY は remove のみ保存
- 設定スキーマ v5→v6（`amg/review_workflow`, `amg/remove_only/*` を追加。既存設定は保持）
- V0.9 完全被覆リージョン機能は維持

## v0.9 (2026-06-15) — 完全被覆・階層型リージョン分割

- 全画素を重複なくリージョンへ割り当てる完全被覆 partition
- SLICO と Grid Watershed の 2 バックエンド
- OpenCV contrib 未導入時の自動フォールバック
- V0.8 SAM 候補をリージョン統合のヒントとして再利用
- Region Adjacency Graph
- 色・テクスチャ・境界・SAM 情報による階層統合
- 粗い／標準／詳細の粒度設定
- 最初は 20～40 程度の大領域を表示
- 選択領域だけの局所細分化
- 親判断の子への継承と子判断による上書き
- 全画素の KEEP／REMOVE 確定
- partition.npz によるバイナリー保存
- partition_manifest と partition_review の分離
- CPU 専用 QProcess
- 最終マスクの一括生成・ロールバック・取り消し
- 設定スキーマ v4→v5

## v0.8 (2026-06-15) — 全画像自動分割 (SAM 2.1 Automatic Mask Generator)

- SAM 2.1 Automatic Mask Generator による全画像自動分割（各画像を独立解析）
- 高速・標準・詳細プリセット
- セグメント結果を SAM 2 公式互換の Fortran-order RLE へ変換
- 画像 1 枚につき圧縮 NPZ 1 ファイル（`allow_pickle=False`・dense マスク禁止・原子保存・再読込検証）
- 管理情報と判断状態を manifest.json で原子的に保存（NPZ は不変）
- キャッシュ有効性・破損(corrupt)・古い(stale) 検出。元画像 fingerprint と設定 hash で判定
- 高解像度で `points_per_batch` を自動縮小した場合も、要求値 (`generator`) と実効値 (`generator_effective`) を分離記録しキャッシュ再利用を維持
- 途中停止・再開・失敗画像のみ再処理。画像 1 枚の失敗は他画像へ波及しない（GUI もバッチ全体を停止しない）
- 必要・不要・未確認の手動レビュー（REUSABLE なキャッシュのみ対象）と重複候補切替
- 最終マスクの生成・通常マスクへの一括適用（QThread・進捗・キャンセル・原子適用・ロールバック・バッチ取り消し）
- 既存マスクとのサイズ不一致は黙って無視せず中止
- RLE によるクリック位置判定（復号は最終マスク生成時のみ）
- 日本語・全角スペースパス対応
- QProcess 実機 Automatic Mask Generator テスト
- 画像伝播を実験的機能として整理
- 設定スキーマ v3→v4 移行

## v0.7 (2026-06-14) — 画像シーケンス伝播 (SAM 2.1 Video Predictor)

- SAM 2.1 Video Predictor による複数画像へのマスク伝播 (1対象)
- 前方向・後方向・前後伝播 (基準フレームは両方向 yield されるため重複除去)
- 画像順序の明示選択 (現在の一覧順 / COLMAP images.txt / ファイル名 / 撮影日時) + 重複除去
- 基準マスク: 現在のAI候補 / 現在の通常マスク
- 同一画像サイズ制限・事前検証 (基準マスクの妥当性・サイズ均一・基準位置・重複・枚数)
- 連番JPEGステージング (日本語/全角スペースパス対応・元画像不変・原子書き込み)
- QProcess 内の **専用スレッドによる非同期伝播ジョブ** (pause / resume / cancel)
- フレームごとの逐次PNG保存 + 品質指標 (前景率・連結成分・面積比・IoU・境界接触) と警告
- 伝播中は単一画像系コマンドを BUSY 拒否、request_id 受付後は job_id で進捗管理
- 結果レビューと採用画像への一括適用 (追加 / 除外 / 置換)
- トランザクション適用 (バックアップ→一時生成→os.replace、途中失敗でロールバック)
- 最後の一括適用のバッチ取り消し
- 設定スキーマ v2→v3 移行 (propagation 設定追加・既存設定を保持)
- 実機 Video Predictor テスト (実 QProcess 経由・前後伝播・GPU解放・再起動後の単一画像推論)

## v0.6.1 (2026-06-14) — 検証強化・ドキュメント整合

### CUDA検証強化
- `sam2._C` の import 確認に加え、`get_connected_components` の CUDA カーネルを**直接実行**（連結成分の面積 100/300 を検証）
- `fill_holes_in_mask_scores` 後処理を実機検証
- 検証の成功判定を純粋関数 `ai/cuda_verification.py:evaluate_verification()` へ分離（import だけ・推論1件だけ等を成功にしない）
- セットアップの完全成功・未検証を終了コードで区別（0/1/2/3/4）。`-BuildOnly` を追加
- マニフェスト `verified` 管理を厳格化（実機フル検証成功後のみ true、失敗時は false へ戻す。`verification` 情報を記録）

### 実機統合テスト
- `test_sam2_qprocess_cuda_integration.py` を追加。**実 QProcess Worker 経由**でモデルロード・Embedding・正/負/矩形推論・NPZ を検証
- 日本語・全角スペースを含むパスで実機推論を検証
- Worker 終了後の QProcess NotRunning と **PID 別 GPU プロセス解放**（nvidia-smi compute-apps）を検証
- Worker 再起動後の再推論を検証
- `test_sam2_cuda_integration.py` にカーネル直接実行・fill_holes テストを追加し、候補数を厳格化
- 通常 pytest 追加: `test_cuda_verification_logic` / `test_sam2_manifest` / `test_setup_result_codes`（torch 非依存）

### ドキュメント
- 右パネルを **4タブ表記**へ統一（編集 / GrabCut / AIセグメント / 保存・確認）
- CUDA 検証手順（カーネル実行・fill_holes・推論）と終了コードを追記
- `APP_VERSION = "0.6.1"`（`SETTINGS_SCHEMA_VERSION` は 2 のまま）

## v0.6 (2026-06-14) — AIセグメンテーション (SAM 2.1)

- Meta SAM 2.1 による AIセグメンテーション追加
- WindowsネイティブCUDA対応・SAM 2 CUDA拡張必須化
- QProcess 常駐 Worker（GUIは torch/sam2/sam2._C を import しない）
- JSON Lines 通信・NPZ によるマスク受け渡し
- 正クリック・負クリック・矩形プロンプト・最大3候補マスク
- AIマスクの追加・除外・置換
- Worker クラッシュ・CUDA OOM・タイムアウト処理（本体は維持）
- CUDA 環境診断・セットアップ・検証スクリプト
- 設定スキーマ v1→v2 移行（AI設定追加・既存設定を保持）
- 右パネルを4タブ化（編集 / GrabCut / AIセグメント / 保存・確認）

## v0.5.1 (2026-06-14) — メンテナンスリリース

### UI整理
- 右パネルを **3つのQTabWidget** に再編成（「編集」「GrabCut」「保存・確認」）
- タブ下部にナビゲーションボタン（前/次の画像・保存・Undo/Redo）を常時表示
- GrabCut系モード選択時にGrabCutタブへ、通常編集モード選択時に編集タブへ自動切替
- 「ヘルプ」メニューに「このアプリについて」ダイアログを追加
- 「設定」メニューに「設定を初期化」を追加

### 設定の自動保存 (QSettings)
- `core/app_settings.py` を新設。Organization="KC-nishihana", App="COLMAPMaskEditor"
- 保存項目: ウィンドウ geometry・スプリッタ位置・右パネルタブ番号・ブラシサイズ・GrabCut設定一式・最後に開いたフォルダ
- 数値設定は (lo, hi) クランプを適用。bool はストリング変換を正しく処理

### 未確定状態の保護
- GrabCut PREVIEW / HINT_EDITING 中に画像切替・プロジェクト変更・終了しようとした場合に **3択ダイアログ**（適用/破棄/キャンセル）を表示
- 未保存マスクがある状態で画像切替・終了しようとした場合に **3択ダイアログ**（保存/破棄/キャンセル）を表示

### Worker終了処理の強化
- `finished` / `failed` / `cancelled` の全シグナルで `thread.quit()` を呼び出し
- GrabCut処理中にウィンドウを閉じた場合は `_close_pending` フラグでキャンセル後に再度 `close()` を実行（`QThread.terminate()` は使用しない）

### バージョン管理の一元化
- `core/version.py` を新設。`APP_VERSION = "0.5.1"` を唯一の定義元とする
- `app.py` で `QCoreApplication.setOrganizationName` / `setApplicationName` を設定

### 新規テスト (合計155テスト)
- `test_version.py` — バージョン文字列整合・README確認
- `test_app_settings.py` — デフォルト値・保存/復元・クランプ・bool変換・リセット
- `test_right_panel_tabs.py` — タブ存在・ラベル・自動切替・設定復元
- `test_pending_state_guard.py` — GrabCut保護・未保存マスク保護の3択動作
- `test_grabcut_thread_integration.py` — QThread統合・シグナル伝達・Worker参照解放

### ツール追加
- `tools/benchmark_grabcut.py` — NumPy合成画像でFHD/4K/8K GrabCutの処理時間・メモリを計測

---

## v0.5 (2026-06) — UI/UX改善

- GrabCut補正ヒントボタンにアクティブ状態（pressed/checked）表示を追加
- GrabCutステータスラベルを右パネルに常時表示
- 保存失敗時の通知ダイアログを追加
- 画像切替時の未保存確認ダイアログを追加
- マスク統計表示にデバウンス処理を追加（高頻度更新の抑制）
- その他9件のUI/UX改善

---

## v0.4B (2025-12)

- GrabCut再推定機能追加（`GC_INIT_WITH_MASK`による繰り返し改善）
- 対象ヒント（緑）・背景ヒント（赤）のキャンバス描画
- ヒントのUndo/Redo・全消去（Ctrl+Z / Ctrl+Y / Ctrl+Shift+Z）
- 既存マスクの0領域をGrabCut背景制約として使用するオプション
- 補正UIグループを右パネルに追加
- テスト4本追加（合計139テスト）

---

## v0.4A.1

- 大画像対応: ROI切り出し・自動縮小（INTER_AREA）・元解像度復元（INTER_NEAREST）
- GrabCutをGUIスレッド外（QThread + Worker）で実行
- プログレスダイアログ・キャンセルボタン
- 処理中のUI操作を無効化
- 例外処理強化・自動テスト追加

---

## v0.4A

- GrabCutによる半自動マスク生成機能を追加
- 有効化（G）・除外（Shift+G）・置換（Ctrl+G）の3モード
- 候補プレビュー表示（Enter適用 / Esc キャンセル）
- 後処理オプション（膨張・収縮・カーネルサイズ）

---

## v0.3

- ポリゴン追加/削除モード
- 差分表示（保存前後のマスクを色分け表示）
- モルフォロジー処理（膨張・収縮・穴埋め）
- 小領域除去
- 品質チェック・CSVログ出力
- COLMAP互換マスク一括出力

---

## v0.2

- 矩形追加/削除モード追加

---

## v0.1

- 初版: ブラシ編集・Undo/Redo・ズーム・パン
