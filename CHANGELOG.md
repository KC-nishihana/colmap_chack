# CHANGELOG

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
