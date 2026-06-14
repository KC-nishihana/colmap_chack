# CHANGELOG

## v0.6.1

### CUDA検証強化
- sam2._Cのimport確認に加え、get_connected_components CUDAカーネルを直接実行
- fill_holes後処理を実機検証
- セットアップの完全成功・未検証を終了コードで区別 (0/1/2/3/4)・-BuildOnly追加
- マニフェストのverified管理を厳格化 (実機フル検証後のみtrue)

### 実機統合テスト
- 実際のQProcess Worker経由でモデルロード・Embedding・推論を検証
- 日本語・全角スペースパスを実機検証
- Worker終了後のGPUプロセス解放 (PID別) を検証
- Worker再起動後の再推論を検証

### ドキュメント
- 右パネルを4タブ表記へ統一
- CUDA検証手順と終了コードを追記

## v0.6

- SAM 2.1 AIセグメンテーション追加
- WindowsネイティブCUDA対応
- SAM 2 CUDA拡張必須化
- QProcess常駐Worker追加
- 正クリック・負クリック・矩形プロンプト
- 最大3候補マスク
- AIマスク追加・除外・置換
- AIプロセスクラッシュ分離
- CUDA環境診断・検証スクリプト
- RTX 4090実機テスト
- 設定スキーマ v1→v2 移行 (AI設定追加・既存設定を保持)
- 右パネルを4タブ化 (編集 / GrabCut / AIセグメント / 保存・確認)

## v0.5.1

- UI整理 (3タブ化)・QSettings設定保存・未確定GrabCut保護・安定性向上

## v0.5 / v0.4B

- GrabCut再推定・ヒント描画・補正UI

## v0.4A.1

- 大画像対応・GUIスレッド分離・例外処理強化・テスト追加

## v0.4A

- GrabCutによる半自動マスク生成機能
