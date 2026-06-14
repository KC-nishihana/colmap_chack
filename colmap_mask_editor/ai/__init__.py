"""
v0.6 AIセグメンテーション (SAM 2.1) のGUI側パッケージ。

重要: このパッケージおよび配下のモジュールは torch / sam2 / sam2._C を
import してはならない。SAM 2 関連の重い依存はすべて sam_backend Worker
(別プロセス) 側に閉じ込める。ここに置くのは

  - protocol      : Worker と GUI の JSON Lines プロトコル定義 (純粋ロジック)
  - model_registry: SAM2 モデルIDとファイル名の対応表
  - ai_prompt     : 正/負クリック・矩形プロンプトとUndo/Redo
  - ai_mask_ops   : 結果NPZの読み込み・候補統計・マスク適用
  - process_manager: QProcess による常駐Worker管理
  - ai_session    : 状態機械 (AiUiState) を持つ高レベルオーケストレーション

のみ。
"""
