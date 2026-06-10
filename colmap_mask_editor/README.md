# COLMAP Mask Editor v0.2

COLMAP形式のプロジェクトに対応した、マスク画像の手動確認・修正GUIツールです。
v0.2 では品質チェック・ステータスフィルタ・COLMAP互換出力・CSVログ出力を追加しました。

## 動作環境

- OS: Windows 11
- Python: 3.12系
- 依存ライブラリ: PySide6, OpenCV, NumPy, Pillow

## インストール

```bash
cd colmap_mask_editor
pip install -r requirements.txt
```

## 起動方法

```bash
python app.py
```

プロジェクトフォルダを引数で指定することもできます:

```bash
python app.py C:\path\to\project
```

## プロジェクトフォルダの構成

```
project/
├─ images/
│  ├─ IMG_0001.jpg
│  ├─ IMG_0002.jpg
│  └─ sub/
│     └─ IMG_0003.jpg
├─ sparse/           (オプション)
│  └─ 0/
│     └─ images.txt
└─ masks/            (オプション)
   ├─ IMG_0001.png
   └─ sub/
      └─ IMG_0003.png
```

- `sparse/0/images.txt` がある場合はCOLMAP登録画像の順序で一覧表示します
- `masks/` がない場合や対応マスクがない場合は、空マスクで開始します

## 基本的な使い方

1. `python app.py` で起動
2. **ファイル > プロジェクトを開く** でプロジェクトフォルダを選択
3. 左ペインで編集したい画像を選択
4. 中央キャンバスに画像とマスク(赤い半透明)が表示される
5. 左クリックドラッグでマスクを追加、右クリックドラッグで削除
6. **S** または **Ctrl+S** で元のマスクファイルを上書き保存（マスクがない場合は `masks/` に新規作成）

## キーボードショートカット

| キー | 操作 |
|------|------|
| S / Ctrl+S | 保存 |
| A | 前の画像 |
| D | 次の画像 |
| Z / Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| M | マスク表示ON/OFF |
| + / = | ブラシサイズを大きくする |
| - | ブラシサイズを小さくする |
| ホイール | ズーム |
| 中ボタンドラッグ | パン |

## 出力ファイル

### 編集済みマスク

元の `masks/` フォルダ内のマスクファイルを直接上書き保存します。

- 既存マスクがある場合: そのファイルをそのまま上書き
- マスクが存在しない場合: `masks/<画像名>.png` として新規作成

```
project/masks/
├─ IMG_0001.png    ← 上書き or 新規作成
├─ IMG_0002.png    ← 上書き or 新規作成
└─ sub/
   └─ IMG_0003.png ← 上書き or 新規作成
```

### COLMAP互換マスク (オプション)

右パネルの「masks_colmap/ にも保存」を有効にすると:

```
project/masks_colmap/
├─ IMG_0001.jpg.png
└─ sub/
   └─ IMG_0003.jpg.png
```

### ログ

```
project/mask_edit_log.csv
```

保存のたびに以下の情報が追記されます:

```csv
image_path,input_mask_path,edited_mask_path,status,width,height,mask_width,mask_height
```

## マスク画像の仕様

- 形式: グレースケール PNG
- マスク領域: 255 (白)
- 非マスク領域: 0 (黒)
- 表示: 赤色の半透明オーバーレイ

## 状態アイコン (左ペイン)

| 表示 | 意味 |
|------|------|
| ✓ マスクあり | 既存マスクが見つかった |
| マスクなし | マスクが存在しない(空マスクで開始) |
| ● 未保存 | 編集済みで未保存 |
| ⚠ サイズ不一致 | マスクサイズが画像と異なる |

サイズ不一致の場合、右パネルの「画像サイズに合わせてリサイズ」ボタンで修正できます。

## 状態アイコン (左ペイン) v0.2

| 表示 | 意味 |
|------|------|
| [OK] | マスクが正常 |
| [マスクなし] | マスクが存在しない |
| [未保存] | 編集済みで未保存 |
| [サイズ不一致] | マスクサイズが画像と異なる |
| [空マスク] | マスクが全て0（未マスク） |
| [全面マスク] | マスク率 95% 以上 |
| [中間値あり] | 0/255 以外のピクセルを含む（自動2値化済み） |
| [画像エラー] | 画像ファイルを読み込めない |
| [マスクエラー] | マスクファイルを読み込めない |
| [要確認] | チェック実行前、または手動確認が必要 |

---

## v0.2 新機能

### 1. マスク品質チェック

左ペイン上部のフィルタと、右パネルの「一括チェック」ボタンで品質管理を行います。

```
状態一覧:
  ok / no_mask / size_mismatch / empty_mask / full_mask
  intermediate_values / unreadable_image / unreadable_mask
  not_saved / needs_check
```

### 2. ステータス別フィルタ

左ペイン上部のプルダウンで、以下の条件で画像一覧を絞り込めます：

```
すべて / 正常 / 要確認 / 未保存 / マスクなし
サイズ不一致 / 空マスク / 全面マスク / 中間値あり / 読み込みエラー
```

### 3. 一括チェックボタン

右パネル「品質チェック」の「一括チェック」ボタンを押すと：
- 全画像のマスク品質をチェック
- 一覧のステータス表示を更新
- ステータスごとの枚数サマリをダイアログで表示

### 4. COLMAP互換出力

「COLMAP互換出力」ボタンで `masks_edited/` の全マスクを `masks_colmap/` に一括出力します。

```
入力: masks_edited/sub/IMG_0001.png
出力: masks_colmap/sub/IMG_0001.jpg.png
```

出力ルール: 元の画像ファイル名 + `.png`

### 5. 保存先

通常保存は `masks_edited/` に出力されます。

```
project/masks_edited/
├─ IMG_0001.png      ← 編集済みマスク
└─ sub/
   └─ IMG_0003.png
```

「保存時にCOLMAP互換マスクも出力する」を ON にすると保存と同時に `masks_colmap/` にも出力します。

### 6. チェックログCSV出力

「ログCSV出力」ボタンで `project/mask_check_log.csv` を出力します。

出力列:
```
image_path, input_mask_path, edited_mask_path, colmap_mask_path,
status, width, height, mask_width, mask_height,
mask_ratio, has_intermediate_values, is_empty_mask, is_full_mask, note
```

パスはプロジェクトフォルダからの相対パスで出力されます。
文字コードは UTF-8 BOM付き（Windows Excelで開いても文字化けしません）。

### 7. マスク統計表示

右パネル「マスク統計」に現在表示中の画像の情報が表示されます：
- 画像サイズ / マスクサイズ / マスク率 / 状態
- 入力マスク・編集済みマスク・COLMAPマスクのパス

### 8. 中間値の自動2値化

読み込んだマスクに 0/255 以外の値が含まれる場合、128 以上を 255、未満を 0 に2値化します。
チェック結果に `has_intermediate_values=True` が記録されます。

---

## ファイル構成 (v0.2)

```
colmap_mask_editor/
├─ app.py
├─ core/
│  ├─ mask_checker.py    # 品質チェック・ステータス判定
│  ├─ colmap_export.py   # COLMAP互換マスク一括出力
│  ├─ check_log.py       # チェックログCSV出力
│  ├─ mask_io.py         # マスク読み書き・パス解決
│  ├─ mask_ops.py        # ブラシ編集・Undo/Redo
│  └─ project_loader.py  # プロジェクト構造解析
└─ ui/
   ├─ main_window.py     # メインウィンドウ
   ├─ image_canvas.py    # キャンバス・ブラシ描画
   └─ image_list_panel.py # 画像一覧・フィルタ
```

## 注意事項

- 日本語パス・全角スペースを含むパスに対応しています
- 大きな画像でも操作が重くならないよう、表示はスケールされた合成画像を使用しています
- Undo は最大50ステップ保持されます
- `masks_edited/` が保存先です（v0.1 の `masks/` 上書き動作から変更）
