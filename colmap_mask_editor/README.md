# COLMAP Mask Editor v0.1

COLMAP形式のプロジェクトに対応した、マスク画像の手動確認・修正GUIツールです。

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

## 注意事項

- 日本語パス・全角スペースを含むパスに対応しています
- 大きな画像でも操作が重くならないよう、表示はスケールされた合成画像を使用しています
- Undo は最大50ステップ保持されます
