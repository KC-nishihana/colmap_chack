"""
SAM 2.1 Worker (子プロセス) パッケージ。

このパッケージは QProcess で起動される子プロセス側でのみ実行される。
torch / sam2 / sam2._C はここ (worker) の中だけで import する。
GUI プロセスからこのパッケージを import してはならない
(import しても __init__ は torch を読み込まないが、規約として禁止)。
"""
