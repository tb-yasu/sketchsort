# Sample code

`sketchsort` の代表的な使い方を 4 本に分けたサンプル集。すべて単体で動く。

| ファイル | 内容 | 入力 |
|---|---|---|
| [01_basic.py](01_basic.py) | `search()` の最小例。合成 5000×64 で API 形状を確認。 | 合成データ |
| [02_real_data.py](02_real_data.py) | 同梱の `dat/sample.txt` (37749×32) で top-K 抽出と分布集計。 | `dat/sample.txt` |
| [03_file_io.py](03_file_io.py) | `run_from_file()` でファイル経由処理。`search()` 結果と比較。 | `dat/sample.txt` |
| [04_auto_mode.py](04_auto_mode.py) | `auto_mode=True` + `missing_ratio` パラメータ感度。 | `dat/sample.txt` |

## 実行方法

リポジトリのルートからエディタブルインストール後、各スクリプトを直接実行:

```bash
pip install -e .
python sample_code/01_basic.py
python sample_code/02_real_data.py
python sample_code/03_file_io.py
python sample_code/04_auto_mode.py
```

すべて 1〜2 秒で終わる (04 は missing_ratio が厳しいケースで数秒)。
