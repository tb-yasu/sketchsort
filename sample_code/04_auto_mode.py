# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SketchSort contributors
"""04_auto_mode.py — auto_mode で自動パラメータチューニング

ham_dist / num_blocks / num_chunks を手で決める代わりに、
missing_ratio (真の近傍を見逃す許容確率) を指定すると
SketchSort が内部で適切なパラメータを選ぶ。

3 段階の missing_ratio で実行して、検出ペア数と所要時間を比較。
"""

import os
import time
import numpy as np
import sketchsort

HERE = os.path.dirname(__file__)
SAMPLE = os.path.join(HERE, "..", "dat", "sample.txt")

X = np.loadtxt(SAMPLE, dtype=np.float32)
print(f"input: {X.shape[0]} vectors of dim {X.shape[1]}")
print(f"threshold: cos_dist <= 0.01")
print()
print(f"  {'missing_ratio':>15}  {'#pairs':>10}  {'time(s)':>8}")
print(f"  {'-'*15:>15}  {'-'*10:>10}  {'-'*8:>8}")

for mr in [0.1, 0.01, 0.001, 0.0001]:
    t0 = time.perf_counter()
    pairs = sketchsort.search(
        X,
        cos_dist=0.01,
        auto_mode=True,
        missing_ratio=mr,
        seed=42,
    )
    elapsed = time.perf_counter() - t0
    print(f"  {mr:>15g}  {len(pairs):>10}  {elapsed:>8.2f}")

# missing_ratio を厳しくするほど、より多くのスケッチを使うので
# 取りこぼしが減りペア数は増えるが、時間も増える。
# 0.1 = 10% 見逃し OK   -> 高速・粗い
# 0.0001 = 0.01% 見逃し -> 遅い・網羅的 (デフォルト)
print()
print("note: 厳しい missing_ratio ほど取りこぼしが減るが処理時間が増える。")
print("      研究用の探索的解析では 0.001 あたりが時間と網羅性のバランス良好。")
