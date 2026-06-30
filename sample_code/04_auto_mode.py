# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SketchSort contributors
"""04_auto_mode.py — `missing_ratio` で精度を調整するデモ

sketchsort.search() の **デフォルト動作 = auto mode** なので、ユーザは
`ham_dist` / `num_blocks` / `num_chunks` の 3 つを意識せず、
`cos_dist` (どこまでを近傍とみなすか) と `missing_ratio` (真の近傍を
見逃す許容確率) の 2 ノブだけで網羅性と速度を調整できる。

3 段階の missing_ratio で実行して、検出ペア数と所要時間を比較する。
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
    # auto mode は明示不要 — ham_dist / num_blocks / num_chunks を渡さなければ自動
    pairs = sketchsort.search(
        X,
        cos_dist=0.01,
        missing_ratio=mr,
        seed=42,
    )
    elapsed = time.perf_counter() - t0
    print(f"  {mr:>15g}  {len(pairs):>10}  {elapsed:>8.2f}")

print()
print("note: 厳しい missing_ratio ほど取りこぼしが減るが処理時間が増える。")
print("      研究用の探索的解析では 0.001 あたりが時間と網羅性のバランス良好。")
