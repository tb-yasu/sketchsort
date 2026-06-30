# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SketchSort contributors
"""02_real_data.py — 同梱の dat/sample.txt を使った実データ例

37749 個の 32 次元ベクトルから:
- 全近傍ペアを取得
- 距離順 top-10 を表示
- 特定 id にぶら下がるペアだけ抽出
- 距離分布を簡単に集計
"""

import os
import numpy as np
import sketchsort

HERE = os.path.dirname(__file__)
SAMPLE = os.path.join(HERE, "..", "dat", "sample.txt")

X = np.loadtxt(SAMPLE, dtype=np.float32)
print(f"loaded {X.shape[0]} vectors of dim {X.shape[1]}")

# しきい値 cos_dist <= 0.01 の全ペア
pairs = sketchsort.search(X, cos_dist=0.01, seed=42)
print(f"\n{len(pairs)} pairs within cos_dist <= 0.01")

# 距離順 top-10 (最も似ているペア)
top10 = pairs[np.argsort(pairs["cos_dist"])][:10]
print(f"\ntop-10 nearest pairs:")
print(f"  {'id1':>6} {'id2':>6}  cos_dist")
for r in top10:
    print(f"  {int(r['id1']):>6} {int(r['id2']):>6}  {float(r['cos_dist']):.6e}")

# 特定 id にひっかかるペアだけ抽出
target_id = int(top10[0]["id1"])
hits = pairs[(pairs["id1"] == target_id) | (pairs["id2"] == target_id)]
print(f"\nid {target_id} appears in {len(hits)} pairs")

# 距離分布のヒストグラム (テキスト)
print(f"\ncos_dist distribution (10 bins, 0..0.01):")
hist, edges = np.histogram(pairs["cos_dist"], bins=10, range=(0.0, 0.01))
peak = hist.max()
for h, e in zip(hist, edges):
    bar = "#" * int(40 * h / peak)
    print(f"  [{e:.4f}, ..) {h:>7}  {bar}")
