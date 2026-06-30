# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SketchSort contributors
"""01_basic.py — sketchsort.search() の最小例

合成データで近傍ペアを取得し、戻り値の dtype・形・典型ペアを表示する。

セットアップ: 100 個の "中心ベクトル" を作り、各中心に小さなノイズを乗せて
50 個ずつ複製 → 5000×64 の行列。中心が同じペアは cos_dist が小さくなる。
"""

import numpy as np
import sketchsort

print(f"sketchsort version: {sketchsort.__version__}")

# 100 クラスタ × 50 サンプル = 5000 ベクトル
rng = np.random.default_rng(0)
N_CLUSTERS, PER_CLUSTER, D = 100, 50, 64
centers = rng.normal(size=(N_CLUSTERS, D)).astype(np.float32)
noise   = rng.normal(scale=0.02, size=(N_CLUSTERS, PER_CLUSTER, D)).astype(np.float32)
X = (centers[:, None, :] + noise).reshape(-1, D)
X /= np.linalg.norm(X, axis=1, keepdims=True)

print(f"input: {X.shape}, dtype={X.dtype}")

# 同じクラスタの 2 サンプルは cos_dist がほぼ 0、別クラスタは ~1。
# 閾値 0.01 でクラスタ内ペアだけ拾える。
pairs = sketchsort.search(X, cos_dist=0.01, seed=42)

print(f"\nfound {len(pairs)} pairs with cos_dist <= 0.01")
print(f"dtype: {pairs.dtype}")

# 期待値: 同一クラスタ内の C(50, 2) = 1225 ペア × 100 クラスタ = 122500
print(f"expected (no false negative): {N_CLUSTERS * PER_CLUSTER * (PER_CLUSTER - 1) // 2}")
print(f"detection ratio: {len(pairs) / (N_CLUSTERS * PER_CLUSTER * (PER_CLUSTER - 1) // 2):.1%}")

# 同一クラスタか別クラスタか
def cluster_of(idx: int) -> int:
    return idx // PER_CLUSTER

same_cluster = sum(1 for r in pairs if cluster_of(int(r["id1"])) == cluster_of(int(r["id2"])))
print(f"\nsame-cluster pairs:  {same_cluster}")
print(f"cross-cluster pairs: {len(pairs) - same_cluster}  (false positives)")

print(f"\nfirst 5 pairs:")
for row in pairs[:5]:
    c1, c2 = cluster_of(int(row['id1'])), cluster_of(int(row['id2']))
    tag = "same" if c1 == c2 else "DIFF"
    print(f"  id1={int(row['id1']):>4} id2={int(row['id2']):>4} cos_dist={float(row['cos_dist']):.6f} ({tag})")

print(f"\ncos_dist stats:")
print(f"  min:  {pairs['cos_dist'].min():.6f}")
print(f"  max:  {pairs['cos_dist'].max():.6f}")
print(f"  mean: {pairs['cos_dist'].mean():.6f}")
