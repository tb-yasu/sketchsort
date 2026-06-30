# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SketchSort contributors
"""03_file_io.py — run_from_file の使い方

巨大データで search() の戻り値がメモリに乗らない時の選択肢。
search() と run_from_file() の出力を比較して、id ペア集合が
完全一致することを確認する (cos_dist 値は float32 表現精度内で一致)。
"""

import os
import tempfile
import numpy as np
import sketchsort

HERE = os.path.dirname(__file__)
SAMPLE = os.path.join(HERE, "..", "dat", "sample.txt")

# run_from_file: file -> file ストリーム出力 (メモリ消費小)
with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
    out_path = tmp.name

sketchsort.run_from_file(
    input_path=SAMPLE,
    output_path=out_path,
    cos_dist=0.01,
    seed=42,
)

# 出力を読む (空白区切り 3 列)
file_pairs = np.loadtxt(
    out_path,
    dtype=[("id1", "<u4"), ("id2", "<u4"), ("cos_dist", "<f4")],
)
print(f"run_from_file output: {len(file_pairs)} lines, dtype={file_pairs.dtype.names}")
print(f"  first: {file_pairs[0]}")

# 比較対象: search() で同じパラメータ
X = np.loadtxt(SAMPLE, dtype=np.float32)
mem_pairs = sketchsort.search(X, cos_dist=0.01, seed=42)
print(f"\nsearch() output:      {len(mem_pairs)} pairs")
print(f"  first: {mem_pairs[0]}")

# id ペア集合の完全一致確認
file_set = set(zip(file_pairs["id1"].tolist(), file_pairs["id2"].tolist()))
mem_set  = set(zip(mem_pairs["id1"].tolist(),  mem_pairs["id2"].tolist()))

if file_set == mem_set:
    print(f"\nOK: id-pair sets match exactly ({len(file_set)} pairs)")
else:
    only_in_file = file_set - mem_set
    only_in_mem  = mem_set - file_set
    print(f"\nDIFF: file-only {len(only_in_file)}, mem-only {len(only_in_mem)}")

# cos_dist 値の数値比較 (top-5 で)
sorted_by_id = file_pairs[np.lexsort((file_pairs["id2"], file_pairs["id1"]))]
sorted_mem   = mem_pairs[np.lexsort((mem_pairs["id2"], mem_pairs["id1"]))]
np.testing.assert_allclose(
    sorted_by_id["cos_dist"][:100],
    sorted_mem["cos_dist"][:100],
    rtol=1e-4,
)
print("cos_dist values agree within rtol=1e-4 (first 100 pairs)")

os.unlink(out_path)
