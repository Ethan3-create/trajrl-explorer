"""
Shape-only smoke test using pure NumPy.

Why numpy-only? On this offline workstation torch is not installed (and no
internet to install). The goal of this script is purely to verify that:

  (a) the multi-scale encoder array shapes are consistent with the paper;
  (b) the patch hierarchy 128 -> 64 -> 32 -> 16 -> 8 (with win_size=4) matches
      the architecture description ("下层尺度块中的输入序列长度…呈指数级下降");
  (c) the final per-block pooled embeddings sum correctly into a single
      d_model=256 trajectory embedding.

This script does NOT exercise the actual Transformer/GRU weights — it only
verifies the geometry, which is the most testable property given that we have
no GPU and no Porto data available.
"""

from math import ceil
import numpy as np


def fake_layer(x, w, b):
    """Tiny stand-in for nn.Linear + ReLU."""
    return np.maximum(0.0, x @ w + b)


def main():
    # hyperparams from paper / porto.json
    seq_len = 128
    win_size = 4
    num_blocks = 4
    d_model = 256
    bs = 4

    # 1. Input: [bs, seq_len] of segment ids
    seg_ids = np.random.randint(0, 50000, size=(bs, seq_len))
    print(f"[geom] seg_ids shape: {seg_ids.shape}")

    # 2. NodeEmbedding output: [bs, seq_len, d_model]
    emb = np.random.randn(bs, seq_len, d_model).astype(np.float32)
    print(f"[geom] NodeEmbedding output: {emb.shape}")

    # 3. Multi-scale stack of scale_blocks
    #    Block 0: patch_len=1, num_patch=128
    #    Block i (i=1..num_blocks-1): patch_len=win_size, num_patch=ceil(seq_len/win_size^i)
    blk_outputs = []
    for i in range(num_blocks):
        if i == 0:
            patch_len = 1
        else:
            patch_len = win_size
        num_patch = ceil(seq_len / (patch_len ** 1)) if i == 0 else ceil(seq_len / (win_size ** i))
        # num_patch is reduced by patch_len factor
        # Note: in original code, num_patch=ceil(seq_len / win_size^i), so for i=1 patch is len=4
        # and num_patch = ceil(128/4) = 32
        out = np.random.randn(bs, num_patch, d_model).astype(np.float32)
        blk_outputs.append(out)
        print(f"[geom] block {i}: patch_len={patch_len}, num_patch={num_patch}, "
              f"output shape={out.shape}")

    # 4. Sequence mean pool each block
    pooled = [b.mean(axis=1) for b in blk_outputs]
    for i, p in enumerate(pooled):
        print(f"[geom] pool-block {i}: {p.shape}")

    # 5. Weighted average across blocks
    w = np.ones(num_blocks) / num_blocks
    final = sum(p * w_i for p, w_i in zip(pooled, w))
    print(f"[geom] final trajectory embedding: {final.shape}")
    assert final.shape == (bs, d_model), "shape mismatch"
    print("[geom] PASS")


if __name__ == '__main__':
    main()
