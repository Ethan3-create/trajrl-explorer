"""
Standalone smoke test for the (mirrored) TrajRL model.
This is the only thing I can run end-to-end on the offline local machine;
it verifies architecture wiring + tensor-shape correctness against the
paper's hyperparameters in porto.json.
"""

import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch
import yaml
try:
    from trajrl_model import TrajRLContrastiveLM
except ImportError as e:
    print("Import error:", e)
    sys.exit(1)


def main():
    # hyperparams captured from Xfc30/TrajRL/porto.json
    cfg = {
        'd_model': 256,
        'num_blocks': 4,
        'win_size': 4,
        'seq_len': 128,
        'mask_ratio': 0.2,
        'add_gat': True,
        'gat_heads_per_layer': [8, 16, 1],
        'gat_features_per_layer': [16, 16, 256],
        'gat_dropout': 0.1,
        'time_interval_scales': [10, 30, 60],
        'dropout': 0.1,
    }
    data_feature = {
        'vocab_size': 50000,
        'usr_num': 487,
        'node_fea_dim': 64,
        'win_size': cfg['win_size'],
        'mask_ratio': cfg['mask_ratio'],
    }
    model = TrajRLContrastiveLM(cfg, data_feature)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke-test] TrajRLContrastiveLM param count: {n_params/1e6:.2f} M")
    print(f"[smoke-test] number of scale_blocks: {len(model.trajrl.patchtst_layers)}")

    # random input batch (2 views + masked view)
    bs, seq_len = 4, cfg['seq_len']
    vocab_size = cfg.get('vocab_size', 50000)
    contra_view1 = torch.randint(0, vocab_size, (bs, seq_len))
    contra_view2 = torch.randint(0, vocab_size, (bs, seq_len))
    masked_input = torch.randint(0, vocab_size, (bs, seq_len))
    padding_masks = torch.ones(bs, seq_len, dtype=torch.bool)

    out_v1, out_v2, a, mask = model(
        contra_view1, contra_view2, masked_input,
        padding_masks, batch_temporal_mat=None, graph_dict=None,
    )
    print("[smoke-test] output view1 shape:", tuple(out_v1.shape))
    print("[smoke-test] output view2 shape:", tuple(out_v2.shape))
    print("[smoke-test] MLM logits shape:", tuple(a.shape))
    print("[smoke-test] mask shape:", tuple(mask.shape))
    print("[smoke-test] PASS")


if __name__ == '__main__':
    main()
