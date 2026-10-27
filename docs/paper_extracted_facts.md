# TrajRL Reproduction — Paper Grounding Materials

This folder is the **ground-truth, paper-and-code-anchored** knowledge base I
captured before/during the reproduction attempt. Every entry cites either
(ICDE 2025 paper) or (Xfc30/TrajRL repository). No values are fabricated.

## 1. Paper bibliographic anchor

- Title: Self-Supervised Trajectory Representation Learning with Multi-Scale
  Spatio-Temporal Feature Exploration
- Authors: Hong Xia, Xiao Zhang, Yuan Cao (Ocean University of China);
  Lei Cao (The University of Arizona); Yanwei Yu*, Junyu Dong (Ocean University of China)
- Venue: IEEE 41st International Conference on Data Engineering (ICDE 2025),
  Hong Kong, May 19–23, 2025, pp. 779–792
- DOI: 10.1109/ICDE65448.2025.00064
- Acceptance rate: 300 / 1465 = 20.5 %
- Code: https://github.com/Xfc30/TrajRL
- Page: https://www.computer.org/csdl/proceedings-article/icde/2025/360300a779/26FZzwKY6nm
- dblp: https://dblp.org/rec/conf/icde/XiaZCCYD25

## 2. Datasets (from paper + repo)

| Dataset | Description | Where to get it |
|---------|-------------|-----------------|
| Porto | Taxi GPS traces in Porto (Kaggle / UCI) | LibCity-Datasets Porto bundle, https://drive.google.com/... (see below) |
| T-Drive | Taxi GPS in Beijing, 2 Feb – 8 Feb 2008 | https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-set/ |
| Both share the same preprocessing: road network matching, segment ID vocabulary, time stamps in seconds | paper §4 | — |

Paper claim: "Extensive experiments demonstrate the superiority of our TrajRL
as compared to state-of-the-art baselines on **two** real-world datasets across
various downstream tasks." → Porto is one of them; the second dataset is
Beijing/T-Drive (inferred from Table II having different metrics ACC / F1 / AUC
used by the Porto classifier setting, while the second task uses different
metric groupings).

## 3. Downstream tasks

| Task | Description | Metrics |
|------|-------------|---------|
| Trajectory classification | per-segment / per-`usrid` classification | Micro-F1, Macro-F1 (Table I); ACC, F1, AUC (Table II) |
| Travel time estimation | regress trip duration in minutes | MAE, MAPE, RMSE |
| Most similar trajectory search | k-nearest-neighbour in embedding space | MR (Mean Rank, ↓), HR@1, HR@5 |

All metrics follow LibCity's standard trajectory embedding evaluator
(`MultiClassificationEvaluator`).

## 4. Baselines (Table I / Table II)

From the paper §4 (verbatim citation list):

1. traj2vec — IJCNN-17
2. t2vec — ICDE-18
3. Trembr — TIST-20
4. PIM — IJCAI-21
5. Toast — CIKM-21
6. JCLRNT — CIKM-22
7. LightPath — KDD-23
8. TrajCL — ICDE-23
9. START — ICDE-23
10. **TrajRL** (this paper)

Baselines are sourced from either the original repos or the LibCity framework.

## 5. TABLE I — Porto main results (verbatim from paper, raw OCR transcript)

```
Method          | Trajectory Classification            | Travel Time Estimation                    | Most Similar Trajectory Search
                | Micro-F1 | Macro-F1 | Recall@5       | MAE     | MAPE | RMSE     | MR     | HR@1  | HR@5
traj2vec [IJCNN-17] | 0.066 | 0.041 | 0.181 | 1.549 | 0.241 | 2.346 | 30.69 | 0.543 | 0.732
t2vec [ICDE-18]     | 0.064 | 0.053 | 0.189 | 1.521 | 0.237 | 2.324 | 11.54 | 0.769 | 0.858
Trembr [TIST-20]    | 0.071 | 0.049 | 0.192 | 1.480 | 0.226 | 2.164 | 4.635 | 0.846 | 0.929
PIM [IJCAI-21]      | 0.059 | 0.037 | 0.158 | 1.568 | 0.247 | 2.642 | 20.75 | 0.617 | 0.764
Toast [CIKM-21]     | 0.062 | 0.035 | 0.184 | 1.624 | 0.246 | 2.445 | 22.61 | 0.684 | 0.789
JCLRNT [CIKM-22]    | 0.069 | 0.046 | 0.191 | 1.647 | 0.238 | 2.334 | 5.017 | 0.852 | 0.931
LightPath [KDD-23]  | 0.068 | 0.049 | 0.180 | 1.561 | 0.242 | 2.362 | 13.26 | 0.607 | 0.788
TrajCL [ICDE-23]    | 0.056 | 0.033 | 0.148 | 1.960 | 0.269 | 2.963 | 5.461 | 0.834 | 0.915
START [ICDE-23]     | 0.084 | 0.061 | 0.238 | 1.374 | 0.219 | 2.185 | 1.927 | 0.921 | 0.974
TrajRL (ours)       | 0.092 | 0.070 | 0.249 | 1.259 | 0.202 | 2.051 | 1.241 | 0.988 | 0.997
Improvement         | 9.52% | 14.75% | 4.62% | 8.36% | 7.76% | 6.13% | 35.6% | 8.36% | 2.36%
```

(Caption: ↑ larger better, ↓ smaller better. Travel-time is in minutes.
Bold = best, underline = 2nd best.)

## 6. TABLE II — Second dataset (most likely T-Drive/Beijing)

From the reproduction transcript, Table II lists the same 10 rows but with
ACC / F1 / AUC columns. The full numeric block was **truncated in the only
public transcript I found**. The 8th-baseline row only partially reaches
column 4. Cells after HR@1 for row 8 onwards are not available; therefore
this table is **partially filled** in the reproduction notebook, marked as
"原论文未公开" rather than fabricated.

Numbers that *were* captured (from the transcript):

```
traj2vec [IJCNN-17]: ACC 0.802, F1 0.841, AUC 0.869, MAE 10.14, MAPE 0.399,
                     RMSE 57.28, MR 10.26, HR@1 0.573, HR@5 0.758
t2vec [ICDE-18]:     ACC 0.799, F1 0.859, AUC 0.861, MAE 10.27, MAPE 0.372,
                     RMSE 56.15, MR 9.684, HR@1 0.763, HR@5 0.905
Trembr [TIST-20]:    ACC 0.818, F1 0.871, AUC 0.880, MAE 9.997, MAPE 0.342,
                     RMSE 36.97, MR 2.509, HR@1 0.884, HR@5 0.952
PIM [IJCAI-21]:      ACC 0.803, F1 0.874, AUC 0.866, MAE 11.56, MAPE 0.486,
                     RMSE 35.78, MR 26.75, HR@1 0.731, HR@5 0.874
Toast [CIKM-21]:     ACC 0.810, F1 0.870, AUC 0.871, MAE 10.69, MAPE 0.350,
                     RMSE 57.41, MR 29.53, HR@1 0.611, HR@5 0.746
JCLRNT [CIKM-22]:    ACC 0.825, F1 0.867, AUC 0.883, MAE 10.73, MAPE 0.406,
                     RMSE 57.84, MR  6.382, HR@1 0.802, HR@5 0.927
LightPath [KDD-23]:  ACC 0.802, F1 0.859, AUC 0.865, MAE 10.66, MAPE 0.506,
                     RMSE 55.80, MR  6.484, (next col unclear)
```

The remaining rows for LightPath, TrajCL, START, and TrajRL (full row +
Improvement row) were not captured in the public transcript I found.

## 7. Ablation variants (paper §4, Figure 2 / Table III)

| Variant id | Description (paper §4) |
|------------|------------------------|
| w/o MFTime-Enc | Remove entire multi-faceted time encoder |
| w/o Time Interval | Remove only the time-interval embedding |
| w/o MSTraj-Enc | Replace multi-scale trajectory encoder with plain Transformer |
| w/o InPatch-Enc | Replace InPatch encoder with a linear layer |
| w/o TrajRec | Remove the trajectory reconstruction (MLM) self-supervised task |
| w/o CL | Remove the contrastive-learning self-supervised task |

The exact numeric ablation table was **not** in the public transcript; we
record that the qualitative claims from the paper are:

> • The variant "w/o MFTime-Enc" performs the worst, demonstrating the
>   importance of the multi-faceted time encoder.
> • "w/o MSTraj-Enc" shows a clear drop, indicating the multi-scale
>   trajectory encoder is more suitable than a vanilla Transformer for
>   trajectory spatio-temporal feature extraction.
> • "w/o TrajRec" and "w/o CL" both degrade performance, confirming the
>   importance of the dual self-supervised signal.

## 8. Efficiency experiment (paper §4, Figure 3)

The paper compares TrajRL vs. START for inference latency. The exact numeric
table is not in the public transcript I found; the qualitative claim:

> "TrajRL shows a certain efficiency advantage over the SOTA TRL method
> START. Although our multi-scale trajectory encoder is a deeper model, the
> experimental results show that TrajRL's inference speed does not decrease
> compared to other self-attention-based models."
>
> Two reasons:
> 1. Self-attention in the CrossPatch encoder is computed on trajectory
>    segments rather than per point, which significantly reduces time
>    complexity.
> 2. Because the InPatch encoder merges trajectory segments, the input
>    sequence length and parameter count of lower scale blocks decrease
>    exponentially compared to upper scale blocks.

## 9. Hyperparameters captured from `porto.json` (Xfc30/TrajRL/main)

```json
{
  "n_layers": 6,                  // number of scale_block layers
  "d_model": 256,                 // hidden dim
  "attn_heads": 8,                // transformer heads
  "max_epoch": 30,
  "batch_size": 64,
  "grad_accmu_steps": 1,
  "learning_rate": 2e-4,
  "dataset": "porto",
  "roadnetwork": "porto_roadmap_edge_porto_True_1_merge",
  "geo_file":   "porto_roadmap_edge_porto_True_1_merge_withdegree",
  "rel_file":   "porto_roadmap_edge_porto_True_1_merge_withdegree",
  "merge": true,
  "min_freq": 1,
  "seq_len": 128,                 // trajectory length cap
  "test_every": 50,
  "temperature": 0.05,            // contrastive temperature
  "contra_loss_type": "simclr",
  "classify_label": "usrid",
  "type_ln": "post",
  "add_cls": false,
  "add_time_in_day": true,        // period-of-day encoding
  "add_day_in_week": true,        // period-of-week encoding
  "add_time_interval": true,      // time-interval encoding
  "add_pe": true,                 // positional encoding
  "add_temporal_bias": true,
  "temporal_bias_dim": 64,
  "use_mins_interval": false,
  "add_gat": true,                // GAT-based segment encoder
  "gat_heads_per_layer": [8, 16, 1],
  "gat_features_per_layer": [16, 16, 256],
  "gat_dropout": 0.1,
  "gat_K": 1,
  "gat_avg_last": true,
  "load_trans_prob": true,
  "append_degree2gcn": true,
  "normal_feature": false,
  "pooling": "mean",
  "win_size": 4,                  // patch window
  "num_blocks": 4,                // scale blocks in multi-scale stack
  "max_time_scale_s": 1500,
  "time_interval_scales": [10, 30, 60]
}
```

## 10. Executor configuration (from PatchContrastiveSplitMLMExecutor.json)

```json
{
  "batch_size": 32,
  "grad_accmu_steps": 1,
  "max_epoch": 20,
  "learner": "adamw",
  "learning_rate": 2e-4,
  "lr_eta_min": 0,
  "lr_warmup_epoch": 4,
  "lr_warmup_init": 1e-6,
  "lr_decay": true,
  "lr_scheduler": "cosinelr",
  "lr_decay_ratio": 0.1,
  "t_in_epochs": true,
  "clip_grad_norm": true,
  "max_grad_norm": 5,
  "use_early_stop": true,
  "patience": 50,
  "test_every": 10,
  "log_batch": 500,
  "log_every": 1,
  "n_views": 2,
  "similarity": "cosine",
  "temperature": 0.05,
  "contra_loss_type": "simclr",
  "contra_ratio": 1.0,
  "mlm_ratio": 1.0,
  "data_argument1": null,         // overridden by CLI --out_data_argument1 trun
  "data_argument2": null,         // overridden by CLI --out_data_argument2 shift
  "cutoff_row_rate": 0.2,
  "cutoff_column_rate": 0.2,
  "cutoff_random_rate": 0.2,
  "sample_rate": 0.2
}
```

Note the conflict between `porto.json` (batch_size=64, max_epoch=30) and the
executor default (batch_size=32, max_epoch=20). In practice the README uses
the CLI override of the executor config (batch_size 32 used in the bundled
example).

## 11. CLI used to launch (from README)

```bash
python run_model.py \
  --task trajectory_embedding \
  --model TrajRLContrastiveLM \
  --dataset porto \
  --config porto \
  --gpu_id 0 \
  --split true \
  --masking_ratio 0.2 \
  --distribution geometric \
  --out_data_argument1 trun \
  --out_data_argument2 shift
```

## 12. Code structure (Xfc30/TrajRL)

```
TrajRL/
├── libcity/
│   ├── model/
│   │   ├── layers/
│   │   │   ├── attention.py      # MultiheadAttention
│   │   │   ├── basics.py         # Transpose, get_activation_fn
│   │   │   ├── nodeembedding.py  # NodeEmbedding (segment + time + GAT)
│   │   │   ├── patch_mask.py     # Patcher, PatchMasker, MaskedLanguageModel
│   │   │   ├── pos_encoding.py   # positional_encoding()
│   │   │   ├── revin.py          # RevIN
│   │   │   ├── sequencepooler.py # SequenceMeanPoolLayer, WeightedAverage
│   │   │   └── transformer.py    # CrossEncoder, TransformerEncoderLayer
│   │   ├── loss.py               # NoFussCrossEntropyLoss + masked_*_np helpers
│   │   ├── trajectory_embedding/
│   │   │   └── trajrl.py         # TrajRLContrastiveLM, TrajRL, scale_block, Encoder, …
│   │   └── utils.py
│   ├── executor/                  # training + evaluation loop
│   ├── evaluator/                 # metric calculators
│   ├── data/                      # dataset loaders (ContrastiveLMDataset)
│   ├── pipeline.py
│   ├── config/
│   │   ├── PatchContrastiveSplitMLMExecutor.json   # main contrastive+MLM config
│   │   ├── PatchExecutor.json
│   │   ├── PatchLMExecutor.json
│   │   ├── PatchContrastiveExecutor.json
│   │   ├── PatchContrastiveSplitExecutor.json
│   │   ├── PatchContrastiveMLMExecutor.json
│   │   └── task_config.json
│   └── utils.py
├── libcity_data/                  # NOT IN REPO — must download from LibCity-Datasets
├── porto.json                     # dataset-specific configs
├── run_model.py
├── setup_vocab.py
└── trans_prob.py
```

This file intentionally avoids any value not directly observable in either
the paper text or the public GitHub repository.
