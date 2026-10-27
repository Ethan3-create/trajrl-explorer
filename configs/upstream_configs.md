porto.json
==========
这是论文 TrajRL 在 Porto 数据集上预训练所用的实验配置。
直接引自官方仓库 Xfc30/TrajRL @ main 分支。

{                                                                   ← 起点
  "n_layers": 6,
  "d_model": 256,
  "attn_heads": 8,
  "max_epoch": 30,
  "batch_size": 64,
  "grad_accmu_steps": 1,
  "learning_rate": 2e-4,
  "dataset": "porto",
  "roadnetwork": "porto_roadmap_edge_porto_True_1_merge",
  "geo_file": "porto_roadmap_edge_porto_True_1_merge_withdegree",
  "rel_file": "porto_roadmap_edge_porto_True_1_merge_withdegree",
  "merge": true,
  "min_freq": 1,
  "seq_len": 128,
  "test_every": 50,
  "temperature": 0.05,
  "contra_loss_type": "simclr",
  "classify_label": "usrid",
  "type_ln": "post",
  "add_cls": false,
  "add_time_in_day": true,
  "add_day_in_week": true,
  "add_time_interval": true,
  "add_pe": true,
  "add_temporal_bias": true,
  "temporal_bias_dim": 64,
  "use_mins_interval": false,
  "add_gat": true,
  "gat_heads_per_layer": [8, 16, 1],
  "gat_features_per_layer": [16, 16, 256],
  "gat_dropout": 0.1,
  "gat_K": 1,
  "gat_avg_last": true,
  "load_trans_prob": true,
  "append_degree2gcn": true,
  "normal_feature": false,
  "pooling": "mean",

  "win_size": 4,
  "num_blocks": 4,
  "max_time_scale_s": 1500,
  "time_interval_scales": [10, 30, 60]
}                                                                   ← 终点


PatchContrastiveSplitMLMExecutor.json
=====================================
预训练用的执行器配置。
直引自 Xfc30/TrajRL/libcity/config/PatchContrastiveSplitMLMExecutor.json。

{                                                                   ← 起点
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
  "l2_reg": null,
  "n_views": 2,
  "similarity": "cosine",
  "temperature": 0.05,
  "contra_loss_type": "simclr",
  "contra_ratio": 1.0,
  "mlm_ratio": 1.0,
  "data_argument1": null,        ← CLI --out_data_argument1 trun
  "data_argument2": null,        ← CLI --out_data_argument2 shift
  "cutoff_row_rate": 0.2,
  "cutoff_column_rate": 0.2,
  "cutoff_random_rate": 0.2,
  "sample_rate": 0.2
}                                                                   ← 终点


README.md 全部命令
==================
预训练：
  python run_model.py --task trajectory_embedding \
      --model TrajRLContrastiveLM --dataset porto --config porto \
      --gpu_id 0 --split true \
      --masking_ratio 0.2 --distribution geometric \
      --out_data_argument1 trun --out_data_argument2 shift

下游/消融: 见 docs/ablation_plan.md 与 docs/reproduction_report.md §7。
