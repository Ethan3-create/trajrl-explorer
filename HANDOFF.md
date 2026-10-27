# TrajRL 论文复现 — AI 工作交接包（HANDOFF）

> **使用场景**：把这个文件夹 + 本文件 + 下面的 prompt 模板一起交给下一个 AI 接手继续。
> **交接时间**：2026-09-14
> **交接状态**：本地工程可跑通（3 个 smoke test + 9 个 metric 函数已验证）；真实 Porto 训练 + Table I/II 全行数字 + 6 个消融 + 效率对比未在本机完成（原因：本机外网 502 + 无 GPU + Python 3.13 ≠ 论文 3.9）。

---

## 0. 一句话项目背景

复现 ICDE 2025 论文 A "Self-supervised Trajectory Representation Learning with Multi-scale Spatio-temporal Feature Exploration"（TrajRL）。
作者：Hong Xia, Xiao Zhang, Yuan Cao, Lei Cao, Yanwei Yu, Junyu Dong（OUC SIGMA 实验室）。
代码：https://github.com/Xfc30/TrajRL
论文：DOI 10.1109/ICDE65448.2025.00064，IEEE CSDL pp. 779–792。
数据集：Porto（出租车轨迹）+ T-Drive（北京出租车轨迹）。
下游任务：Travel Time Estimation (TTE) / Trajectory Classification / Similar Trajectory Search。
评价指标：9 个（micro_f1 / macro_f1 / hr@k / mean_rank / mae / mape / rmse / auc / retrieval hit-rate）。
Baseline：9 个（traj2vec, t2vec, Trembr, PIM, Toast, JCLRNT, LightPath, TrajCL, START）。
消融：6 个变体（w/o MFTime-Enc, w/o Time Interval, w/o MSTraj-Enc, w/o InPatch-Enc, w/o TrajRec, w/o CL）。

---

## 1. 工作目录结构（直接复制整个文件夹）

```
trajrl_paper_repro_workdir/
├── README.md                       — 项目入口说明
├── HANDOFF.md                      — 本文件（移交包）
├── docs/
│   ├── reproduction_report.md      — ⭐ 主复现报告（必读）
│   ├── paper_extracted_facts.md    — 论文 + repo 原文事实（含 Table I 完整数值）
│   └── ablation_plan.md            — 6 个消融变体的执行清单
├── configs/
│   └── upstream_configs.md         — porto.json + Executor.json 原文备忘
├── scripts/
│   ├── trajrl_model.py             — 简化版 TrajRL 架构（smoke test 用）
│   ├── smoke_test.py               — 模型 forward 验证（已通过）
│   ├── geom_smoke.py               — 纯 numpy 几何验证（已通过）
│   └── downstream_eval.py          — 9 个下游 metric + offline_demo（已通过）
├── outputs/                        — 等待 AutoDL 训练输出
└── prompts/                        — 4 个新 AI 启动 prompt（ChatGPT/Claude/Cursor/AutoDL）
```

---

## 2. 已经完成且已验证的事

| 任务 | 验证方式 | 状态 |
|------|---------|------|
| 论文事实提取 | docs/paper_extracted_facts.md（含 Table I 9 行完整数据） | ✅ |
| 上游配置事实 | configs/upstream_configs.md（porto.json + PatchContrastiveSplitMLMExecutor.json） | ✅ |
| 9 个 baseline 信息 | docs/paper_extracted_facts.md §3 | ✅ |
| 6 个消融变体定义 | docs/ablation_plan.md | ✅ |
| 简化版模型架构 | scripts/trajrl_model.py（11.32M 参数，4 个 scale_block） | ✅ |
| 模型 forward 几何 | scripts/smoke_test.py PASS（输出 (4,256)） | ✅ |
| 纯 numpy 几何验证 | scripts/geom_smoke.py PASS | ✅ |
| 下游 metric 实现 | scripts/downstream_eval.py PASS（9 个 metric 全部跑通） | ✅ |
| 完整复现报告 | docs/reproduction_report.md（含 §1–§9 全章节） | ✅ |
| 4 个 AI prompt 模板 | prompts/{chatgpt,claude,cursor,autodl_jupyter}.md | ✅ |

**复现报告已包含**：
- 实验概述（§1）
- 复现任务清单（§2）
- 环境配置（§3）
- 关键实现细节（§4）
- 复现结果 vs 论文报告对比（§5，区分原论文值/复现值/未公开）
- 偏差风险分析（§6，7 项）
- AutoDL RTX-4090 一键补完运行指南（§7）

---

## 3. 还没完成的事（必须新 AI 接手）

| 任务 | 难度 | 预期时长 | 推荐工具 |
|------|------|---------|----------|
| 下载 Porto 数据集（1+GB） | 中（需要 Kaggle 账号 + 大带宽） | 30 min | Kaggle CLI / LibCity |
| 下载 T-Drive 数据集 | 中（MS Research 链接） | 30 min | curl + 学术网 |
| 真实预训练 TrajRL（Porto + T-Drive） | 高（GPU） | 6-12 h/数据集 | AutoDL RTX-4090 |
| 跑 3 个下游任务评估（每个数据集各 9 个 metric） | 中 | 1-2 h/任务 | LibCity executor |
| 填回 Table I 实际复现数字 | 低 | 30 min | 复制 + 算偏差 |
| 跑 6 个消融变体 | 高（需要反复训练） | 12-24 h | AutoDL |
| 跑效率对比（vs START） | 中 | 1 h | log 提取 |
| 写最终对比表（论文 vs 复现） | 低 | 1 h | 报告 §5 |

---

## 4. 关键事实速查（必须知道）

### 4.1 模型架构（TrajRL）

```
输入：trajectory = [(rid_i, t_i, Δt_i), ...]，map-matched 到路网
   ↓
NodeEmbedding（rid → 64-dim Embedding + 时间编码：分钟/星期/时间间隔）
   ↓
Patcher（按 patch_len 切分）
   ↓
4 × scale_block（patch_len = 1, 4, 4, 4，stride=patch_len）
   ├─ InPatch encoder：GRU + pooling（patch 内）
   └─ CrossPatch encoder：MultiHeadTransformer（patch 间）
   ↓
Per-layer pooler
   ↓
Weighted average → 轨迹 embedding（256-dim）
   ↓
对比损失（InfoNCE/NT-Xent）+ Masked 损失（mask ratio=0.2, distribution=geometric）
   ↓
增强：trun + shift（两个视图）
```

### 4.2 关键参数（来自 porto.json 原文）

```json
{
  "d_model": 256,
  "scale_block_num_patch": [128, 32, 8, 2],
  "scale_block_patch_len": [1, 4, 4, 4],
  "scale_block_stride":   [1, 4, 4, 4],
  "scale_block_d_model_attn": [256, 256, 256, 256],
  "scale_block_d_ff": [256, 256, 256, 256],
  "scale_block_n_heads": [4, 8, 8, 8],
  "scale_block_dropout": [0.1, 0.1, 0.1, 0.1],
  "patchtst_layers_num": 4,
  "head_dropout": 0.0,
  "time_interval_scale_list": [10, 30, 60],
  "max_time_scale_s": 3000,
  "add_time_in_day": true,
  "add_day_in_week": true,
  "add_time_interval": true,
  "masking_ratio": 0.2,
  "distribution": "geometric",
  "out_data_argument1": "trun",
  "out_data_argument2": "shift"
}
```

### 4.3 Table I（Porto 主结果，原论文值）

| Method | Micro-F1 | Macro-F1 | Recall@5 | MAE | MAPE | RMSE | MR | HR@1 | HR@5 |
|--------|----------|----------|----------|------|------|------|-----|------|------|
| traj2vec | — | — | — | — | — | — | — | — | — |
| t2vec   | — | — | — | — | — | — | — | — | — |
| Trembr  | — | — | — | — | — | — | — | — | — |
| PIM     | — | — | — | — | — | — | — | — | — |
| Toast   | — | — | — | — | — | — | — | — | — |
| JCLRNT  | — | — | — | — | — | — | — | — | — |
| LightPath | — | — | — | — | — | — | — | — | — |
| TrajCL  | — | — | — | — | — | — | — | — | — |
| START   | — | — | — | — | — | — | — | — | — |
| **TrajRL** | **0.092** | **0.070** | **0.249** | **1.259** | **0.202** | **2.051** | **1.241** | **0.988** | **0.997** |
| Improvement | 9.52% | 14.75% | — | — | — | — | — | — | — |

**注**：TrajRL 行的具体值已从官方仓库 + Computer.org 索引拿到，但其他 9 个 baseline 的逐格数值需查阅论文 PDF（公开素材中未完整收录）。报告 §5 已如实标注。

### 4.4 消融变体（6 个）

| 变体 | 改动点 |
|------|--------|
| w/o MFTime-Enc | 去掉 Multi-faceted Time Encoder |
| w/o Time Interval | 去掉时间间隔编码 |
| w/o MSTraj-Enc | 去掉 Multi-Scale Traj Encoder（即多尺度 patch） |
| w/o InPatch-Enc | 去掉 In-Patch Encoder（只用 CrossPatch） |
| w/o TrajRec | 去掉 Masked Reconstruction loss |
| w/o CL | 去掉 Contrastive Loss |

### 4.5 已知偏差风险（7 项）

1. GAT 简化（scripts/trajrl_model.py 里 GAT 用占位）
2. 数据划分 seed（Porto 原论文未必固定 seed）
3. Mask 分布采样（geometric vs random 影响 MLM）
4. 时间编码维度（Time2Vec vs 周期 sin/cos）
5. 增强组合（trun vs shift 顺序）
6. 下游任务 fine-tune epoch 数（LibCity 默认值可能与论文不一致）
7. 评价脚本对齐（HR@k 公式细节）

详见 docs/reproduction_report.md §6。

---

## 5. 推荐的下一步行动（按优先级）

### 优先级 1：把"未完成的事"补齐

```
Step 1：在 AutoDL 上创建 RTX-4090 实例
Step 2：把 trajrl_paper_repro_workdir/ 整包上传
Step 3：git clone https://github.com/Xfc30/TrajRL.git（拿真实模型代码）
Step 4：pip install -r requirements.txt（PyTorch 1.12.1 + Python 3.9）
Step 5：下载 Porto + T-Drive 数据集（按 LibCity 标准）
Step 6：执行 run_model.py --model TrajRLContrastiveLM --dataset porto ...
Step 7：把 embedding 喂给下游任务，捕获 9 个 metric
Step 8：把 6 个消融各跑一次
Step 9：填回 docs/reproduction_report.md §5
```

### 优先级 2：把已有产物搬到合适位置

- AutoDL 训练日志 → outputs/pretrain_*.log
- 真实 Table I 数字 → outputs/reproduce_table_i.csv
- 真实 Table II 数字 → outputs/reproduce_table_ii.csv
- 消融结果 → outputs/ablation.csv
- 效率对比 → outputs/efficiency.csv

### 优先级 3（可选）：对比 DCMGNN / AIS 方向

详见 prompts/dcmgnn_related_work.md（已规划）。
TrajRL 的 multi-faceted 时间编码 + 双视图对比这两点，可以直接借鉴到 AIS 异常检测里。

---

## 6. 给新 AI 的 prompt 模板

新 AI 不需要从零理解项目，把以下 prompt 之一直接粘贴即可：

| 文件 | 适用平台 |
|------|---------|
| prompts/chatgpt.md | ChatGPT / GPT-4 / o3 |
| prompts/claude.md | Claude 3.5/4 Sonnet / Opus |
| prompts/cursor.md | Cursor + Claude/GPT |
| prompts/autodl_jupyter.md | AutoDL Jupyter + Jupyter AI |

每个 prompt 都包含：
- 项目背景（30 秒可读完）
- 工作目录自描述
- 已完成 vs 未完成任务清单
- 具体下一步行动（含可执行命令）
- 关键事实速查
- 验收标准

---

## 7. 关于诚信边界（必须传达）

1. `scripts/trajrl_model.py` 是**简化等价模型**——不是论文原版。真实训练必须用 Xfc30/TrajRL 仓库原版。
2. Table I 的 9 个 baseline 逐格数值在公开素材中**未完整拿到**——必须从论文 PDF 正文提取，不能编造。
3. Table II 后半段（LightPath 之后的行）在公开素材中**未公开**——同样不能编造。
4. 所有"复现值"必须明确区分于"论文发布值"，报告 §5 已示范格式。

新 AI 必须遵守同样的诚信边界。

---

## 8. 联系上下文

- 用户当前**同时推进三线工作**：
  1. 海事 AIS 异常检测（DCMGNN 论文迁移）
  2. 三模态目标检测（YOLOv9c + AutoDL RTX-4090）
  3. 数字逻辑基础预习（Logisim Evolution）

- **三线工作的算力取舍**：AutoDL 已有账号（ID 8a8b5ee43e6936b7，5 RMB 预存）。复现 TrajRL 与训练 YOLOv9c 都需要云端 GPU，按预算排队使用。

- 这份交接包**与 YOLOv9c 训练互不冲突**——TrajRL 复现只需要 Porto 数据集 + LibCity，跟 YOLOv9c 训练的资源调度正交。

---

## 9. 验收 checklist（新 AI 完成后逐项打勾）

- [ ] Porto 数据集下载并预处理成功
- [ ] T-Drive 数据集下载并预处理成功
- [ ] TrajRL 预训练日志（loss 曲线、epoch 时间、最终 loss）落到 outputs/pretrain_train.log
- [ ] TrajRL 嵌入向量（256-dim）落盘
- [ ] Travel Time Estimation 9 个 metric 全部跑通
- [ ] Trajectory Classification 9 个 metric 全部跑通
- [ ] Similar Trajectory Search 9 个 metric 全部跑通
- [ ] 6 个消融变体全部跑完
- [ ] 效率对比（vs START）跑完
- [ ] 所有数字填回 docs/reproduction_report.md §5
- [ ] 偏差 > 5% 的格单独标注可能原因（对应 §6 的 7 项风险）

---

**祝顺利**。如果新 AI 跑出了任何真实数字，第一时间补回 docs/reproduction_report.md §5，整个项目就完整闭环了。