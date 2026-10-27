# Claude 启动 Prompt

你接手一个**半完成**的论文复现项目。请按以下节奏工作：

## 1. 你的角色与目标

- 角色：AI 工程师
- 目标：在 AutoDL RTX-4090 上端到端复现 ICDE 2025 论文 TrajRL
- 工作目录：`trajrl_paper_repro_workdir/`
- 必读：`HANDOFF.md`、`docs/reproduction_report.md`、`configs/upstream_configs.md`

## 2. 上下文

论文："Self-supervised Trajectory Representation Learning with Multi-scale Spatio-temporal Feature Exploration"（ICDE 2025，OUC SIGMA 实验室）。
代码：https://github.com/Xfc30/TrajRL（基于 LibCity）。
数据集：Porto + T-Drive。下游任务：TTE / Trajectory Classification / Similar Trajectory Search。
指标：9 个（micro_f1 / macro_f1 / hr_at_k / mean_rank / mae / mape / rmse / auc / retrieval hit-rate）。
Baseline：9 个。消融：6 个变体。

## 3. 已完成

- 论文事实提取 ✅
- 简化版模型 + 3 个 smoke test PASS ✅
- 9 个 metric 函数 + offline demo PASS ✅
- 复现报告骨架 ✅

## 4. 待办（按优先级）

1. **部署**：上传整包到 AutoDL RTX-4090，安装 PyTorch 1.12.1 + Python 3.9 + LibCity
2. **数据**：下载 Porto（Kaggle）+ T-Drive（MS Research），按 LibCity 标准预处理
3. **预训练**：跑 `python run_model.py --model TrajRLContrastiveLM --dataset porto --masking_ratio 0.2 --distribution geometric --out_data_argument1 trun --out_data_argument2 shift`
4. **下游评估**：3 个任务 × 9 个 metric
5. **消融**：6 个变体
6. **效率对比**：vs START
7. **回填**：docs/reproduction_report.md §5

## 5. 关键约束（必须遵守）

- 不编造 Table I/II 中未公开的 baseline 数字
- 真实训练必须用 Xfc30/TrajRL 原仓库，不要用 scripts/trajrl_model.py（那是 smoke test 用）
- 所有复现数字与论文发布数字分列标注
- 任何"做不到"直接说，不要凑数

## 6. 你的第一步

读 HANDOFF.md + docs/reproduction_report.md §1-§7，然后告诉我：**3 步启动计划**（每步含具体命令 + 预期产物）。等我说"开始"再执行。

如果你对任何任务边界不清楚（特别是数据集下载、LibCity 集成、Table II 后半段数据），请先在 §7 提问，不要自己猜。

## 7. 需要你主动澄清的问题（如果你看到时已经有答案，跳过）

- Porto 数据集的标准版本是 LibCity 提供的还是用户自己从 Kaggle 拿？
- 9 个 baseline 在哪里找实现？LibCity 是否已经集成？
- Table II 后半段是否有机会通过 IEEE Xplore / 图书馆订阅拿论文 PDF？
- AutoDL 5 RMB 预算够不够跑完整流程？

请开始。