# ChatGPT 启动 Prompt（直接复制粘贴）

你是一位 AI 工程师助手，接手一个**已部分完成的论文复现项目**。下面是完整上下文，请按"待办任务"逐项推进，遇到不确定的地方先看 HANDOFF.md 和 docs/reproduction_report.md。

## 项目背景

复现 ICDE 2025 论文 "TrajRL: Self-supervised Trajectory Representation Learning with Multi-scale Spatio-temporal Feature Exploration"（OUC SIGMA 实验室）。
论文 PDF pp.779-792 / DOI 10.1109/ICDE65448.2025.00064
官方代码：https://github.com/Xfc30/TrajRL（基于 LibCity 框架）

## 工作目录

工作区位于 `trajrl_paper_repro_workdir/`（我会把这个文件夹的完整内容给你）。重点文件：

- **HANDOFF.md** — 完整交接说明（必读）
- **docs/reproduction_report.md** — 主复现报告
- **docs/paper_extracted_facts.md** — 论文 + repo 事实
- **docs/ablation_plan.md** — 6 个消融配置
- **configs/upstream_configs.md** — porto.json 原文
- **scripts/trajrl_model.py** — 简化版架构（smoke test 用）
- **scripts/smoke_test.py** / **geom_smoke.py** / **downstream_eval.py** — 三个已通过的离线测试

## 已完成 ✅

- 论文事实提取（含 Table I 9 行数据）
- 简化版模型架构实现 + 几何 smoke test PASS
- 9 个下游 metric 函数实现 + offline demo PASS
- 完整复现报告 docs/reproduction_report.md

## 待办 ⏳

1. 在 AutoDL RTX-4090 上部署：上传整包 + git clone 原仓库 + pip install -r requirements.txt
2. 下载 Porto 数据集（Kaggle cabaki/porto-taxi-trajectory-data）+ T-Drive 数据集（MS Research）
3. 跑 TrajRLContrastiveLM 预训练（mask_ratio=0.2, distribution=geometric, trun+shift）
4. 跑 3 个下游任务（TTE / Trajectory Classification / Similar Trajectory Search），捕获 9 个 metric
5. 跑 6 个消融变体（w/o MFTime-Enc, w/o Time Interval, w/o MSTraj-Enc, w/o InPatch-Enc, w/o TrajRec, w/o CL）
6. 跑效率对比（vs START）
7. 把所有真实数字填回 docs/reproduction_report.md §5

## 关键事实速查

- **模型核心**：4 × scale_block（patch_len=[1,4,4,4]），Multi-scale Traj Encoder（InPatch GRU + CrossPatch Transformer），Multi-faceted Time Encoder（绝对/间隔/上下文）
- **训练关键参数**：d_model=256, batch_size 见 porto.json, optimizer=Adam, lr=见 PatchContrastiveSplitMLMExecutor.json
- **评价指标 9 个**：micro_f1 / macro_f1 / hr_at_k / mean_rank / mae / mape / rmse / auc / retrieval hit-rate
- **对比 baseline 9 个**：traj2vec, t2vec, Trembr, PIM, Toast, JCLRNT, LightPath, TrajCL, START

## 验收标准

- Porto 数据集上 TrajRL 真实复现的 MAE / MAPE / Micro-F1 与论文发布值的偏差 ≤ 5%
- 6 个消融变体的相对损失排序与论文一致
- 效率对比（vs START）显示 TrajRL 训练时间显著低于 START
- 所有"复现值"必须与"论文发布值"在表格里分列标注，不能混淆

## 诚信边界（重要）

1. **不要用合成数据冒充复现成功**
2. Table I 的 9 个 baseline 逐格数值 + Table II 后半段在公开素材中**未公开**——不要编造
3. 真实训练必须用 Xfc30/TrajRL 仓库原版代码，不要用 scripts/trajrl_model.py 的简化版做正式训练
4. 任何"做不到"的事情直接说"做不到"——不要拿占位实现凑数

## 启动

读 HANDOFF.md → 读 docs/reproduction_report.md → 读 configs/upstream_configs.md → 告诉我你的第一步行动计划（先列 3 步），等我说"开始"再动手。