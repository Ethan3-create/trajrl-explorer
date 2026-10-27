# Cursor 启动 Prompt（IDE 编辑器场景）

## 上下文（5 秒读完）

我在做一个论文复现项目。完整背景写在 `trajrl_paper_repro_workdir/HANDOFF.md`，主报告在 `docs/reproduction_report.md`。

**项目**：ICDE 2025 TrajRL 论文（OUC SIGMA）端到端复现
**官方代码**：https://github.com/Xfc30/TrajRL
**数据集**：Porto + T-Drive
**下游任务**：TTE / Trajectory Classification / Similar Trajectory Search（9 个 metric）
**Baseline × 9 + 消融 × 6 + 效率对比 × 1**

## 当前状态

✅ 论文事实提取、简化版模型架构、3 个 smoke test 通过、9 个 metric 函数 PASS、复现报告骨架
❌ 真实 Porto/T-Drive 训练、Table I/II 全部数字、6 个消融、效率对比

## 待办（按顺序）

1. **打开仓库**：执行 `code trajrl_paper_repro_workdir`（Cursor 命令面板：Cmd+Shift+P → "Code: Open Folder"）
2. **通读 3 个文档**（Cursor 中可点击跳转）：
   - `HANDOFF.md` — 完整交接说明
   - `docs/reproduction_report.md` — 主复现报告
   - `configs/upstream_configs.md` — porto.json 原文
3. **先跑离线 smoke**（验证环境正确）：
   - `python scripts/geom_smoke.py`
   - `python scripts/smoke_test.py`
   - `python scripts/downstream_eval.py --offline_demo`
4. **规划 AutoDL 部署路径**（用 Cursor Agent 帮你生成部署脚本）：
   - `scripts/deploy_to_autodl.sh`（创建）
   - `scripts/download_porto.sh`（创建）
   - `scripts/download_tdrive.sh`（创建）
   - `scripts/run_pretrain.sh`（创建）
   - `scripts/run_downstream.sh`（创建）
   - `scripts/run_ablation.sh`（创建）
5. **逐个生成脚本 → 我审阅 → 推到 AutoDL 跑**

## Cursor 内的快捷指令

- **Cmd+L** 选中代码问 AI："解释这段为什么用 geometric mask distribution 而不是 uniform"
- **Cmd+I** 写新代码时："基于 scripts/trajrl_model.py 的 NodeEmbedding 写出真实 GAT 实现"
- **Cmd+K** 跨文件搜索："找出所有 paper_extracted_facts.md 提到的消融变体"
- **Agent 模式**：让它帮你写 `scripts/deploy_to_autodl.sh`，明确告诉它"使用 bash 语法，AutoDL RTX-4090 + PyTorch 1.12.1"

## 关键约束

- 不编造 Table I/II 未公开的数字
- 真实训练用 Xfc30/TrajRL 原仓库
- 复现数字 vs 论文数字分列标注
- 简化版模型（`scripts/trajrl_model.py`）只用于 smoke test

## 第一步

读完 3 个必读文档后，告诉我：
1. 你理解了什么
2. 你的 5 步启动计划（含具体命令）
3. 哪些任务你认为不需要真跑（比如 Table II 后半段无原文 PDF）

等你确认后，再开始动手写部署脚本。