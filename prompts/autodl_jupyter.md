# AutoDL Jupyter 启动 Prompt

## 场景

你在 AutoDL 的 Jupyter Notebook 环境（推荐 RTX-4090 24G 实例）。
整个 `trajrl_paper_repro_workdir/` 已上传到 Jupyter 工作目录（一般是 `/root/trajrl_paper_repro_workdir/`）。

## 第一步：环境自检（运行这个 cell）

```python
import os, subprocess
print("CWD:", os.getcwd())
print("Python:", subprocess.check_output(["python", "--version"]).decode().strip())
print("CUDA available:", subprocess.check_output(["python", "-c", "import torch; print(torch.cuda.is_available())"]).decode().strip())
print("Files:", os.listdir("trajrl_paper_repro_workdir")[:10])
```

## 第二步：通读 3 个核心文档（用 Jupyter Markdown cell 渲染）

```python
from IPython.display import Markdown, display
display(Markdown(open("trajrl_paper_repro_workdir/HANDOFF.md").read()))
display(Markdown(open("trajrl_paper_repro_workdir/docs/reproduction_report.md").read()))
```

## 第三步：跑离线 smoke（验证环境 + 模型骨架）

```python
%cd trajrl_paper_repro_workdir
!python scripts/geom_smoke.py        # numpy-only
!python scripts/smoke_test.py        # torch
!python scripts/downstream_eval.py --offline_demo
```

全部 PASS 后，进入下一步。

## 第四步：拉真实仓库 + 装依赖

```python
!git clone https://github.com/Xfc30/TrajRL.git trajrl_repo
!pip install -r trajrl_repo/requirements.txt
!pip install -r trajrl_paper_repro_workdir/configs/requirements_check.txt
```

## 第五步：下载数据集

```python
# Porto 数据集（~1GB）
!mkdir -p data && cd data
!kaggle datasets download -d cabaki/porto-taxi-trajectory-data -p porto --unzip

# T-Drive 数据集（从 MS Research 拿，curl）
!wget https://download.microsoft.com/download/...tdrive... -O tdrive.zip
!unzip tdrive.zip
```

## 第六步：预处理数据（按 LibCity 标准）

```python
%cd trajrl_repo
!python scripts/data_preprocess/porto.py
!python scripts/data_preprocess/tdrive.py
```

## 第七步：跑预训练

```python
%cd trajrl_repo/TrajRL
!python run_model.py --model TrajRLContrastiveLM --dataset porto \
    --gpu_id 0 --split true \
    --masking_ratio 0.2 --distribution geometric \
    --out_data_argument1 trun --out_data_argument2 shift \
    2>&1 | tee /root/trajrl_paper_repro_workdir/outputs/pretrain_train.log
```

## 第八步：下游任务

```python
# TTE / Classification / SimSearch 三个 executor
!python run_model.py --task tte --dataset porto --model TrajRL --pretrain_path outputs/porto_pretrained.pkl
!python run_model.py --task classification --dataset porto --model TrajRL --pretrain_path outputs/porto_pretrained.pkl
!python run_model.py --task similar_search --dataset porto --model TrajRL --pretrain_path outputs/porto_pretrained.pkl
```

## 第九步：消融

```python
# 6 个变体，每个改 porto.json 的某个开关位
for variant in ["wo_MFTime", "wo_TimeInterval", "wo_MSTraj", "wo_InPatch", "wo_TrajRec", "wo_CL"]:
    !python run_model.py --model TrajRLContrastiveLM_{variant} --dataset porto ...
```

## 第十步：填回报告

把 outputs/*.csv 的数字复制到 `docs/reproduction_report.md §5`，**与论文发布数字分列**。

---

## 验收 checklist

- [ ] Porto 数据集下载并预处理成功
- [ ] T-Drive 数据集下载并预处理成功
- [ ] TrajRL 预训练日志（loss 曲线、epoch 时间、最终 loss）落到 outputs/pretrain_train.log
- [ ] 嵌入向量（256-dim）落盘
- [ ] 3 个下游任务 × 9 个 metric 全部跑通
- [ ] 6 个消融变体全部跑完
- [ ] 效率对比（vs START）跑完
- [ ] 所有数字填回 docs/reproduction_report.md §5
- [ ] 偏差 > 5% 的格单独标注可能原因

## 关键约束

- 不编造 Table I/II 未公开的数字
- 复现数字 vs 论文数字分列标注
- 真实训练必须用 Xfc30/TrajRL 原仓库
- 简化版模型（scripts/trajrl_model.py）只用于 smoke test

## 如果遇到问题

- **数据集下载失败**：检查 Kaggle token（`~/.kaggle/kaggle.json`）
- **GPU OOM**：batch_size 从 porto.json 默认值往下调
- **LibCity 版本冲突**：用 `pip install libcity==1.3.0`（Xfc30/TrajRL 文档可能指明版本）
- **Table I/II 数字找不到**：公开素材只给了 TrajRL 那行的 9 个数字；baseline 9 行的数字必须从 IEEE Xplore 订阅或图书馆拿原文 PDF