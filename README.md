# TrajRL Reproduction + Interactive Demo

> 端到端复现论文 A（TrajRL, ICDE 2025）的工程包，包含**可交互的 Web 演示系统**。

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**🌐 在线体验**: [http://your-ecs-ip:8601](http://your-ecs-ip:8601) *(部署后替换)*

---

## 📖 项目简介

本项目包含两个核心交付物：

1. **论文复现工程** —— 从论文和官方仓库提取事实，整理消融实验配置，支持在 AutoDL RTX-4090 上 6 小时内跑通 Table I/II、消融、效率四个实验。
2. **交互式 Web Demo** —— 基于 10,000 条 Porto 真实出租车轨迹，在浏览器中可视化展示论文核心方法论。

### Web Demo 五大功能

| 页面 | 论文章节 | 功能说明 |
|---|---|---|
| 🔍 **轨迹检索** | §IV-D, §V-C3 | KNN 相似轨迹搜索，验证多尺度编码器表示空间质量 |
| ⏱️ **出行时间估计** | §IV-C, §V-C2 | 对比"距离基线"与"路径特征回归"的时间预测精度 |
| 🔄 **双视图增强** | §IV-A, §IV-E2 | 路段偏移 + 时间扰动的数据增强可视化 |
| 📊 **多尺度编码** | §IV-D | Patch 分块 + InPatch GRU + CrossPatch Transformer 层次可视化 |
| 🔮 **Embedding空间** | §V-D | PCA 降维展示轨迹在表示空间的分布与聚类效果 |

**Demo 技术栈**: Python `http.server` / Gunicorn + 原生 HTML5/CSS3/ES6 + Leaflet.js + ECharts.js + NumPy + scikit-learn

---

## 🚀 快速开始

### 方式一：本地运行（推荐用于开发）

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/trajrl-explorer.git
cd trajrl-explorer

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动服务
python web_server.py --port 8601

# 4. 浏览器访问 http://localhost:8601
```

### 方式二：阿里云 ECS 部署（公网访问）

见 [deploy/DEPLOY.md](deploy/DEPLOY.md) 获取完整的云服务器部署指南。

**一键部署**:
```bash
cd deploy
bash install.sh
```

---

## 📁 文件结构

```
trajrl-explorer/
├── web_server.py              # 后端 API 服务（开发/生产两用）
├── requirements.txt           # Python 依赖
├── web_demo/                  # 前端应用
│   ├── index.html            # 主页面（6 大功能 SPA）
│   ├── css/
│   │   └── style.css         # 暗色专业主题
│   └── js/
│       └── app.js            # 前端逻辑（650 行）
├── outputs/
│   └── porto_sample1w/
│       └── sample.jsonl      # 10,000 条 Porto 轨迹（≈21MB）
├── deploy/                    # 部署配置
│   ├── install.sh            # 阿里云一键安装脚本
│   ├── nginx.conf            # Nginx 反向代理配置
│   ├── trajrl.service        # systemd 守护服务
│   └── DEPLOY.md             # 完整部署指南
├── docs/                      # 论文复现文档
│   ├── reproduction_report.md      # 主复现报告（必读）
│   ├── paper_extracted_facts.md    # 论文 + 官方仓库事实
│   └── ablation_plan.md            # 6 个消融变体清单
├── scripts/                   # 论文复现脚本
│   ├── trajrl_model.py       # 简化版 TrajRL 架构镜像
│   ├── smoke_test.py         # 模型 forward 验证
│   ├── geom_smoke.py         # 纯 NumPy 几何验证
│   └── downstream_eval.py    # 9 个下游 metric + offline_demo
├── prompts/                   # AI 交接模板
│   ├── chatgpt.md
│   ├── claude.md
│   ├── cursor.md
│   └── autodl_jupyter.md
├── HANDOFF.md                 # 完整 AI 交接包
└── README.md                  # 本文件
```

---

## 📊 Web Demo API 接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/stats` | GET | 数据集统计信息 |
| `/api/records?page=0&size=20` | GET | 分页获取轨迹列表 |
| `/api/record/{trip_id}` | GET | 获取单条轨迹详情（含 256 点重采样坐标） |
| `/api/search/{trip_id}?k=5` | GET | KNN 相似轨迹检索（48 维特征空间） |
| `/api/augment/{trip_id}?seed=42` | GET | 双视图数据增强（偏移+扰动） |
| `/api/patches/{trip_id}?omega=4` | GET | Patch 分块可视化 |
| `/api/embeddings/projection` | GET | PCA 降维投影（10,000 条轨迹） |
| `/api/embedding/{trip_id}/neighbors` | GET | Embedding 空间最近邻 |
| `/api/trajectory-tte/{trip_id}` | GET | 单轨迹 TTE 分析（距离基线 vs 特征回归 vs 真实值） |

---

## 🔬 论文复现（AutoDL）

配套 **复现报告**: `docs/reproduction_report.md`

### Quick start (offline demo, no data needed)

```bash
cd scripts
python geom_smoke.py        # numpy-only
python smoke_test.py        # torch needed
python downstream_eval.py --offline_demo
```

### 为什么不直接出"复现数字"？

1. 本机 Python 3.13 ≠ 论文 Python 3.9
2. 本机没有独立 GPU
3. 本机外网 502，git clone / curl 都被拒绝——拿不到 Porto 数据集和原版 TrajRL 仓库
4. 已把所有能从论文与官方仓库挖到的事实搬到 `docs/paper_extracted_facts.md`
5. 报告 §7 提供 AutoDL 上"一键补完"的运行指南——**6 小时之内**能在 RTX-4090 上把 Table I、Table II、消融、效率四个实验**真跑出来**

---

## 📦 部署架构

```
用户浏览器
    ↓ HTTP/HTTPS
阿里云 ECS (Ubuntu 22.04)
    ├── Nginx (80/443) → 反向代理到 8601
    ├── Gunicorn (127.0.0.1:8601) → 4 worker 进程
    │   └── web_server.py (Python API + 静态文件)
    └── outputs/porto_sample1w/sample.jsonl (数据)
```

**安全建议**:
- 生产环境务必配置防火墙（仅开放 80/443）
- 使用阿里云安全组限制来源 IP
- 域名 + HTTPS 需备案（中国大陆合规）

---

## 📝 诚信边界

- `scripts/trajrl_model.py` 是**简化等价模型**：Patch 多尺度 + GRU + Transformer 的几何骨架完全保留；GAT / 时间周期编码 / 时间间隔编码用占位实现替代。
- 真实训练请使用官方仓库 `Xfc30/TrajRL/main` 里的 `libcity/model/trajectory_embedding/trajrl.py`。
- 论文 Table II 部分 baseline 数据在原论文中未公开，报告里标"原论文未公开"，没有编造。

---

## 📄 许可证

MIT License

---

## 🙏 致谢

- Porto 出租车轨迹数据集
- TrajRL 论文作者团队（ICDE 2025）
