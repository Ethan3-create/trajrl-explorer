# TrajRL Explorer - Web Demo 系统

## 系统概述

TrajRL Explorer 是一个基于 TrajRL 论文（ICDE 2023）的专业级轨迹表示学习演示系统。

### 与论文的对应关系

| Demo 页面 | 论文章节 | 展示内容 |
|---|---|---|
| 🔍 轨迹检索 | §IV-D, §V-C3 | 基于 KNN 的相似轨迹搜索，体现多尺度轨迹编码器学到的表示空间 |
| ⏱️ 出行时间估计 | §IV-C, §V-C2 | 距离-时长关系分析，体现多面时间编码器对时间信息的建模 |
| 🔄 双视图增强 | §IV-A, §IV-E2 | 路段偏移+时间扰动的数据增强，对应 InfoNCE 对比学习的正样本对 |
| 📊 多尺度编码 | §IV-D | Patch 分块 + InPatch GRU + CrossPatch Transformer 的层次可视化 |
| 🔮 Embedding空间 | §V-D | PCA 降维展示轨迹在表示空间的分布，体现表示学习质量 |

## 技术栈

- **后端**: Python + 内置 HTTP 服务器 (http.server)
- **前端**: 原生 HTML5/CSS3/ES6
- **地图**: Leaflet.js + OpenStreetMap（瓦片源 tile.openstreetmap.de 镜像）
- **图表**: ECharts.js
- **数据处理**: NumPy, scikit-learn (PCA, KNN)

## 启动方式

### 前置依赖

```bash
pip install numpy scikit-learn
```

### 启动服务

```bash
# 进入项目目录
cd C:\Users\PRLOUIS\Desktop\项目与代码\trajrl_paper_repro_workdir

# 启动后端服务
python web_server.py --port 8601
```

### 访问系统

打开浏览器访问: http://localhost:8601

## 文件结构

```
trajrl_paper_repro_workdir/
├── web_server.py              # 后端 API 服务
├── web_demo/                  # 前端应用
│   ├── index.html             # 主页面
│   ├── css/
│   │   └── style.css          # 样式表 (暗色专业主题)
│   ├── js/
│   │   └── app.js             # 前端逻辑
│   └── README.md              # 本文件
├── outputs/porto_sample1w/    # 数据目录
│   └── sample.jsonl           # 10,000 条 Porto 轨迹样本
└── scripts/                   # 原有脚本目录
    ├── demo_app.py            # 旧版 Streamlit Demo
    └── demo_data.py           # 数据处理层
```

## API 接口文档

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/stats` | GET | 数据集统计信息 |
| `/api/records?page=0&size=20` | GET | 分页获取轨迹列表 |
| `/api/record/{trip_id}` | GET | 获取单条轨迹详情 |
| `/api/search/{trip_id}?k=5` | GET | KNN 相似轨迹检索 |
| `/api/augment/{trip_id}?seed=42` | GET | 双视图数据增强 |
| `/api/patches/{trip_id}?omega=4` | GET | Patch 分块可视化数据 |
| `/api/embeddings/projection` | GET | PCA 降维投影数据 |
| `/api/embedding/{trip_id}/neighbors?k=5` | GET | 嵌入空间邻近轨迹 |
| `/api/tte/stats` | GET | TTE 基线统计 |
| `/api/tte/predict?dist=5000` | GET | TTE 预测 |
| `/api/trajectory-tte/{trip_id}` | GET | 单条轨迹 TTE 分析：距离基线、特征回归（距离+点数+弯曲度）、真实推断时长及误差 |

## 免责声明

本演示系统使用简化模型与启发式特征（48 维重采样坐标），**不代表论文 SOTA 精度**。出行时间使用推断值（15s 均匀采样），标注为"推断值"。旨在展示 TrajRL 方法论思想，非真实复现。
