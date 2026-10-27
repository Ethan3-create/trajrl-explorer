# TrajRL 结果向演示 Demo（Streamlit）

基于 Porto 出租车轨迹抽样的交互演示，一期实现两个"结果向"功能：

1. **行程时间预测（TTE）**：按轨迹距离用线性基线预测行程时长，并展示全样本"距离 vs 时长"分布与拟合基线。
2. **相似路径检索**：从 1 万条轨迹中用 48 维特征向量做 KNN 检索，地图叠加展示查询轨迹与最相似的 Top-K 路径。

## 快速开始

```bash
# 1. 安装依赖（Python 3.13 已验证；streamlit/pandas/plotly/scikit-learn）
python -m pip install streamlit pandas plotly scikit-learn

# 2. 启动（先确认 sample.jsonl 已生成于 ../outputs/porto_sample1w/）
cd scripts
streamlit run demo_app.py
```

浏览器会自动打开 `http://localhost:8501`（或在终端提示的地址打开）。

## 文件说明

| 文件 | 作用 |
|------|------|
| `scripts/demo_app.py` | Streamlit 主程序（两页 + 预留扩展位） |
| `scripts/demo_data.py` | 数据准备层：加载 sample.jsonl、特征化、KNN 索引、TTE 基线拟合 |
| `outputs/porto_sample1w/sample.jsonl` | 数据源：Porto 抽样 10,000 条轨迹（预处理产物） |
| `outputs/porto_sample1w/summary.json` | 数据源统计与口径说明 |

## 操作说明

- 左侧 `radio` 切换功能页；移动滑块"抽样种子"即可换一条轨迹查看预测/检索。
- TTE 页：右侧卡片给出**距离 → 模型预测 vs 推断时长**对比，下方为全样本散点与基线；底部支持自定义输入距离直接预测。
- 检索页：地图深色粗线为查询轨迹，彩色细线为 Top-K 相似轨迹；表格列出每条结果的平均点偏移（越小越相似）、距离、推断时长与点数。

## 口径与已知限制（重要）

- **时长标签为推断近似值**：Porto 原始 CSV 只有起点时间戳、无行程时长真值（论文的 TTE 标签由路网匹配得到，本机外网 502 拿不到路网产物）。故本 demo 的"时长"一律为预处理按 **15s 均匀采样推断**的近似值，界面已明确标注；**任何数字都不代表论文指标，不承诺 SOTA 精度**。
- **TTE 基线**：仅 `时长 = a × 距离 + b`（最小二乘拟合，R²≈0.57，MAE≈235s），用于演示"路径→时间"的交互逻辑，非论文模型。
- **检索特征**：轨迹等距重采样 24 点（48 维平面米制坐标），欧氏距离 KNN；偏好"空间位置接近"的路径，"形状相似但位置不同"的路径不会命中。
- **效率**：索引在应用首次加载时构建（约 1–2 秒），滑动种子即时响应；1 万条规模交互流畅。
- 本机无 GPU、无外网：demo 完全不依赖网络（Plotly 离线路网图），可在本机直接运行。

## 后续扩展位（架构已预留）

- `demo_app.py` 的 `PAGES` 字典注册页面函数，新增"原理向"展示（双视图对比增强、多尺度时空特征可视化、Embedding 投影）时，新增一个 `render_xxx()` 并注册即可。
- 若后续获得官方 TrajRL checkpoint 或路网匹配产物，可将 TTE 基线/检索特征替换为真实模型输出，展示效果与精度可升级。