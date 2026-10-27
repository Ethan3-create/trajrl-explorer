# -*- coding: utf-8 -*-
"""
demo_data.py - TrajRL 结果向 Demo 数据准备层

职责:
  * 加载 Porto 抽样预处理产物 sample.jsonl (10000 条真实轨迹)
  * 将变长轨迹特征化为固定长度向量 (等距重采样到 N_SIG 个点)
  * 构建相似轨迹 KNN 检索索引
  * 拟合 TTE 简单基线 (距离 -> 推断时长 的线性回归)
  * 提供轨迹绘图用的经纬度序列 (超长轨迹降采样, 保证渲染流畅)

口径说明:
  * elapsed_inferred 为 15s 均匀推断时长 (Porto 原始数据无真值),
    本层只使用并标注, 不宣称其为论文口径。
  * 距离用等距圆柱投影近似 (lat0=41.15, Porto 中心), 检索在平面米制坐标上做。

独立于 streamlit, 可被单测 / 命令行直接调用:
    python demo_data.py    # 自检: 加载 -> 特征化 -> 建索引 -> 示例检索
"""

import json
import os
import random
import sys

import numpy as np

# 默认数据路径 (workdir/outputs/porto_sample1w/sample.jsonl)
SAMPLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "porto_sample1w", "sample.jsonl",
)

N_SIG = 24          # 检索用重采样点数 (24 点 -> 48 维向量)
N_DISPLAY = 256     # 渲染用最大点数
LAT0 = 41.15        # Porto 中心纬度, 用于等距圆柱投影


# ---------------------------------------------------------------- loading
def load_records(path=SAMPLE_PATH):
    """加载 sample.jsonl -> list[dict] (原样返回预处理产物记录)."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------- geometry
def to_plane(points, lat0=LAT0):
    """lon/lat 点列 -> 平面米制坐标 (等距圆柱近似, x=东, y=北)."""
    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0] * 111320.0 * np.cos(np.radians(lat0))
    y = pts[:, 1] * 110540.0
    return np.column_stack([x, y])


def resample(points_plane, n):
    """在平面坐标上按累计距离等距重采样到 n 个点 (n>=2).

    实现: 先压缩连续重复点 (距离为 0 的段), 再按累计距离线性插值,
    避免零长度区间导致的除零/NaN。
    """
    pts = np.asarray(points_plane, dtype=np.float64)
    m = len(pts)
    if m <= 2:
        out = np.empty((max(n, m), 2))
        reps = max(n, m) // max(m, 1)
        out[: reps * m] = np.tile(pts, (reps, 1))
        if m * reps < len(out):
            out[m * reps :] = pts[-1]
        return out[:n]
    # 压缩连续重复点
    keep = np.ones(m, dtype=bool)
    keep[1:] = ~(np.all(np.abs(np.diff(pts, axis=0)) < 1e-9, axis=1))
    if not keep[-1]:  # 始终保留终点
        keep[-1] = True
    pts = pts[keep]
    m = len(pts)
    if m < 2:
        return np.tile(pts[0], (n, 1))
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-9:  # 退化为一个点
        return np.tile(pts[0], (n, 1))
    targets = np.linspace(0.0, total, n)
    # 每个 target 所在区间的右端点索引 (searchsorted 保证 cum 单调不减)
    j = np.searchsorted(cum, targets, side="right")
    j = np.clip(j, 1, m - 1)
    left = cum[j - 1]
    right = cum[j]
    w = np.zeros(n)
    ok = (right - left) > 1e-12
    w[ok] = (targets[ok] - left[ok]) / (right[ok] - left[ok])
    out = pts[j - 1] + w[:, None] * (pts[j] - pts[j - 1])
    return out


def feature_vector(record, n=N_SIG):
    """单一轨迹 -> 检索特征向量 (重采样点平面坐标展平, 48 维).

    保留绝对位置信息 (同一路段 -> 空间邻近), 这是"相同路径"检索的依据。
    """
    plane = to_plane(record["points"])
    sig = resample(plane, n)
    return sig.reshape(-1)


def display_points(record, n=N_DISPLAY):
    """轨迹 -> 渲染用 (lon, lat) 列表, 超长轨迹重采样以保证交互流畅."""
    pts = record["points"]
    if len(pts) <= n:
        return pts
    plane = to_plane(pts)
    sig = resample(plane, n)
    # 平面坐标 -> lon/lat (逆投影)
    lon = sig[:, 0] / (111320.0 * np.cos(np.radians(LAT0)))
    lat = sig[:, 1] / 110540.0
    return [[float(a), float(b)] for a, b in zip(lon, lat)]


# ---------------------------------------------------------------- corpus
class PortoCorpus:
    """加载 + 特征矩阵 + KNN 索引 + TTE 基线, 一次构建多次复用."""

    def __init__(self, records, n_sig=N_SIG):
        from sklearn.neighbors import NearestNeighbors

        self.records = records
        self.n_sig = n_sig
        self.ids = [r["trip_id"] for r in records]
        self.idx_of = {tid: i for i, tid in enumerate(self.ids)}

        # 特征矩阵
        feats = np.vstack([feature_vector(r, n_sig) for r in records])
        self.features = feats
        # 元信息数组 (展示用)
        self.cum_dist = np.array([r["cum_dist"] for r in records])
        self.elapsed = np.array([r["elapsed_inferred"] for r in records], dtype=np.float64)
        self.n_points = np.array([r["n_points"] for r in records])

        # KNN 索引
        self.knn = NearestNeighbors(n_neighbors=min(21, len(records)), metric="euclidean")
        self.knn.fit(feats)

        # TTE 基线: elapsed ~ a * cum_dist + b (最小二乘)
        X = self.cum_dist
        y = self.elapsed
        self.a, self.b = np.polyfit(X, y, 1)
        pred = self.a * X + self.b
        mae = float(np.mean(np.abs(pred - y)))
        frac = np.maximum(np.abs(y), 1e-9)
        mape = float(np.mean(np.abs(pred - y) / frac) * 100.0)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        self.tte_stats = {"mae_s": mae, "mape_pct": mape, "r2": r2,
                          "a": float(self.a), "b": float(self.b)}

    # ---- TTE
    def predict_tte(self, cum_dist_m):
        """由行程距离(米)预测推断时长(秒). 返回 (pred_seconds, pred_minutes)."""
        pred = float(self.a * cum_dist_m + self.b)
        return max(pred, 0.0), max(pred, 0.0) / 60.0

    # ---- retrieval
    def search(self, query_record, k=5):
        """给定查询轨迹, 返回 k 条最相似记录 (排除自身) 的 (记录, 距离) 列表."""
        q = feature_vector(query_record, self.n_sig).reshape(1, -1)
        dists, nn_idx = self.knn.kneighbors(q)
        out = []
        q_id = query_record["trip_id"]
        for d, i in zip(dists[0], nn_idx[0]):
            if self.ids[i] == q_id:
                continue
            out.append((self.records[i], float(d)))
            if len(out) >= k:
                break
        return out




# ---------------------------------------------------------------- principle functions

def augment_shift(record, shift_ratio=0.15, seed=None):
    """数据增强: 路段偏移 + 时间扰动. 返回增强后的轨迹记录副本.

    论文 III-A 做法: 随机替换部分路段为其 1-hop 或 2-hop 邻居;
    无真实路网时退化为"将部分 GPS 点做微小随机偏移"来模拟."""
    rng = random.Random(seed) if seed is not None else random.Random()
    pts = np.asarray(record["points"], dtype=np.float64)
    m = len(pts)
    if m <= 3:
        return record
    n_shift = max(1, int(m * shift_ratio))
    # 随机选一个连续子段做微小偏移
    start = rng.randint(0, m - n_shift - 1)
    shifted = pts.copy()
    # 偏移量: 约 50~200 米 (0.0005~0.002 度)
    noise = rng.gauss(0.0, 0.001)
    shifted[start : start + n_shift, 0] += noise
    shifted[start : start + n_shift, 1] += rng.gauss(0.0, 0.001)
    # 时间扰动: 整体时间偏移 -5% ~ +5%
    elapsed = int(record["elapsed_inferred"] * (1.0 + rng.uniform(-0.05, 0.05)))
    return {
        **record,
        "points": shifted.tolist(),
        "elapsed_inferred": max(elapsed, 15),
        "_shift_start": start,
        "_shift_end": start + n_shift,
    }


def patch_partition(points, omega=4):
    """将轨迹切分为 patch (每 patch ω 个 GPS 点).

    论文: 路网匹配后按路段切分 (每 patch ω 个 road segments);
    无路网时退化为"每 ω 个 GPS 点为一个 patch"."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    patches = []
    for i in range(0, n, omega):
        patch = pts[i : i + omega]
        if len(patch) > 0:
            patches.append(patch.tolist())
    return patches


def patch_centroids(points, omega=4):
    """计算每个 patch 的质心 (均值 lon/lat), 用于可视化."""
    patches = patch_partition(points, omega)
    return [(np.mean(np.asarray(p)[:, 0]), np.mean(np.asarray(p)[:, 1])) for p in patches]


def project_embedding(features, method="pca", n_components=2):
    """将特征矩阵降维到 2D. method: 'pca' | 'tsne'."""
    if method == "pca":
        from sklearn.decomposition import PCA
        proj = PCA(n_components=n_components).fit_transform(features)
    else:
        from sklearn.manifold import TSNE
        proj = TSNE(n_components=n_components, random_state=42, perplexity=30).fit_transform(features)
    return proj


def main():
    path = SAMPLE_PATH if os.path.exists(SAMPLE_PATH) else sys.argv[1]
    print("[demo_data] loading:", path)
    recs = load_records(path)
    print("  records:", len(recs))
    corp = PortoCorpus(recs)
    print("  feature dim:", corp.features.shape)
    print("  TTE baseline: elapsed = %.6f * dist + %.1f (sec)" % (corp.a, corp.b))
    print("  TTE fit: MAE=%.1f s  MAPE=%.2f%%  R2=%.3f" % (
        corp.tte_stats["mae_s"], corp.tte_stats["mape_pct"], corp.tte_stats["r2"]))

    # 示例检索: 取 3 条轨迹各查 top-3
    import random
    rng = random.Random(1)
    for _ in range(3):
        q = rng.choice(recs)
        hits = corp.search(q, k=3)
        print("  query %s (%.2f km, %ds)" % (
            q["trip_id"], q["cum_dist"] / 1000.0, q["elapsed_inferred"]))
        for rec, d in hits:
            print("     -> %s  dist=%.1f m  %.2f km  %ds" % (
                rec["trip_id"], d, rec["cum_dist"] / 1000.0, rec["elapsed_inferred"]))
    print("[demo_data] self-check OK")


if __name__ == "__main__":
    main()