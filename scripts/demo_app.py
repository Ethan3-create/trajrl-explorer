# -*- coding: utf-8 -*-
"""
demo_app.py v4 - TrajAgg 风格三栏 + 原理向三页 (Streamlit)

界面结构:
  结果向检索页: 左栏查询列表 / 中栏地图 / 右栏 TTE+检索详情
  双视图增强页: A/B 对比原始轨迹与增强后轨迹
  多尺度时空编码页: Patch 切分 + InPatch/CrossPatch 层次 + MFTime 时间编码
  Embedding 投影页: PCA 降维散点图 + 高亮查询与相似轨迹

session_state:
  selected_idx, page, k, current_tab
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import random

from demo_data import (
    SAMPLE_PATH,
    PortoCorpus,
    display_points,
    load_records,
    augment_shift,
    patch_partition,
    patch_centroids,
    project_embedding,
)

st.set_page_config(page_title="TrajRL 轨迹探索", page_icon="🚕", layout="wide")

# ---------------------------------------------------------------- session state
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0
if "page" not in st.session_state:
    st.session_state.page = 0
if "k" not in st.session_state:
    st.session_state.k = 5
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "🔍 结果向检索"

# ---------------------------------------------------------------- data
@st.cache_resource
def get_corpus():
    recs = load_records(SAMPLE_PATH)
    return PortoCorpus(recs)


corpus = get_corpus()
records = corpus.records

BG_SEED = 7
random.seed(BG_SEED)
BG_COUNT = 2000
BG_IDX = random.sample(range(len(records)), min(BG_COUNT, len(records)))
BG_TRACES = [display_points(records[i]) for i in BG_IDX]

@st.cache_resource
def get_embedding_proj():
    proj = project_embedding(corpus.features, method="pca", n_components=2)
    return proj


EMB_2D = get_embedding_proj()

# ---------------------------------------------------------------- constants
PAGE_SIZE = 20

# ---------------------------------------------------------------- helpers

def mini_svg(rec, width=130, height=65):
    pts = rec["points"]
    if len(pts) > 50:
        step = len(pts) // 50
        pts = pts[::step]
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    if lat_max - lat_min < 1e-6:
        lat_max += 0.001
        lat_min -= 0.001
    if lon_max - lon_min < 1e-6:
        lon_max += 0.001
        lon_min -= 0.001
    w, h = width - 6, height - 6

    def tx(lon):
        return 3 + (lon - lon_min) / (lon_max - lon_min) * w

    def ty(lat):
        return 3 + (lat_max - lat) / (lat_max - lat_min) * h

    path_d = "M" + " L".join([f"{tx(p[0])},{ty(p[1])}" for p in pts])
    svg = (
        f'<svg width="{width}" height="{height}" '
        'style="background:#f8fafc;border-radius:5px;border:1px solid #e2e8f0;">'
        f'<path d="{path_d}" fill="none" stroke="#2563EB" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" opacity="0.85"/>'
    )
    sx, sy = tx(pts[0][0]), ty(pts[0][1])
    ex, ey = tx(pts[-1][0]), ty(pts[-1][1])
    svg += f'<circle cx="{sx}" cy="{sy}" r="3" fill="#16A34A"/>'
    svg += f'<circle cx="{ex}" cy="{ey}" r="3" fill="#DC2626"/>'
    svg += "</svg>"
    return svg


def build_map_figure(query_rec, hits):
    fig = go.Figure()

    # 伪路网底图
    for pts in BG_TRACES:
        lons = [p[0] for p in pts]
        lats = [p[1] for p in pts]
        fig.add_trace(
            go.Scatter(
                x=lons, y=lats, mode="lines",
                line=dict(color="#B0B0B0", width=0.4),
                opacity=0.08, hoverinfo="skip", showlegend=False,
            )
        )

    # 查询轨迹
    qpts = display_points(query_rec)
    fig.add_trace(
        go.Scatter(
            x=[p[0] for p in qpts], y=[p[1] for p in qpts],
            mode="lines", line=dict(color="#2563EB", width=4),
            name="查询轨迹", opacity=0.9,
            hovertemplate="查询轨迹<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[qpts[0][0], qpts[-1][0]], y=[qpts[0][1], qpts[-1][1]],
            mode="markers",
            marker=dict(size=[10, 10], color=["#16A34A", "#DC2626"], symbol=["circle", "x"]),
            name="起/终点", text=["起点", "终点"],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # Top-K 相似轨迹
    palette = [
        "#0F766E", "#D97706", "#DC2626", "#7C3AED",
        "#0891B2", "#65A30D", "#DB2777", "#4F46E5", "#EA580C", "#2563EB",
    ]
    for i, (hrec, dist) in enumerate(hits):
        hpts = display_points(hrec)
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in hpts], y=[p[1] for p in hpts],
                mode="lines", line=dict(color=color, width=2),
                name=f"Top-{i + 1} {hrec['trip_id'][-6:]}",
                opacity=0.75,
                hovertemplate=f"Top-{i + 1}<br>{hrec['trip_id']}<extra></extra>",
            )
        )

    fig.update_layout(
        height=620, margin=dict(l=5, r=5, t=35, b=5),
        xaxis=dict(range=[-8.70, -8.55], scaleanchor="y"),
        yaxis=dict(range=[41.10, 41.25]),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#e2e8f0", borderwidth=1,
        ),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    return fig


# ================================================================
# 页面 1: 结果向检索
# ================================================================
def render_map_search():
    st.title("🚕 TrajRL 结果向检索")
    st.caption("数据: Porto 出租车轨迹 10,000 条 | 离线演示 | 时长为推断近似值")

    left, center, right = st.columns([1.0, 2.4, 1.2])

    # ---------- LEFT ----------
    with left:
        st.markdown("#### 查询轨迹列表")
        search = st.text_input("🔍 搜索 ID 或点数", "", key="search_input")
        filtered = records
        if search.strip():
            s = search.strip().lower()
            filtered = [r for r in records if s in r["trip_id"].lower() or str(r["n_points"]) == s]
        total_filtered = len(filtered)
        total_pages = max(1, (total_filtered + PAGE_SIZE - 1) // PAGE_SIZE)
        st.session_state.page = min(st.session_state.page, total_pages - 1)

        c_prev, c_page, c_next = st.columns([1, 2.2, 1])
        with c_prev:
            if st.button("◀", disabled=st.session_state.page <= 0, key="btn_prev"):
                st.session_state.page -= 1
                st.rerun()
        with c_page:
            st.caption(f"第 {st.session_state.page + 1}/{total_pages} 页 ({total_filtered} 条)")
        with c_next:
            if st.button("▶", disabled=st.session_state.page >= total_pages - 1, key="btn_next"):
                st.session_state.page += 1
                st.rerun()

        start = st.session_state.page * PAGE_SIZE
        page_recs = filtered[start : start + PAGE_SIZE]
        if not page_recs:
            st.info("无匹配轨迹")
        else:
            for i, rec in enumerate(page_recs):
                abs_idx = records.index(rec)
                is_selected = abs_idx == st.session_state.selected_idx
                with st.container():
                    c_img, c_info = st.columns([0.42, 0.58])
                    with c_img:
                        st.markdown(mini_svg(rec), unsafe_allow_html=True)
                    with c_info:
                        label = "✅ " if is_selected else ""
                        st.markdown(f"**{label}{rec['trip_id']}**")
                        st.caption(f"📏 {rec['cum_dist']/1000:.2f} km · 📍 {rec['n_points']} 点")
                    btn_label = "已选中" if is_selected else "选择"
                    if st.button(btn_label, key=f"sel_{abs_idx}", disabled=is_selected, use_container_width=True):
                        st.session_state.selected_idx = abs_idx
                        st.rerun()
                if i < len(page_recs) - 1:
                    st.markdown("<hr style='margin:6px 0;border:none;border-top:1px solid #f1f5f9;'>", unsafe_allow_html=True)

    # ---------- CENTER ----------
    with center:
        rec = records[st.session_state.selected_idx]
        hits = corpus.search(rec, k=st.session_state.k)
        st.markdown(f"**查询**: `{rec['trip_id']}` · {rec['cum_dist']/1000:.2f} km · {rec['n_points']} 点 · Top-{st.session_state.k} 相似路径")
        fig = build_map_figure(rec, hits)
        st.plotly_chart(fig, use_container_width=True)

    # ---------- RIGHT ----------
    with right:
        st.markdown("#### 行程时间预测 (TTE)")
        pred_s, pred_min = corpus.predict_tte(rec["cum_dist"])
        actual_s = float(rec["elapsed_inferred"])
        actual_min = actual_s / 60.0
        c1, c2 = st.columns(2)
        c1.metric("距离", f"{rec['cum_dist']/1000:.2f} km")
        c2.metric("推断时长", f"{actual_min:.1f} min")
        st.markdown(f"**基线预测**: `{pred_min:.1f} min`")
        st.caption(f"基线: 时长 = {corpus.a:.4f} × 距离 + {corpus.b:.1f} s (R²={corpus.tte_stats['r2']:.2f})")

        st.markdown("---")
        st.markdown("#### 位置驱动推算")
        pos_ratio = st.slider("当前位置 (0%=起点, 100%=终点)", 0.0, 1.0, 0.5, 0.01, key="pos_slider")
        remaining_dist = rec["cum_dist"] * (1.0 - pos_ratio)
        rem_pred_s, rem_pred_min = corpus.predict_tte(remaining_dist)
        st.info(f"距终点剩余 **{remaining_dist/1000:.2f} km** → 预计还需 **{rem_pred_min:.1f} min**")

        st.markdown("---")
        st.markdown(f"#### 相似路径 Top-{st.session_state.k}")
        st.session_state.k = st.slider("返回 K 条", 3, 10, st.session_state.k, 1, key="k_slider")
        palette = ["#0F766E", "#D97706", "#DC2626", "#7C3AED", "#0891B2", "#65A30D", "#DB2777", "#4F46E5", "#EA580C", "#2563EB"]
        for i, (hrec, dist) in enumerate(hits):
            color = palette[i % len(palette)]
            with st.container():
                st.markdown(f"**#{i + 1}** `{hrec['trip_id']}` <span style='color:{color}'>●</span>", unsafe_allow_html=True)
                st.caption(f"偏移 {dist/(corpus.n_sig**0.5):.0f} m · {hrec['cum_dist']/1000:.2f} km · {hrec['elapsed_inferred']/60.0:.1f} min · {hrec['n_points']} 点")
            if i < len(hits) - 1:
                st.markdown("<hr style='margin:6px 0;border:none;border-top:1px solid #f1f5f9;'>", unsafe_allow_html=True)
        st.info("偏移 = 特征向量欧氏距离在 24 个重采样点上的平均空间偏移（米）；越小表示路径在空间上越接近。")
        st.markdown("---")
        st.caption("**标签说明**: Porto 原始数据无行程时长真值，推断时长为预处理按 15s 均匀采样推断的近似值，仅用于演示。")


# ================================================================
# 页面 2: 双视图对比增强
# ================================================================
def render_augmentation():
    st.title("🔀 双视图数据增强对比")
    st.caption("论文 III-A: 同一条轨迹生成两个增强视图，用于对比学习 (InfoNCE)")

    seed = st.slider("抽样种子", 0, 9999, 42, key="aug_seed")
    rec = records[seed % len(records)]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("视图 A: 原始轨迹")
        qpts = display_points(rec)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[p[0] for p in qpts], y=[p[1] for p in qpts],
            mode="lines", line=dict(color="#2563EB", width=3), name="原始"))
        fig.add_trace(go.Scatter(
            x=[qpts[0][0], qpts[-1][0]], y=[qpts[0][1], qpts[-1][1]],
            mode="markers", marker=dict(size=10, color=["#16A34A", "#DC2626"])))
        fig.update_layout(height=400, xaxis=dict(scaleanchor="y", range=[-8.70, -8.55]),
            yaxis=dict(range=[41.10, 41.25]), showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.metric("距离", f"{rec['cum_dist']/1000:.2f} km")
        st.metric("推断时长", f"{rec['elapsed_inferred']/60:.1f} min")
        st.metric("点数", rec["n_points"])

    with col2:
        st.subheader("视图 B: 增强轨迹")
        aug = augment_shift(rec, shift_ratio=0.15, seed=seed)
        apts = display_points(aug)
        shift_s = aug.get("_shift_start", 0)
        shift_e = aug.get("_shift_end", len(apts))
        fig2 = go.Figure()
        # 偏移前
        if shift_s > 0:
            fig2.add_trace(go.Scatter(
                x=[p[0] for p in apts[:shift_s]], y=[p[1] for p in apts[:shift_s]],
                mode="lines", line=dict(color="#2563EB", width=3)))
        # 偏移段（红色高亮）
        if shift_e > shift_s:
            fig2.add_trace(go.Scatter(
                x=[p[0] for p in apts[shift_s:shift_e]], y=[p[1] for p in apts[shift_s:shift_e]],
                mode="lines", line=dict(color="#DC2626", width=4), name="偏移段"))
        # 偏移后
        if shift_e < len(apts):
            fig2.add_trace(go.Scatter(
                x=[p[0] for p in apts[shift_e:]], y=[p[1] for p in apts[shift_e:]],
                mode="lines", line=dict(color="#2563EB", width=3)))
        fig2.add_trace(go.Scatter(
            x=[apts[0][0], apts[-1][0]], y=[apts[0][1], apts[-1][1]],
            mode="markers", marker=dict(size=10, color=["#16A34A", "#DC2626"])))
        fig2.update_layout(height=400, xaxis=dict(scaleanchor="y", range=[-8.70, -8.55]),
            yaxis=dict(range=[41.10, 41.25]), showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

        st.metric("距离", f"{aug['cum_dist']/1000:.2f} km")
        st.metric("推断时长", f"{aug['elapsed_inferred']/60:.1f} min")
        st.metric("点数", aug["n_points"])

    st.markdown("---")
    st.info("""
    **论文做法**: 将部分路段随机替换为其 1-hop/2-hop 路网邻居，并对时间戳做扰动。
    两个增强视图的 embedding 在对比学习中被拉近，迫使模型提取对局部扰动和时间变化不变的通用表示。
    """)


# ================================================================
# 页面 3: 多尺度时空编码
# ================================================================
def render_multiscale():
    st.title("📐 多尺度时空编码可视化")
    st.caption("论文核心创新: InPatch + CrossPatch 多尺度轨迹编码 + MFTime 多面时间编码")

    seed = st.slider("抽样种子", 0, 9999, 42, key="ms_seed")
    rec = records[seed % len(records)]

    # --- ① Patch 切分 ---
    st.subheader("① 轨迹 → Patch 切分 (ω=4)")
    patches = patch_partition(rec["points"], omega=4)
    n_patches = len(patches)

    fig = go.Figure()
    qpts = display_points(rec)
    fig.add_trace(go.Scatter(
        x=[p[0] for p in qpts], y=[p[1] for p in qpts],
        mode="lines", line=dict(color="#E2E8F0", width=2), name="原始轨迹"))
    palette_p = ["#0F766E", "#D97706", "#DC2626", "#7C3AED", "#0891B2", "#65A30D", "#DB2777", "#4F46E5"]
    for i, patch in enumerate(patches):
        col = palette_p[i % len(palette_p)]
        fig.add_trace(go.Scatter(
            x=[p[0] for p in patch], y=[p[1] for p in patch],
            mode="lines+markers", line=dict(color=col, width=3),
            marker=dict(size=5), name=f"Patch {i+1} ({len(patch)}点)"))
    fig.update_layout(height=450, xaxis=dict(scaleanchor="y", range=[-8.70, -8.55]),
        yaxis=dict(range=[41.10, 41.25]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"共 {n_patches} 个 patch, 每 patch ω=4 个 GPS 点")

    # --- ② 层次结构 ---
    st.subheader("② 多尺度轨迹编码器结构")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **InPatch 编码器** (每层内部)
        - 输入: 一个 patch 内的 ω 个路段/点
        - 机制: GRU 学习 patch 内部的先后顺序和依赖关系
        - 输出: patch 级别的聚合表示
        """)
    with col2:
        st.markdown("""
        **CrossPatch 编码器** (每层之间)
        - 输入: 所有 patch 的聚合表示序列
        - 机制: Transformer 自注意力学习 patch 之间的关系
        - 输出: 更新后的 patch 表示（感受野扩大）
        """)

    layers = ["Layer 1", "Layer 2", "Layer 3", "Layer 4"]
    receptive = [4, 16, 64, 256]
    colors_bar = ["#E2E8F0", "#93C5FD", "#3B82F6", "#1D4ED8"]
    fig2 = go.Figure(go.Bar(
        x=layers, y=receptive,
        marker_color=colors_bar,
        text=receptive, textposition="auto"))
    fig2.update_layout(height=300, yaxis_title="感受野 (点/路段数)",
        xaxis_title="编码器层", margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("堆叠 L_T=4 层: 感受野指数级扩大，从 patch 内关系到全局轨迹语义")

    # --- ③ 时间编码 ---
    st.subheader("③ 多面时间编码 (MFTime)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **周期性编码**
        - 周几: 0-6 (工作日/周末)
        - 日内时段: 0-23 (早/晚高峰)
        - 30-min 区间: 0-47
        """)
        st.caption("捕捉'早高峰''周末'等宏观时间规律")
    with c2:
        st.markdown("""
        **时间间隔多尺度编码**
        - λ=10s 尺度: 短暂停车/起步
        - λ=30s 尺度: 正常路段行驶
        - λ=60s 尺度: 交通拥堵/等红灯
        """)
        st.caption("三个尺度分别编码后融合，捕捉微观驾驶节奏")


# ================================================================
# 页面 4: Embedding 投影
# ================================================================
def render_embedding():
    st.title("🔮 Embedding 投影分布")
    st.caption("PCA 降维: 48维特征向量 → 2D 空间，观察相似轨迹是否自然聚集")

    seed = st.slider("抽样种子", 0, 9999, 42, key="emb_seed")
    rec = records[seed % len(records)]
    q_idx = seed % len(records)

    hits = corpus.search(rec, k=5)
    hit_indices = [corpus.idx_of[hrec["trip_id"]] for hrec, _ in hits]

    colors = ["#CBD5E1"] * len(records)
    sizes = [3] * len(records)
    for idx in hit_indices:
        colors[idx] = "#0F766E"
        sizes[idx] = 8
    colors[q_idx] = "#2563EB"
    sizes[q_idx] = 12

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=EMB_2D[:, 0], y=EMB_2D[:, 1],
        mode="markers",
        marker=dict(color=colors, size=sizes, opacity=0.7),
        hovertemplate="轨迹 %{text}<extra></extra>",
        text=[r["trip_id"] for r in records],
    ))
    fig.add_trace(go.Scatter(
        x=[EMB_2D[q_idx, 0]], y=[EMB_2D[q_idx, 1]],
        mode="markers+text",
        marker=dict(size=14, color="#2563EB", symbol="star"),
        text=["查询"], textposition="top center",
    ))
    fig.update_layout(
        height=550, showlegend=False,
        xaxis_title="PC1", yaxis_title="PC2",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("查询轨迹", rec["trip_id"])
    with c2:
        st.metric("距离", f"{rec['cum_dist']/1000:.2f} km")
    with c3:
        top1_dist = corpus.search(rec, k=1)[0][1]
        st.metric("Top-1 距离", f"{top1_dist:.0f} m")

    st.markdown("---")
    st.info("""
    **观察要点**:
    - 蓝色星 = 查询轨迹，绿色点 = Top-5 相似轨迹
    - 如果绿点聚集在蓝星附近 → KNN 检索有效（空间邻近的轨迹 embedding 也邻近）
    - 稀疏区域 = 独特路线；密集区域 = 常见通勤路线
    """)
    st.caption("注: 当前 embedding 为 48 维原始坐标 PCA 降维，非论文训练后的深度表示；仅演示降维可视化效果")


# ================================================================
# 页面注册与渲染
# ================================================================
PAGES = {
    "🔍 结果向检索": render_map_search,
    "🔀 双视图增强": render_augmentation,
    "📐 多尺度时空编码": render_multiscale,
    "🔮 Embedding投影": render_embedding,
}

st.sidebar.title("TrajRL 探索")
selected = st.sidebar.radio("选择功能", list(PAGES.keys()),
    index=list(PAGES.keys()).index(st.session_state.get("current_tab", "🔍 结果向检索")))
if selected != st.session_state.get("current_tab"):
    st.session_state.current_tab = selected
    st.rerun()

PAGES[selected]()
