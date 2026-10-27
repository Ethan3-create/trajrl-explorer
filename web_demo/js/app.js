// TrajRL Explorer - Frontend Application
// =========================================

const API_BASE = '';
const COLORS = {
  query: '#e74c3c',
  hit1: '#3498db',
  hit2: '#2ecc71',
  hit3: '#f39c12',
  hit4: '#9b59b6',
  hit5: '#1abc9c',
};

// ============ State ============
let state = {
  page: 'home',
  records: [],
  totalRecords: 0,
  currentPage: 0,
  pageSize: 20,
  selectedIdx: null,
  selectedTripId: null,
  maps: {},
  charts: {},
  tteStats: null,
};

// ============ Navigation ============
function initNavigation() {
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const page = btn.dataset.page;
      switchPage(page);
    });
  });
}

function switchPage(page) {
  state.page = page;
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.page === page);
  });
  document.querySelectorAll('.page').forEach(p => {
    p.classList.toggle('active', p.id === `page-${page}`);
  });
  
  // Initialize page-specific content
  if (page === 'home') initHomePage();
  if (page === 'search') initSearchPage();
  if (page === 'tte') initTTEPage();
  if (page === 'augment') initAugmentPage();
  if (page === 'patches') initPatchesPage();
  if (page === 'embedding') initEmbeddingPage();
}

// ============ API Helpers ============
async function apiGet(path) {
  const res = await fetch(API_BASE + path);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ============ Home Page ============
async function initHomePage() {
  if (state.totalRecords > 0) return;
  try {
    const stats = await apiGet('/api/stats');
    document.getElementById('stat-total').textContent = stats.total_records.toLocaleString();
    document.getElementById('stat-avg-dist').textContent = (stats.avg_dist / 1000).toFixed(1);
    document.getElementById('stat-avg-time').textContent = (stats.avg_elapsed / 60).toFixed(1);
    document.getElementById('stat-avg-points').textContent = stats.avg_points.toFixed(0);
    state.totalRecords = stats.total_records;
  } catch (e) {
    console.error('Failed to load stats:', e);
  }
}

// ============ Search Page ============
let searchMap = null;
let searchLayers = [];

async function initSearchPage() {
  if (searchMap) return;
  await loadRecordList();
  
  // Initialize map
  searchMap = L.map('search-map').setView([41.15, -8.6], 13);
  L.tileLayer('https://tile.openstreetmap.de/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 19,
  }).addTo(searchMap);
  
  // Select first record
  if (state.records.length > 0) {
    selectRecord(0);
  }
}

async function loadRecordList() {
  const data = await apiGet(`/api/records?page=${state.currentPage}&size=${state.pageSize}`);
  state.records = data.items;
  state.totalRecords = data.total;
  renderRecordList();
  renderPagination();
}

function renderRecordList() {
  const container = document.getElementById('record-list');
  container.innerHTML = state.records.map((rec, i) => `
    <div class="record-item ${state.selectedIdx === i ? 'selected' : ''}" data-idx="${i}">
      <div class="record-id">${rec.trip_id}</div>
      <div class="record-meta">${rec.n_points}点 · ${(rec.cum_dist/1000).toFixed(1)}km · ${(rec.elapsed_inferred/60).toFixed(1)}min</div>
    </div>
  `).join('');
  
  container.querySelectorAll('.record-item').forEach(item => {
    item.addEventListener('click', () => {
      selectRecord(parseInt(item.dataset.idx));
    });
  });
}

function renderPagination() {
  const totalPages = Math.ceil(state.totalRecords / state.pageSize);
  const container = document.getElementById('pagination');
  let html = '';
  
  html += `<button ${state.currentPage === 0 ? 'disabled' : ''} onclick="changePage(${state.currentPage - 1})">←</button>`;
  
  const start = Math.max(0, state.currentPage - 2);
  const end = Math.min(totalPages, start + 5);
  for (let i = start; i < end; i++) {
    html += `<button class="${i === state.currentPage ? 'active' : ''}" onclick="changePage(${i})">${i + 1}</button>`;
  }
  
  html += `<button ${state.currentPage >= totalPages - 1 ? 'disabled' : ''} onclick="changePage(${state.currentPage + 1})">→</button>`;
  container.innerHTML = html;
}

function changePage(page) {
  const totalPages = Math.ceil(state.totalRecords / state.pageSize);
  if (page < 0 || page >= totalPages) return;
  state.currentPage = page;
  loadRecordList();
}

async function selectRecord(idx) {
  state.selectedIdx = idx;
  state.selectedTripId = state.records[idx].trip_id;
  renderRecordList();
  
  const rec = state.records[idx];
  const detail = await apiGet(`/api/record/${rec.trip_id}`);
  const k = parseInt(document.getElementById('search-k')?.value || '5');
  const searchResults = await apiGet(`/api/search/${rec.trip_id}?k=${k}`);
  
  renderSearchDetail(detail, searchResults);
  renderSearchMap(detail, searchResults);
}

function renderSearchDetail(detail, results) {
  const container = document.getElementById('search-detail');
  container.innerHTML = `
    <div class="detail-section">
      <div class="detail-label">查询轨迹</div>
      <div class="detail-value mono">${detail.trip_id}</div>
    </div>
    <div class="detail-section">
      <div class="detail-label">基本信息</div>
      <div class="detail-value">${detail.n_points} 个 GPS 点 · ${(detail.cum_dist/1000).toFixed(2)} km</div>
      <div class="detail-value">推断时长: ${(detail.elapsed_inferred/60).toFixed(1)} 分钟</div>
      <div class="detail-value">司机ID: ${detail.driver_id}</div>
    </div>
    <div class="detail-section">
      <div class="detail-label">最相似轨迹 (${results.results.length}条)</div>
      ${results.results.map((r, i) => `
        <div class="detail-value" style="margin-bottom: 8px; padding: 8px; background: var(--bg-primary); border-radius: 4px;">
          <span style="color: ${Object.values(COLORS)[i+1] || COLORS.hit5}">●</span> 
          <strong>#${i+1}</strong> ${r.trip_id}<br>
          <span style="font-size: 11px; color: var(--text-muted);">
            特征距离: ${r.distance.toFixed(0)} · 路径 ${(r.cum_dist/1000).toFixed(2)}km · 时长 ${(r.elapsed/60).toFixed(1)}min
          </span>
        </div>
      `).join('')}
    </div>
  `;
}

function renderSearchMap(detail, results) {
  if (!searchMap) return;
  
  // Clear layers
  searchLayers.forEach(l => searchMap.removeLayer(l));
  searchLayers = [];
  
  // Query trajectory
  const qPts = detail.points.map(p => [p[1], p[0]]);
  const qLine = L.polyline(qPts, { color: COLORS.query, weight: 3, opacity: 0.9 }).addTo(searchMap);
  searchLayers.push(qLine);
  
  // Start/end markers for query
  const startIcon = L.divIcon({ className: 'map-marker', html: '<div style="width:10px;height:10px;background:#e74c3c;border-radius:50%;border:2px solid white;"></div>' });
  const endIcon = L.divIcon({ className: 'map-marker', html: '<div style="width:10px;height:10px;background:#c0392b;border-radius:50%;border:2px solid white;"></div>' });
  searchLayers.push(L.marker(qPts[0], { icon: startIcon }).addTo(searchMap).bindPopup('起点'));
  searchLayers.push(L.marker(qPts[qPts.length-1], { icon: endIcon }).addTo(searchMap).bindPopup('终点'));
  
  // Result trajectories
  results.results.forEach((r, i) => {
    const color = Object.values(COLORS)[i+1] || COLORS.hit5;
    if (r.points && r.points.length > 0) {
      const pts = r.points.map(p => [p[1], p[0]]);
      const line = L.polyline(pts, { color, weight: 2, opacity: 0.7, dashArray: '5, 5' }).addTo(searchMap);
      searchLayers.push(line);
    }
  });
  
  // Fit bounds
  searchMap.fitBounds(qLine.getBounds().pad(0.1));
}

// ============ TTE Page ============
let tteMap = null;
let tteChart = null;
let tteTrips = [];

async function initTTEPage() {
  if (tteMap) return;

  // 轨迹列表（前 200 条）
  const data = await apiGet('/api/records?page=0&size=200');
  tteTrips = data.items;
  const listEl = document.getElementById('tte-list');
  listEl.innerHTML = tteTrips.map((rec, i) => `
    <div class="record-item" data-idx="${i}">
      <div class="record-id">${rec.trip_id}</div>
      <div class="record-meta">${rec.n_points}点 · ${(rec.cum_dist/1000).toFixed(1)}km · ${(rec.elapsed_inferred/60).toFixed(1)}min</div>
    </div>
  `).join('');
  listEl.querySelectorAll('.record-item').forEach(item => {
    item.addEventListener('click', () => selectTTETrip(parseInt(item.dataset.idx)));
  });

  // 亮色底图（CartoDB Positron，Porto 街道清晰可见）
  tteMap = L.map('tte-map').setView([41.15, -8.6], 13);
  L.tileLayer('https://tile.openstreetmap.de/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 19,
  }).addTo(tteMap);

  // 选中第一条
  if (tteTrips.length > 0) selectTTETrip(0);
}

async function selectTTETrip(idx) {
  document.querySelectorAll('#tte-list .record-item').forEach(el => el.classList.remove('selected'));
  document.querySelector(`#tte-list .record-item[data-idx="${idx}"]`)?.classList.add('selected');

  const tid = tteTrips[idx].trip_id;
  const data = await apiGet(`/api/trajectory-tte/${tid}`);

  // ---- 地图：路径 + 起终点 ----
  tteMap.eachLayer(l => { if (l instanceof L.Polyline || l instanceof L.Marker) tteMap.removeLayer(l); });
  const latlngs = data.points.map(p => [p[1], p[0]]);
  const line = L.polyline(latlngs, { color: '#2563eb', weight: 4, opacity: 0.85 }).addTo(tteMap);
  const mk = (pos, color, label) => L.marker(pos, {
    icon: L.divIcon({ className: '', html: `<div style="width:10px;height:10px;background:${color};border-radius:50%;border:2px solid #fff;box-shadow:0 1px 3px rgba(0,0,0,.4)"></div>` })
  }).addTo(tteMap).bindTooltip(label, { direction: 'top', offset: [0, -6] });
  mk(latlngs[0], '#10b981', '起点');
  mk(latlngs[latlngs.length - 1], '#ef4444', '终点');
  tteMap.fitBounds(line.getBounds().pad(0.12));

  // ---- 信息栏 ----
  const f = data.features;
  document.getElementById('tte-info').innerHTML = `
    <b>${data.trip_id}</b><br>
    <span style="font-size:12px;color:var(--text-muted)">
      距离 ${(f.cum_dist/1000).toFixed(2)} km · ${f.n_points} 个 GPS 点 · 弯曲度 ${f.curvature.toFixed(2)} · 推断时长 ${(f.elapsed_inferred/60).toFixed(1)} min
    </span>
  `;

  // ---- 路径特征卡片 ----
  document.getElementById('tte-features').innerHTML = `
    <span class="feat-chip">📏 距离 <b>${(f.cum_dist/1000).toFixed(2)} km</b></span>
    <span class="feat-chip">🕒 时间步数 <b>${f.n_points}</b></span>
    <span class="feat-chip">🌀 弯曲度 <b>${f.curvature.toFixed(2)}</b></span>
    <span class="feat-chip">🎯 真实(推断) <b>${(f.elapsed_inferred/60).toFixed(1)} min</b></span>
  `;

  // ---- 三种预测对比卡片 ----
  const p = data.predictions;
  const cards = [
    { key: 'baseline', label: '距离基线', title: '📏 距离线性基线', sub: '仅按距离：a×d+b', value: p.baseline.pred, err: p.baseline, color: '#f59e0b' },
    { key: 'feature', label: '特征化回归', title: '🧬 特征化多元回归', sub: '距离+时间步数+弯曲度', value: p.feature.pred, err: p.feature, color: '#8b5cf6' },
    { key: 'true', label: '真实推断值', title: '🎯 真实推断值', sub: '15s 均匀采样', value: f.elapsed_inferred, err: null, color: '#10b981' },
  ];
  document.getElementById('tte-cards').innerHTML = cards.map(c => {
    const mins = (c.value / 60).toFixed(1);
    const errHtml = c.err
      ? `<div class="pred-err ${c.err.err_s >= 0 ? 'pos' : 'neg'}">误差 ${c.err.err_s >= 0 ? '+' : ''}${(c.err.err_s / 60).toFixed(1)} min (${c.err.err_pct >= 0 ? '+' : ''}${c.err.err_pct.toFixed(1)}%)</div>`
      : '<div class="pred-err true">参照值</div>';
    return `
      <div class="pred-card" style="--pc:${c.color}">
        <div class="pred-card-title">${c.title}</div>
        <div class="pred-card-sub">${c.sub}</div>
        <div class="pred-card-value">${mins} <span>分钟</span></div>
        ${errHtml}
      </div>
    `;
  }).join('');

  // ---- 全量模型指标对比（每次返回时刷新，保证与后端一致）----
  renderTTEModelBrief(data.models);

  // ---- 对比柱状图 ----
  renderTTEChart(cards);
}

function renderTTEModelBrief(models) {
  const b = models.baseline.metrics;
  const f2 = models.feature.metrics;
  document.getElementById('tte-model-brief').innerHTML = `
    <table class="tte-model-table">
      <thead><tr><th>模型</th><th>R²</th><th>MAE (秒)</th><th>MAPE (%)</th></tr></thead>
      <tbody>
        <tr><td>📏 距离线性基线</td><td>${b.r2.toFixed(3)}</td><td>${b.mae_s.toFixed(0)}</td><td>${b.mape_pct.toFixed(1)}</td></tr>
        <tr><td>🧬 特征化多元回归</td><td>${f2.r2.toFixed(3)}</td><td>${f2.mae_s.toFixed(0)}</td><td>${f2.mape_pct.toFixed(1)}</td></tr>
      </tbody>
    </table>
    <p class="model-note">在全量 ${models.n_train.toLocaleString()} 条轨迹上拟合。⚠️ 数据口径：真实值为推断时长（15s 均匀采样，= 15s × 时间步数），故特征化模型利用「时间步数 + 路径弯曲度」后误差接近零——这直观体现了 §IV-C 多面时间信息（时间序列长度）对 TTE 的决定性作用；实际路网中该增益来自时空联合建模。</p>
  `;
}

function renderTTEChart(cards) {
  if (!tteChart) tteChart = echarts.init(document.getElementById('tte-chart'));
  const cats = cards.map(c => c.label);
  const vals = cards.map(c => +(c.value / 60).toFixed(1));
  tteChart.setOption({
    backgroundColor: 'transparent',
    grid: { left: 55, right: 20, top: 30, bottom: 30 },
    xAxis: {
      type: 'category',
      data: cats,
      axisLine: { lineStyle: { color: '#64748b' } },
      axisLabel: { color: '#94a3b8', interval: 0 },
    },
    yAxis: {
      type: 'value',
      name: '分钟',
      nameLocation: 'middle',
      nameGap: 35,
      axisLine: { lineStyle: { color: '#64748b' } },
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: '#334155' } },
    },
    series: [{
      type: 'bar',
      data: vals.map((v, i) => ({
        value: v,
        itemStyle: { color: cards[i].color, borderRadius: [6, 6, 0, 0] },
      })),
      barWidth: 44,
      label: { show: true, position: 'top', color: '#e2e8f0', formatter: '{c} min' },
    }],
    tooltip: {
      trigger: 'item',
      backgroundColor: '#1e293b',
      borderColor: '#334155',
      textStyle: { color: '#f1f5f9' },
    },
  });
}

// ============ Augment Page ============
let augmentMaps = { orig: null, v1: null, v2: null };

async function initAugmentPage() {
  if (augmentMaps.orig) return;
  
  // Load record list
  const data = await apiGet('/api/records?page=0&size=50');
  const container = document.getElementById('augment-list');
  container.innerHTML = data.items.map((rec, i) => `
    <div class="record-item" data-trip-id="${rec.trip_id}" onclick="selectAugmentRecord('${rec.trip_id}')">
      <div class="record-id">${rec.trip_id}</div>
      <div class="record-meta">${rec.n_points}点 · ${(rec.cum_dist/1000).toFixed(1)}km</div>
    </div>
  `).join('');
  
  // Init maps
  ['orig', 'v1', 'v2'].forEach(key => {
    augmentMaps[key] = L.map(`augment-map-${key}`).setView([41.15, -8.6], 13);
    L.tileLayer('https://tile.openstreetmap.de/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors',
      maxZoom: 18,
    }).addTo(augmentMaps[key]);
  });
  
  // Select first
  if (data.items.length > 0) {
    selectAugmentRecord(data.items[0].trip_id);
  }
}

async function selectAugmentRecord(tripId) {
  const data = await apiGet(`/api/augment/${tripId}?seed=42`);
  
  // Update info
  document.getElementById('orig-info').innerHTML = `
    原始: ${data.original.trip_id} · ${(data.original.elapsed/60).toFixed(1)}min
  `;
  document.getElementById('v1-info').innerHTML = `
    视图A: 偏移段 ${data.view1.shift_start}-${data.view1.shift_end} · ${(data.view1.elapsed/60).toFixed(1)}min
  `;
  document.getElementById('v2-info').innerHTML = `
    视图B: 偏移段 ${data.view2.shift_start}-${data.view2.shift_end} · ${(data.view2.elapsed/60).toFixed(1)}min
  `;
  
  // Render maps
  renderAugmentMap('orig', data.original.points, '#3b82f6');
  renderAugmentMap('v1', data.view1.points, '#ef4444');
  renderAugmentMap('v2', data.view2.points, '#10b981');
}

function renderAugmentMap(key, points, color) {
  const map = augmentMaps[key];
  map.eachLayer(l => { if (l instanceof L.Polyline || l instanceof L.Marker) map.removeLayer(l); });
  
  const latlngs = points.map(p => [p[1], p[0]]);
  const line = L.polyline(latlngs, { color, weight: 3, opacity: 0.8 }).addTo(map);
  
  const startIcon = L.divIcon({ className: '', html: `<div style="width:8px;height:8px;background:${color};border-radius:50%;border:2px solid white;"></div>` });
  L.marker(latlngs[0], { icon: startIcon }).addTo(map);
  
  map.fitBounds(line.getBounds().pad(0.1));
}

// ============ Patches Page ============
let patchMap = null;

async function initPatchesPage() {
  if (patchMap) return;
  
  const data = await apiGet('/api/records?page=0&size=50');
  const container = document.getElementById('patch-list');
  container.innerHTML = data.items.map((rec, i) => `
    <div class="record-item" data-trip-id="${rec.trip_id}" onclick="selectPatchRecord('${rec.trip_id}')">
      <div class="record-id">${rec.trip_id}</div>
      <div class="record-meta">${rec.n_points}点</div>
    </div>
  `).join('');
  
  patchMap = L.map('patch-map-orig').setView([41.15, -8.6], 13);
  L.tileLayer('https://tile.openstreetmap.de/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 19,
  }).addTo(patchMap);
  
  // Omega slider
  document.getElementById('omega-slider').addEventListener('input', function() {
    document.getElementById('omega-val').textContent = this.value;
    document.getElementById('patch-omega').textContent = this.value;
    if (state.selectedTripId) selectPatchRecord(state.selectedTripId);
  });
  
  if (data.items.length > 0) {
    selectPatchRecord(data.items[0].trip_id);
  }
}

async function selectPatchRecord(tripId) {
  state.selectedTripId = tripId;
  const omega = document.getElementById('omega-slider').value;
  const data = await apiGet(`/api/patches/${tripId}?omega=${omega}`);
  
  // Render original map
  patchMap.eachLayer(l => { if (l instanceof L.Polyline || l instanceof L.Marker) patchMap.removeLayer(l); });
  const latlngs = data.points.map(p => [p[1], p[0]]);
  const line = L.polyline(latlngs, { color: '#3b82f6', weight: 3, opacity: 0.7 }).addTo(patchMap);
  patchMap.fitBounds(line.getBounds().pad(0.1));
  
  // Render patch visualization
  renderPatchViz(data);
}

function renderPatchViz(data) {
  const container = document.getElementById('patch-viz');
  const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#06b6d4', '#e74c3c', '#2ecc71'];
  
  container.innerHTML = `
    <div style="padding: 16px; display: flex; flex-direction: column; gap: 8px;">
      <div style="font-size: 14px; color: var(--text-secondary); margin-bottom: 8px;">
        共 ${data.n_patches} 个 Patch (ω=${data.omega})
      </div>
      ${data.patches.map((p, i) => {
        const color = colors[i % colors.length];
        return `
          <div style="display: flex; align-items: center; gap: 12px; padding: 8px; background: var(--bg-secondary); border-radius: 6px;">
            <div style="width: 40px; height: 40px; background: ${color}20; border: 2px solid ${color}; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; color: ${color};">
              P${i+1}
            </div>
            <div style="flex: 1;">
              <div style="font-size: 13px; color: var(--text-primary); font-weight: 500;">
                Patch ${i+1}: ${p.n_points} 个点
              </div>
              <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">
                质心: (${p.centroid.lon.toFixed(4)}, ${p.centroid.lat.toFixed(4)})
              </div>
            </div>
            <div style="width: 120px; height: 30px; background: var(--bg-primary); border-radius: 4px; position: relative; overflow: hidden;">
              ${p.points.slice(0, 20).map((pt, j) => `
                <div style="position: absolute; width: 3px; height: 3px; background: ${color}; border-radius: 50%; 
                            left: ${(j / Math.min(p.points.length, 20)) * 100}%; top: 50%; transform: translateY(-50%);"></div>
              `).join('')}
            </div>
          </div>
        `;
      }).join('')}
    </div>
  `;
}

// ============ Embedding Page ============
let embChart = null;

async function initEmbeddingPage() {
  if (embChart) return;
  
  const data = await apiGet('/api/embeddings/projection');
  
  embChart = echarts.init(document.getElementById('embedding-chart'));
  renderEmbeddingChart(data, 'cum_dist');
  
  document.getElementById('color-map').addEventListener('change', function() {
    renderEmbeddingChart(data, this.value);
  });
}

function renderEmbeddingChart(data, colorBy) {
  const values = data.points.map(p => p[colorBy]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  
  const option = {
    backgroundColor: 'transparent',
    grid: { left: 60, right: 30, top: 30, bottom: 50 },
    xAxis: {
      type: 'value',
      name: 'PC1',
      nameLocation: 'middle',
      nameGap: 30,
      axisLine: { lineStyle: { color: '#64748b' } },
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: '#334155' } },
    },
    yAxis: {
      type: 'value',
      name: 'PC2',
      nameLocation: 'middle',
      nameGap: 40,
      axisLine: { lineStyle: { color: '#64748b' } },
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: '#334155' } },
    },
    visualMap: {
      min,
      max,
      dimension: 2,
      orient: 'horizontal',
      right: 10,
      top: 0,
      textStyle: { color: '#94a3b8' },
      inRange: {
        color: ['#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444'],
      },
    },
    series: [{
      type: 'scatter',
      data: data.points.map(p => [p.x, p.y, p[colorBy]]),
      symbolSize: 4,
      itemStyle: { opacity: 0.6 },
      emphasis: {
        itemStyle: { opacity: 1, borderColor: '#fff', borderWidth: 2 },
      },
    }],
    tooltip: {
      trigger: 'item',
      formatter: (p) => {
        const point = data.points[p.dataIndex];
        return `轨迹: ${point.trip_id}<br>距离: ${(point.cum_dist/1000).toFixed(1)}km<br>时长: ${(point.elapsed/60).toFixed(1)}min`;
      },
      backgroundColor: '#1e293b',
      borderColor: '#334155',
      textStyle: { color: '#f1f5f9' },
    },
  };
  
  embChart.setOption(option);
  
  // Click handler
  embChart.off('click');
  embChart.on('click', async (params) => {
    const point = data.points[params.dataIndex];
    const detail = await apiGet(`/api/record/${point.trip_id}`);
    const neighbors = await apiGet(`/api/embedding/${point.trip_id}/neighbors?k=5`);
    
    document.getElementById('emb-detail').innerHTML = `
      <div class="detail-section">
        <div class="detail-label">选中轨迹</div>
        <div class="detail-value mono">${point.trip_id}</div>
        <div class="detail-value">${detail.n_points} 点 · ${(detail.cum_dist/1000).toFixed(2)}km · ${(detail.elapsed_inferred/60).toFixed(1)}min</div>
      </div>
      <div class="detail-section">
        <div class="detail-label">Embedding 空间邻近轨迹</div>
        ${neighbors.neighbors.map((n, i) => `
          <div class="detail-value" style="margin-bottom: 6px; padding: 6px; background: var(--bg-primary); border-radius: 4px; font-size: 12px;">
            #${i+1} ${n.trip_id} · 2D距离: ${Math.sqrt(n.dist2d).toFixed(2)}
          </div>
        `).join('')}
      </div>
    `;
  });
}

// ============ Init ============
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initHomePage();
  
  // Search input
  document.getElementById('search-input')?.addEventListener('input', function() {
    const term = this.value.toLowerCase();
    document.querySelectorAll('#record-list .record-item').forEach(item => {
      const id = item.querySelector('.record-id').textContent.toLowerCase();
      item.style.display = id.includes(term) ? '' : 'none';
    });
  });
  
  // K selector
  document.getElementById('search-k')?.addEventListener('change', function() {
    if (state.selectedTripId) {
      selectRecord(state.selectedIdx);
    }
  });
  
  // Resize charts
  window.addEventListener('resize', () => {
    tteChart?.resize();
    embChart?.resize();
  });
});
