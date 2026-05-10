"""
🌾 Analisis Fenologi Padi — NDVI Clustering Pipeline
Alur vertikal top-to-bottom. Sampling via canvas paint interaktif.
"""
import json
import random
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from data_loader import load_raw_data, apply_smoothing
from clustering import run_clustering, get_dtw_description
from visualization import (
    plot_grid_preview,
    plot_sample_grid,
    plot_sample_ts_preview,
    calculate_metrics,
    plot_comparison,
    plot_individual_clusters,
    plot_spatial_map,
)

# ─────────────────────────────────────────────
#  PAGE CONFIG & CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title="Fenologi Padi", page_icon="🌾", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
  .step-card {
    border: 1.5px solid #dcedc8;
    border-left: 5px solid #388e3c;
    border-radius: 10px;
    padding: 1.4rem 1.8rem 1rem;
    margin-bottom: 0.5rem;
    background: #f9fff9;
  }
  .step-card h3 { margin-top: 0; color: #1b5e20; font-size: 1.1rem; }
  .step-divider { border: none; border-top: 2px dashed #c8e6c9; margin: 1.8rem 0; }
  .badge-ok   { display:inline-block; background:#e8f5e9; color:#2e7d32;
                border:1px solid #a5d6a7; border-radius:20px;
                padding:0.2rem 0.9rem; font-size:0.85rem; font-weight:600; }
  .badge-warn { display:inline-block; background:#fff8e1; color:#e65100;
                border:1px solid #ffe082; border-radius:20px;
                padding:0.2rem 0.9rem; font-size:0.85rem; font-weight:600; }
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:linear-gradient(135deg,#1b5e20,#388e3c,#1b5e20);
            padding:1.8rem 2.5rem; border-radius:12px; margin-bottom:2rem; color:white;">
  <h1 style="margin:0;font-size:1.9rem;">🌾 Analisis Fenologi Padi</h1>
  <p style="margin:0.3rem 0 0; opacity:.85;">Pipeline NDVI · DTW · HDBSCAN — Kabupaten Lamongan</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  STEP 1 — LOAD DATA
# ═══════════════════════════════════════════════════════
st.markdown('<div class="step-card"><h3>📁 Step 1 — Load Data</h3>', unsafe_allow_html=True)

if "df_raw" not in st.session_state:
    if st.button("⬇️ Muat Data", type="primary", use_container_width=True):
        bar = st.progress(0)
        try:
            df_raw, nr, nc = load_raw_data(
                progress_callback=lambda p, t: bar.progress(p, text=t))
            bar.empty()
            st.session_state.update(df_raw=df_raw, nr=nr, nc=nc)
            st.rerun()
        except Exception as e:
            bar.empty()
            st.error(f"❌ Gagal membaca data: {e}")
            st.stop()
else:
    df_raw = st.session_state["df_raw"]
    nr, nc = st.session_state["nr"], st.session_state["nc"]
    n_lok = df_raw['id_lokasi'].nunique()
    cols = st.columns(4)
    cols[0].metric("Lokasi Unik", f"{n_lok:,}")
    cols[1].metric("Total Baris", f"{len(df_raw):,}")
    cols[2].metric("Dimensi Grid", f"{nr}×{nc}")
    cols[3].metric("Periode", f"{df_raw['tanggal'].min().date()} → {df_raw['tanggal'].max().date()}")
    st.markdown('<span class="badge-ok">✅ Data siap</span>', unsafe_allow_html=True)
    with st.expander("🗺️ Peta Grid Seluruh Lokasi", expanded=False):
        st.pyplot(plot_grid_preview(df_raw, nr, nc))

st.markdown('</div>', unsafe_allow_html=True)
if "df_raw" not in st.session_state:
    st.stop()


# ═══════════════════════════════════════════════════════
#  STEP 2 — PILIH TAHUN
# ═══════════════════════════════════════════════════════
st.markdown('<hr class="step-divider">', unsafe_allow_html=True)
st.markdown('<div class="step-card"><h3>📅 Step 2 — Pilih Tahun Analisis</h3>', unsafe_allow_html=True)

c1, c2, _ = st.columns([1, 1, 5])
for yr, col in [("2023", c1), ("2024", c2)]:
    is_active = st.session_state.get("tahun") == yr
    if col.button(f"📆 {yr}", type="primary" if is_active else "secondary", use_container_width=True):
        if st.session_state.get("tahun") != yr:
            for k in ["df_smooth", "df_year", "sampled_ids", "pivot_df"]:
                st.session_state.pop(k, None)
        st.session_state["tahun"] = yr
        st.rerun()

if "tahun" in st.session_state:
    st.markdown(f'<span class="badge-ok">✅ Tahun: {st.session_state["tahun"]}</span>',
                unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
if "tahun" not in st.session_state:
    st.info("👆 Pilih tahun terlebih dahulu.")
    st.stop()

tahun = st.session_state["tahun"]
df_raw_year = df_raw[df_raw["tahun"] == tahun].copy()


# ═══════════════════════════════════════════════════════
#  STEP 3 — PREPROCESSING (Savitzky-Golay)
# ═══════════════════════════════════════════════════════
st.markdown('<hr class="step-divider">', unsafe_allow_html=True)
st.markdown('<div class="step-card"><h3>⚙️ Step 3 — Preprocessing (Savitzky-Golay)</h3>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
window_size = c1.slider("Window Size", 5, 61, 31, 2, help="Harus ganjil")
poly_order  = c2.slider("Polynomial Order", 1, 5, 2)

if st.button("▶ Jalankan Smoothing", type="primary", use_container_width=True):
    bar = st.progress(0)
    df_smooth = apply_smoothing(df_raw_year, window_size, poly_order,
                                progress_callback=lambda p, t: bar.progress(p, text=t))
    bar.empty()
    st.session_state["df_smooth"] = df_smooth
    st.session_state["df_year"]   = df_smooth[df_smooth["tahun"] == tahun].copy()
    for k in ["sampled_ids", "pivot_df"]:
        st.session_state.pop(k, None)
    st.rerun()

if "df_smooth" in st.session_state:
    df_year = st.session_state["df_year"]
    n_ts = df_year["tanggal"].nunique()
    st.markdown(f'<span class="badge-ok">✅ {df_year["id_lokasi"].nunique():,} lokasi × {n_ts} hari</span>',
                unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
if "df_smooth" not in st.session_state:
    st.info("👆 Klik **Jalankan Smoothing** untuk melanjutkan.")
    st.stop()


# ═══════════════════════════════════════════════════════
#  STEP 4 — SAMPLING (Canvas Paint Interaktif)
# ═══════════════════════════════════════════════════════
st.markdown('<hr class="step-divider">', unsafe_allow_html=True)
st.markdown('<div class="step-card"><h3>🎯 Step 4 — Sampling Data (Paint Mode)</h3>', unsafe_allow_html=True)
st.caption("Klik-drag untuk memilih lokasi. Klik kanan untuk deselect. Atau gunakan Random.")

df_year = st.session_state["df_year"]

# Bangun lookup grid → id_lokasi (untuk dikirim ke JS)
id_to_pos = (df_year[['id_lokasi', 'grid_row', 'grid_col']]
             .drop_duplicates('id_lokasi')
             .copy())

# grid_data: list of {r, c, id} — hanya sel yang ada lokasi
grid_cells = [
    {"r": int(row.grid_row), "c": int(row.grid_col), "id": row.id_lokasi}
    for row in id_to_pos.itertuples()
]

# Kirim data grid ke JS sebagai JSON
grid_json = json.dumps(grid_cells)
nr_js, nc_js = int(nr), int(nc)

# Inisialisasi selected dari session (supaya persistent saat rerun)
init_selected = json.dumps(st.session_state.get("sampled_ids", []))

CANVAS_HTML = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: sans-serif; background: #f9fff9; padding: 8px; }}

  #toolbar {{
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    margin-bottom: 8px; padding: 8px 12px;
    background: #fff; border: 1px solid #c8e6c9; border-radius: 8px;
  }}
  #toolbar label {{ font-size: 13px; color: #333; }}
  #counter {{
    font-weight: 700; color: #2e7d32; font-size: 14px;
    margin-left: auto; background: #e8f5e9;
    padding: 4px 12px; border-radius: 20px;
  }}
  button {{
    padding: 5px 14px; border: none; border-radius: 6px;
    font-size: 13px; cursor: pointer; font-weight: 600;
  }}
  #btn-random {{ background: #1976d2; color: #fff; }}
  #btn-clear  {{ background: #e53935; color: #fff; }}
  #btn-confirm {{ background: #388e3c; color: #fff; }}
  button:hover {{ opacity: 0.85; }}

  #canvas-wrap {{
    overflow: auto; border: 1px solid #c8e6c9; border-radius: 8px;
    background: #fff; max-height: 560px;
  }}
  canvas {{ display: block; cursor: crosshair; }}

  #msg {{
    margin-top: 6px; font-size: 12px; color: #555; min-height: 18px;
  }}
</style>
</head>
<body>

<div id="toolbar">
  <label>Brush Size:
    <input id="brush" type="range" min="1" max="12" value="3"
           style="width:90px;vertical-align:middle;">
    <span id="brush-val">3</span>
  </label>
  <label style="margin-left:6px;">Random:
    <input id="n-random" type="number" value="100" min="1" style="width:70px;padding:3px 6px;border:1px solid #ccc;border-radius:4px;">
  </label>
  <button id="btn-random">🎲 Random</button>
  <button id="btn-clear">🗑 Clear</button>
  <button id="btn-confirm">✅ Confirm Seleksi</button>
  <span id="counter">Terseleksi: 0</span>
</div>

<div id="canvas-wrap">
  <canvas id="c"></canvas>
</div>
<div id="msg">💡 Klik kiri drag = pilih &nbsp;|&nbsp; Klik kanan drag = hapus pilihan</div>

<script>
const NR = {nr_js};
const NC = {nc_js};
const GRID_CELLS = {grid_json};
const INIT_SELECTED = {init_selected};

// Warna
const C_EMPTY    = '#D3D3D3';
const C_AVAIL    = '#4CAF50';
const C_SELECTED = '#FF5722';

// State
const available = new Uint8Array(NR * NC);   // 1 = ada lokasi
const selected  = new Uint8Array(NR * NC);   // 1 = terpilih
const cellId    = new Array(NR * NC).fill(null); // id_lokasi per sel

GRID_CELLS.forEach(d => {{
  const idx = d.r * NC + d.c;
  available[idx] = 1;
  cellId[idx] = d.id;
}});

// Init selected dari session
const initSet = new Set(INIT_SELECTED);
GRID_CELLS.forEach(d => {{
  if (initSet.has(d.id)) selected[d.r * NC + d.c] = 1;
}});

// Canvas setup
const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');

// Hitung cell size agar muat di lebar layar
const WRAP_W = Math.min(window.innerWidth - 40, 900);
const CELL   = Math.max(3, Math.floor(Math.min(WRAP_W / NC, 560 / NR)));
canvas.width  = NC * CELL;
canvas.height = NR * CELL;

function draw() {{
  for (let r = 0; r < NR; r++) {{
    for (let c = 0; c < NC; c++) {{
      const idx = r * NC + c;
      ctx.fillStyle = !available[idx] ? C_EMPTY
                    : selected[idx]   ? C_SELECTED
                    : C_AVAIL;
      ctx.fillRect(c * CELL, r * CELL, CELL - 1, CELL - 1);
    }}
  }}
  const n = selected.reduce((s, v) => s + v, 0);
  document.getElementById('counter').textContent = 'Terseleksi: ' + n.toLocaleString();
}}

draw();

// Brush
const brushSlider = document.getElementById('brush');
const brushVal    = document.getElementById('brush-val');
brushSlider.addEventListener('input', () => {{ brushVal.textContent = brushSlider.value; }});

function getBrushR() {{ return parseInt(brushSlider.value); }}

function applyBrush(x, y, mode) {{
  const col0 = Math.floor(x / CELL);
  const row0 = Math.floor(y / CELL);
  const R    = getBrushR();
  for (let r = Math.max(0, row0 - R); r <= Math.min(NR-1, row0 + R); r++) {{
    for (let c = Math.max(0, col0 - R); c <= Math.min(NC-1, col0 + R); c++) {{
      const dist = Math.sqrt((r - row0)**2 + (c - col0)**2);
      if (dist <= R) {{
        const idx = r * NC + c;
        if (available[idx]) selected[idx] = mode;
      }}
    }}
  }}
}}

// Drag
let dragging = false, dragMode = 0;

canvas.addEventListener('mousedown', e => {{
  e.preventDefault();
  dragging = true;
  dragMode = e.button === 2 ? 0 : 1;
  const rect = canvas.getBoundingClientRect();
  applyBrush(e.clientX - rect.left, e.clientY - rect.top, dragMode);
  draw();
}});

canvas.addEventListener('mousemove', e => {{
  if (!dragging) return;
  const rect = canvas.getBoundingClientRect();
  applyBrush(e.clientX - rect.left, e.clientY - rect.top, dragMode);
  draw();
}});

window.addEventListener('mouseup', () => {{ dragging = false; }});
canvas.addEventListener('contextmenu', e => e.preventDefault());

// Touch support
canvas.addEventListener('touchstart', e => {{
  e.preventDefault();
  dragging = true; dragMode = 1;
  const t = e.touches[0];
  const rect = canvas.getBoundingClientRect();
  applyBrush(t.clientX - rect.left, t.clientY - rect.top, 1);
  draw();
}}, {{passive: false}});
canvas.addEventListener('touchmove', e => {{
  e.preventDefault();
  const t = e.touches[0];
  const rect = canvas.getBoundingClientRect();
  applyBrush(t.clientX - rect.left, t.clientY - rect.top, dragMode);
  draw();
}}, {{passive: false}});
canvas.addEventListener('touchend', () => {{ dragging = false; }});

// Random
document.getElementById('btn-random').addEventListener('click', () => {{
  selected.fill(0);
  const avail = [];
  GRID_CELLS.forEach(d => avail.push(d.r * NC + d.c));
  const n = Math.min(parseInt(document.getElementById('n-random').value) || 100, avail.length);
  // Fisher-Yates shuffle untuk ambil n random
  for (let i = avail.length - 1; i > 0; i--) {{
    const j = Math.floor(Math.random() * (i + 1));
    [avail[i], avail[j]] = [avail[j], avail[i]];
  }}
  for (let i = 0; i < n; i++) selected[avail[i]] = 1;
  draw();
}});

// Clear
document.getElementById('btn-clear').addEventListener('click', () => {{
  selected.fill(0);
  draw();
}});

// Confirm → kirim ke Streamlit
document.getElementById('btn-confirm').addEventListener('click', () => {{
  const ids = [];
  for (let r = 0; r < NR; r++) {{
    for (let c = 0; c < NC; c++) {{
      const idx = r * NC + c;
      if (selected[idx] && cellId[idx]) ids.push(cellId[idx]);
    }}
  }}
  const msg = document.getElementById('msg');
  if (ids.length === 0) {{
    msg.textContent = '⚠️ Belum ada sel yang dipilih!';
    msg.style.color = '#e53935';
    return;
  }}
  msg.textContent = '✅ ' + ids.length + ' lokasi dikonfirmasi — copy JSON di bawah lalu klik Konfirmasi.';
  msg.style.color = '#2e7d32';
  // Tampilkan JSON di textarea tersembunyi di dalam iframe
  document.getElementById('json-out').value = JSON.stringify(ids);
  document.getElementById('json-wrap').style.display = 'block';
}});
</script>

<!-- Output JSON bridge -->
<div id="json-wrap" style="display:none; margin-top:8px;">
  <label style="font-size:12px;color:#555;">📋 Salin JSON ini ke field di bawah canvas:</label><br>
  <textarea id="json-out" rows="3"
    style="width:100%;font-size:11px;font-family:monospace;border:1px solid #a5d6a7;
           border-radius:6px;padding:6px;background:#f1f8e9;resize:vertical;"
    readonly></textarea>
</div>

</body>
</html>
"""

components.html(CANVAS_HTML, height=700, scrolling=False)

# ── Bridge: user copy-paste JSON dari canvas ke sini ─────────
st.markdown("**Langkah terakhir:** Setelah klik **✅ Confirm Seleksi** di canvas, "
            "salin JSON yang muncul dan paste ke kolom di bawah ini:")

confirmed_json = st.text_area(
    "JSON ID Lokasi (dari canvas di atas)",
    value=st.session_state.get("_canvas_json", ""),
    height=90,
    placeholder='["LOC_0001","LOC_0023", ...]  ← paste hasil dari canvas',
    key="canvas_json_input",
    label_visibility="collapsed",
)

c_btn1, c_btn2 = st.columns([2, 5])
with c_btn1:
    do_confirm = st.button("🔒 Konfirmasi & Lanjut", type="primary", use_container_width=True)

if do_confirm:
    raw = confirmed_json.strip()
    if not raw:
        st.warning("⚠️ Field kosong — klik Confirm di canvas dulu, lalu paste JSON-nya.")
    else:
        try:
            ids_list = json.loads(raw)
            if not isinstance(ids_list, list) or len(ids_list) == 0:
                st.warning("⚠️ JSON tidak valid atau list kosong.")
            else:
                st.session_state["sampled_ids"] = ids_list
                st.session_state["_canvas_json"] = raw
                st.session_state.pop("pivot_df", None)
                st.rerun()
        except json.JSONDecodeError:
            st.error("❌ Bukan format JSON yang valid. Pastikan copy dari textarea canvas.")

if "sampled_ids" in st.session_state:
    sampled_ids = st.session_state["sampled_ids"]
    st.markdown(f'<span class="badge-ok">✅ {len(sampled_ids):,} lokasi terkonfirmasi</span>',
                unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🗺️ Peta Sebaran Sampel", "📈 Preview Time Series"])
    with tab1:
        st.pyplot(plot_sample_grid(df_year, sampled_ids, nr, nc))
    with tab2:
        st.pyplot(plot_sample_ts_preview(df_year, sampled_ids, n=3))

st.markdown('</div>', unsafe_allow_html=True)
if "sampled_ids" not in st.session_state:
    st.info("👆 Pilih lokasi di canvas lalu klik **Confirm Seleksi** dan **Konfirmasi & Lanjut**.")
    st.stop()


# ═══════════════════════════════════════════════════════
#  STEP 5 — DESKRIPSI & DTW
# ═══════════════════════════════════════════════════════
st.markdown('<hr class="step-divider">', unsafe_allow_html=True)
st.markdown('<div class="step-card"><h3>📊 Step 5 — Ringkasan Data & Hitung DTW</h3>', unsafe_allow_html=True)

sampled_ids = st.session_state["sampled_ids"]
df_sampled  = df_year[df_year["id_lokasi"].isin(sampled_ids)].copy()
dtw_info    = get_dtw_description(df_sampled)

cols = st.columns(len(dtw_info))
for col, (k, v) in zip(cols, dtw_info.items()):
    col.metric(k, v)

with st.expander("Lihat sampel data (5 baris)"):
    st.dataframe(df_sampled[['id_lokasi', 'tanggal', 'NDVI', 'NDVI_smooth',
                              'lat_y', 'lon_x']].head(5), use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  STEP 6 — CLUSTERING HDBSCAN
# ═══════════════════════════════════════════════════════
st.markdown('<hr class="step-divider">', unsafe_allow_html=True)
st.markdown('<div class="step-card"><h3>🔍 Step 6 — Clustering HDBSCAN</h3>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
min_cluster_size = c1.slider("Min Cluster Size", 2, 20, 3)
min_samples      = c2.slider("Min Samples",      1, 10, 2)
epsilon          = c3.slider("Epsilon",           0.0, 0.5, 0.05, 0.01)

if st.button("🚀 Jalankan DTW + HDBSCAN", type="primary", use_container_width=True):
    bar = st.progress(0)
    pivot_df = run_clustering(
        df_sampled, min_cluster_size, min_samples, epsilon,
        progress_callback=lambda p, t: bar.progress(p, text=t)
    )
    bar.empty()
    st.session_state["pivot_df"] = pivot_df
    n_cls   = len(set(pivot_df['cluster'])) - (1 if -1 in pivot_df['cluster'].values else 0)
    n_noise = (pivot_df['cluster'] == -1).sum()
    st.success(f"✅ Selesai! {n_cls} cluster · {n_noise} noise")
    st.rerun()

if "pivot_df" in st.session_state:
    pv = st.session_state["pivot_df"]
    n_cls   = len(set(pv['cluster'])) - (1 if -1 in pv['cluster'].values else 0)
    n_noise = (pv['cluster'] == -1).sum()
    st.markdown(f'<span class="badge-ok">✅ {n_cls} cluster · {n_noise} noise</span>',
                unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
if "pivot_df" not in st.session_state:
    st.info("👆 Klik **Jalankan DTW + HDBSCAN** untuk memulai.")
    st.stop()


# ═══════════════════════════════════════════════════════
#  STEP 7 — HASIL AKHIR
# ═══════════════════════════════════════════════════════
st.markdown('<hr class="step-divider">', unsafe_allow_html=True)
st.markdown('<div class="step-card"><h3>📈 Step 7 — Hasil Akhir</h3>', unsafe_allow_html=True)

pivot_df = st.session_state["pivot_df"]
df_fenologi, cluster_ts, valid_statuses, df_labeled = calculate_metrics(df_year, pivot_df)

if df_fenologi.empty:
    st.warning("⚠️ Tidak ada cluster valid. Coba turunkan Min Cluster Size atau sesuaikan Epsilon.")
else:
    st.subheader(f"📰 Tabel Metrik Fenologi ({tahun})")
    st.dataframe(
        df_fenologi.style
        .format({'Puncak NDVI': '{:.3f}', 'Min NDVI': '{:.3f}', 'Amplitudo': '{:.3f}',
                 'Rata-rata NDVI': '{:.3f}', 'Rata-rata StdDev': '{:.3f}', 'Jml Titik': '{:,.0f}'})
        .background_gradient(cmap='YlGn', subset=['Puncak NDVI', 'Amplitudo']),
        use_container_width=True
    )

    st.subheader("📊 Grafik Time Series")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_comparison(cluster_ts, valid_statuses, tahun))
    with col2:
        st.pyplot(plot_individual_clusters(cluster_ts, valid_statuses))

    st.subheader("🗺️ Peta Sebaran Cluster")
    st.pyplot(plot_spatial_map(df_labeled, nr, nc, valid_statuses))

st.markdown('</div>', unsafe_allow_html=True)
