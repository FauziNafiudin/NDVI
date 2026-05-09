"""
🌾 Analisis Fenologi Padi — NDVI Clustering Pipeline
Optimasi: Zero-Copy Selection, Heavy DTW Caching, & Adaptive Grid Visualization.
"""
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from data_loader import load_raw_data, apply_smoothing
from clustering import compute_dtw_matrix, run_hdbscan_only
from visualization import (
    plot_grid_general,
    plot_smoothing_comparison,
    plot_sample_ts_preview,
    calculate_metrics,
    plot_individual_clusters
)

# ─────────────────────────────────────────────
#  PAGE CONFIG & CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title="Fenologi Padi", page_icon="🌾", layout="wide")

st.markdown("""
<style>
    .reportview-container .main .block-container { max-width: 95%; }
    .step-card {
        border-left: 5px solid #2e7d32;
        background-color: #f1f8e9;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN APP LOGIC
# ─────────────────────────────────────────────

st.title("🌾 Pipeline Analisis Fenologi Padi")

# --- STEP 1: LOAD DATA & GLOBAL GRID PREVIEW ---
# Data dimuat sekali di awal
if 'df_raw' not in st.session_state:
    with st.spinner("Memuat data utama..."):
        df, nr, nc = load_raw_data()
        st.session_state['df_raw'] = df
        st.session_state['n_rows'] = nr
        st.session_state['n_cols'] = nc

df_raw = st.session_state['df_raw']
n_rows = st.session_state['n_rows']
n_cols = st.session_state['n_cols']

st.markdown('<div class="step-card"><h3>Step 1 — Sebaran Lokasi Seluruh Data</h3></div>', unsafe_allow_html=True)
# Peta Grid Seluruh Lokasi (Muncul SEBELUM pilih tahun)
fig_grid_total = plot_grid_general(df_raw, nr=n_rows, nc=n_cols)
st.pyplot(fig_grid_total)

# --- STEP 2: SELEKSI TAHUN & SMOOTHING ---
st.markdown('<div class="step-card"><h3>Step 2 — Tahun Analisis & Smoothing</h3></div>', unsafe_allow_html=True)
col_y, col_s = st.columns(2)
tahun = col_y.selectbox("Pilih Tahun Analisis", ["2023", "2024"])

if st.button("Proses Smoothing Savitzky-Golay"):
    df_year = df_raw[df_raw['tahun'] == tahun].copy()
    with st.spinner(f"Smoothing data tahun {tahun}..."):
        df_smoothed = apply_smoothing(df_year)
        st.session_state['df_smoothed'] = df_smoothed
    st.success(f"Smoothing selesai untuk tahun {tahun}")

if 'df_smoothed' in st.session_state:
    # Contoh grafik perbandingan Savitzky vs Asli
    st.markdown("#### Contoh Hasil Smoothing")
    sample_id = st.session_state['df_smoothed']['id_lokasi'].iloc[0]
    fig_smooth = plot_smoothing_comparison(st.session_state['df_smoothed'], sample_id)
    st.pyplot(fig_smooth)

# --- STEP 3 & 4: SAMPLING OTOMATIS ---
st.markdown('<div class="step-card"><h3>Step 3 & 4 — Sampling Data (Area Interest)</h3></div>', unsafe_allow_html=True)
# (Disini Anda menggunakan komponen canvas Anda)
# Logika: Setelah konfirmasi, id_lokasi langsung masuk session_state
if st.button("Konfirmasi Seleksi"):
    # Mock-up logika filter berdasarkan koordinat terpilih dari canvas
    # terpilih = filter_id_from_canvas(canvas_data)
    # st.session_state['df_sampled'] = st.session_state['df_smoothed'][st.session_state['df_smoothed']['id_lokasi'].isin(terpilih)]
    st.info("Data sampling telah dikonfirmasi secara otomatis.")

if 'df_sampled' in st.session_state:
    st.subheader("Peta Sebaran Sampel (Grid)")
    # Warna biru untuk yang tersampling, hijau transparan untuk sisanya
    fig_samp = plot_grid_general(df_raw, st.session_state['df_sampled'], mode='sampling', nr=n_rows, nc=n_cols)
    st.pyplot(fig_samp)
    
    st.subheader("Preview Time Series Sampel")
    st.pyplot(plot_sample_ts_preview(st.session_state['df_sampled']))

# --- STEP 5: HITUNG DTW (JALANKAN SEKALI) ---
st.markdown('<div class="step-card"><h3>Step 5 — Hitung DTW (Matrix Jarak)</h3></div>', unsafe_allow_html=True)
if 'df_sampled' in st.session_state:
    if st.button("Jalankan DTW"):
        with st.spinner("Menghitung matriks DTW (Proses Berat)..."):
            dist_matrix, pivot_idx = compute_dtw_matrix(st.session_state['df_sampled'])
            st.session_state['dist_matrix'] = dist_matrix
            st.session_state['pivot_idx'] = pivot_idx
        st.success("Distance Matrix berhasil disimpan!")

# --- STEP 6: TUNING HDBSCAN (INTERAKTIF) ---
st.markdown('<div class="step-card"><h3>Step 6 — Clustering HDBSCAN</h3></div>', unsafe_allow_html=True)
if 'dist_matrix' in st.session_state:
    c1, c2 = st.columns(2)
    m_size = c1.slider("Min Cluster Size", 2, 100, 10)
    eps_val = c2.slider("Epsilon (Cluster Selection)", 0.0, 1.0, 0.05, 0.01)
    
    if st.button("Jalankan Clustering"):
        with st.spinner("Mengelompokkan data..."):
            labels = run_hdbscan_only(st.session_state['dist_matrix'], m_size, eps_val)
            # Simpan hasil label ke dataframe pivot
            result_df = pd.DataFrame({
                'id_lokasi': st.session_state['pivot_idx'],
                'cluster': labels
            })
            st.session_state['cluster_result'] = result_df
        st.success("Clustering Selesai!")

# --- STEP 7: VISUALISASI AKHIR ---
if 'cluster_result' in st.session_state:
    st.markdown('<div class="step-card"><h3>Step 7 — Hasil Akhir & Analisis Spasial</h3></div>', unsafe_allow_html=True)
    
    # Merge label cluster ke data spasial
    df_final = df_raw.merge(st.session_state['cluster_result'], on='id_lokasi', how='left')
    df_overlay = df_final[df_final['cluster'].notna()]
    
    st.subheader("Peta Sebaran Cluster (Grid)")
    # Sesuai catatan: Background hijau pudar, terpilih warna cluster, noise abu
    fig_final = plot_grid_general(df_raw, df_overlay, mode='cluster', nr=n_rows, nc=n_cols)
    st.pyplot(fig_final)
    
    # Tabel Metrik & Grafik Per Cluster
    df_metrics, cluster_ts = calculate_metrics(st.session_state['df_sampled'], st.session_state['cluster_result'])
    st.dataframe(df_metrics, use_container_width=True)
    
    st.subheader("Karakteristik Tiap Cluster")
    st.pyplot(plot_individual_clusters(cluster_ts))
