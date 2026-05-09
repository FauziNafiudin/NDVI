import streamlit as st
import pandas as pd
import numpy as np
import re
from streamlit_drawable_canvas import st_canvas
from data_loader import load_base_data, filter_by_year
from clustering import preprocess_ts, run_clustering
from visualization import plot_spatial_grid, plot_interactive_trends

st.set_page_config(page_title="Monitoring Padi", layout="wide")

st.title("🌾 Rice Phenology Analysis Dashboard")

# Sidebar
st.sidebar.header("Konfigurasi")
file = st.sidebar.file_uploader("Upload CSV Data", type=['csv'])
year = st.sidebar.selectbox("Pilih Tahun", [2023, 2024, 2025])
min_c = st.sidebar.slider("Minimal Anggota Cluster", 2, 20, 5)

if file:
    df = load_base_data(file)
    df_filtered = filter_by_year(df, year)
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("📍 Seleksi Area Spasial")
        # Fitur Canvas sebagai pengganti 'lingkaran geser' di Notebook
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=15,
            drawing_mode="freedraw",
            key="canvas",
            height=400, width=400
        )
        run_btn = st.button("🚀 Jalankan Clustering")

    if run_btn:
        with st.spinner("Memproses Data..."):
            # Analisis
            ids, ts_data = preprocess_ts(df_filtered)
            labels = run_clustering(ts_data, min_cluster_size=min_c)
            
            # Gabungkan Hasil
            res = pd.DataFrame({'id_lokasi': ids, 'cluster_id': labels})
            df_res = df_filtered.merge(res, on='id_lokasi', how='left')
            df_res['status'] = df_res['cluster_id'].apply(lambda x: f"Cluster {int(x)}" if x >= 0 else ("Noise" if x == -1 else "Unselected"))
            
            # Visualisasi di Kolom 2
            with col2:
                st.subheader("📊 Hasil Analisis")
                # (Logika plotting dan metrik fenologi diletakkan di sini)
                st.plotly_chart(plot_interactive_trends(df_res), use_container_width=True)
                st.success("Analisis Selesai!")
else:
    st.info("Silakan unggah data CSV untuk memulai.")
