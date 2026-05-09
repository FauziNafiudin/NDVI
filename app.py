import streamlit as st
import pandas as pd
from data_loader import load_and_preprocess_data
from clustering import run_clustering
from visualization import (
    calculate_metrics,
    plot_comparison,
    plot_individual_clusters,
    plot_spatial_map,
    show_manual_selection_tool
)

st.set_page_config(page_title="Analisis Fenologi Padi", layout="wide")
st.title("🌾 Analisis Fenologi Padi Berbasis NDVI & Clustering")

st.sidebar.header("⚙️ Parameter Pengolahan")
TARGET_YEAR = st.sidebar.selectbox("Tahun Target", ["2023", "2024"], index=1)
WINDOW_SIZE = st.sidebar.slider("Window Size (Savgol)", 11, 51, 31, step=2)
POLY_ORDER = st.sidebar.slider("Polynomial Order", 2, 5, 2)

with st.spinner("📥 Memuat dan memproses data..."):
    file_2023 = "Data_NDVI_Lamongan_2023.csv"
    file_2024 = "Data_NDVI_Lamongan_2024.csv"
    try:
        df_final, nr, nc = load_and_preprocess_data(file_2023, file_2024, WINDOW_SIZE, POLY_ORDER)
        df_all = df_final[df_final['tahun'] == str(TARGET_YEAR)].copy()
        st.success(f"✅ Data berhasil dimuat: `{len(df_all):,}` baris untuk tahun `{TARGET_YEAR}`")
    except FileNotFoundError:
        st.error("❌ File CSV tidak ditemukan. Pastikan file berada di folder yang sama.")
        st.stop()

# State management for manual selection
if 'selected_df' in st.session_state:
    df_active = st.session_state['selected_df']
    st.info(f"📍 **Mode Aktif: Seleksi Manual** ({df_active['id_lokasi'].nunique()} lokasi terpilih).")
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Reset (Gunakan Semua Data)"):
            del st.session_state['selected_df']
            st.rerun()
    with col2:
        if st.button("🎨 Edit Seleksi"):
            st.session_state['show_canvas'] = True
else:
    df_active = df_all
    st.sidebar.divider()
    st.sidebar.header("🎨 Alat Seleksi Manual")
    if st.sidebar.button("Buka Kanvas Seleksi"):
        st.session_state['show_canvas'] = True

if st.session_state.get('show_canvas'):
    st.divider()
    st.subheader("Mode Seleksi Manual")
    st.info("Gambar area pada peta di bawah. Klik 'Konfirmasi' untuk menyimpan.")
    show_manual_selection_tool(df_all, nr, nc)
    if st.button("❌ Batal / Tutup Kanvas"):
        st.session_state['show_canvas'] = False
        st.rerun()
    st.stop()

st.divider()
st.header("Hasil Analisis Clustering")
if st.button("🚀 Jalankan Clustering", type="primary"):
    with st.spinner("⏳ Menghitung DTW & HDBSCAN..."):
        pivot_df = run_clustering(df_active, min_cluster_size=3, min_samples=2, cluster_selection_epsilon=0.05)
    st.session_state['pivot_df'] = pivot_df
    st.success("✅ Clustering selesai!")

if 'pivot_df' in st.session_state:
    pivot_df = st.session_state['pivot_df']
    df_fenologi, cluster_ts, valid_statuses = calculate_metrics(df_active, pivot_df)
    if not df_fenologi.empty:
        st.subheader(f"📊 Ringkasan Metrik Fenologi ({TARGET_YEAR})")
        st.dataframe(df_fenologi.style.format({
            'Puncak NDVI': '{:.2f}', 'Min NDVI': '{:.2f}', 'Amplitudo': '{:.2f}',
            'Rata-rata NDVI': '{:.2f}', 'Rata-rata StdDev': '{:.2f}', 'Jml Titik': '{:,.0f}'
        }).background_gradient(cmap='YlGn', subset=['Puncak NDVI', 'Amplitudo']))

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_comparison(cluster_ts, valid_statuses, TARGET_YEAR))
        with col2:
            st.pyplot(plot_individual_clusters(cluster_ts, valid_statuses))
        st.pyplot(plot_spatial_map(df_active, nr, nc, valid_statuses))
    else:
        st.warning("⚠️ Tidak ada cluster valid yang terbentuk.")
else:
    st.info("💡 Klik tombol **Jalankan Clustering** untuk memulai analisis.")