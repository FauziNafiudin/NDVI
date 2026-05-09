import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_and_preprocess_data
from clustering import run_clustering
from visualization import calculate_metrics, plot_comparison, plot_individual_clusters, plot_spatial_map

st.set_page_config(page_title="Analisis Fenologi Padi", layout="wide")
st.title("🌾 Analisis Fenologi Padi Berbasis NDVI & Clustering")

# Sidebar Controls
st.sidebar.header("⚙️ Parameter Pengolahan")
TARGET_YEAR = st.sidebar.selectbox("Tahun Target", ["2023", "2024"], index=1)
WINDOW_SIZE = st.sidebar.slider("Window Size (Savgol)", 11, 51, 31, step=2)
POLY_ORDER = st.sidebar.slider("Polynomial Order", 2, 5, 2)

st.sidebar.header("🔍 Parameter Clustering")
MIN_CLUSTER_SIZE = st.sidebar.slider("Min Cluster Size", 2, 20, 3)
MIN_SAMPLES = st.sidebar.slider("Min Samples", 1, 10, 2)
EPSILON = st.sidebar.slider("Cluster Selection Epsilon", 0.0, 0.2, 0.05, 0.01)

# Load Data
with st.spinner("📥 Memuat dan memproses data..."):
    file_2023 = "Data_NDVI_Lamongan_2023.csv"
    file_2024 = "Data_NDVI_Lamongan_2024.csv"
    try:
        df_final, nr, nc = load_and_preprocess_data(file_2023, file_2024, WINDOW_SIZE, POLY_ORDER)
        df = df_final[df_final['tahun'] == str(TARGET_YEAR)].copy()
        st.success(f"✅ Data berhasil dimuat: `{len(df):,}` baris untuk tahun `{TARGET_YEAR}`")
    except FileNotFoundError:
        st.error("❌ File CSV tidak ditemukan. Pastikan `Data_NDVI_Lamongan_2023.csv` dan `Data_NDVI_Lamongan_2024.csv` berada di folder yang sama.")
        st.stop()

# Trigger Clustering
if st.sidebar.button("🚀 Jalankan Clustering", type="primary"):
    with st.spinner("⏳ Menghitung DTW & HDBSCAN (membutuhkan waktu)..."):
        pivot_df = run_clustering(df, MIN_CLUSTER_SIZE, MIN_SAMPLES, EPSILON)
    st.session_state['pivot_df'] = pivot_df
    st.success("✅ Clustering selesai! Hasil divisualisasikan di bawah.")

# Display Results
if 'pivot_df' in st.session_state:
    pivot_df = st.session_state['pivot_df']
    df_fenologi, cluster_ts, valid_statuses = calculate_metrics(df, pivot_df)

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
            
        st.pyplot(plot_spatial_map(df, nr, nc, valid_statuses))
    else:
        st.warning("⚠️ Tidak ada cluster valid yang terbentuk. Coba turunkan `Min Cluster Size` atau sesuaikan `Epsilon`.")
else:
    st.info("💡 Klik tombol **Jalankan Clustering** di sidebar untuk memulai analisis.")