import streamlit as st
import pandas as pd
from data_loader import load_and_preprocess_data
from clustering import run_clustering
from visualization import calculate_metrics, plot_comparison, plot_individual_clusters, plot_spatial_map

st.set_page_config(page_title="Analisis Fenologi Padi", layout="wide")
st.title("🌾 Analisis Fenologi Padi Berbasis NDVI & Clustering")

# --- STEP 1: UPLOAD DATA ---
st.sidebar.header("📁 1. Upload Data CSV")
file_2023 = st.sidebar.file_uploader("Data Tahun 2023", type=["csv"], key="f23")
file_2024 = st.sidebar.file_uploader("Data Tahun 2024", type=["csv"], key="f24")

files_ready = file_2023 is not None and file_2024 is not None

if files_ready:
    # --- STEP 2: PILIH TAHUN ---
    st.sidebar.header("📅 2. Pilih Tahun Analisis")
    selected_year = st.sidebar.radio("Pilih Tahun:", ["2023", "2024"], index=1)
    
    # Proses data hanya jika tahun berubah atau pertama kali dipilih
    if "current_year" not in st.session_state or st.session_state["current_year"] != selected_year:
        with st.spinner(f"⏳ Memuat & memproses data tahun {selected_year}..."):
            df_combined, nr, nc = load_and_preprocess_data(file_2023, file_2024)
            df_year = df_combined[df_combined['tahun'] == selected_year].copy()
            
            st.session_state["df"] = df_year
            st.session_state["nr"] = nr
            st.session_state["nc"] = nc
            st.session_state["current_year"] = selected_year
            # Reset clustering result when year changes
            if "pivot_df" in st.session_state:
                del st.session_state["pivot_df"]
                
        st.success(f"✅ Data tahun {selected_year} siap! (`{len(df_year):,}` baris)")
    else:
        df_year = st.session_state["df"]
        nr = st.session_state["nr"]
        nc = st.session_state["nc"]

    # --- STEP 3: CLUSTERING CONTROLS ---
    st.sidebar.header("⚙️ 3. Parameter Clustering")
    min_cluster_size = st.sidebar.slider("Min Cluster Size", 2, 20, 3)
    min_samples = st.sidebar.slider("Min Samples", 1, 10, 2)
    epsilon = st.sidebar.slider("Epsilon", 0.0, 0.2, 0.05, 0.01)

    if st.sidebar.button("🚀 Jalankan Clustering", type="primary"):
        with st.spinner("⏳ Menghitung DTW & HDBSCAN (membutuhkan waktu)..."):
            pivot_df = run_clustering(df_year, min_cluster_size, min_samples, epsilon)
        st.session_state["pivot_df"] = pivot_df
        st.success("✅ Clustering selesai!")

    # --- STEP 4: TAMPILKAN HASIL ---
    if "pivot_df" in st.session_state:
        pivot_df = st.session_state["pivot_df"]
        df_fenologi, cluster_ts, valid_statuses = calculate_metrics(df_year, pivot_df)

        if not df_fenologi.empty:
            st.subheader(f"📊 Ringkasan Metrik Fenologi ({selected_year})")
            st.dataframe(df_fenologi.style.format({
                'Puncak NDVI': '{:.2f}', 'Min NDVI': '{:.2f}', 'Amplitudo': '{:.2f}',
                'Rata-rata NDVI': '{:.2f}', 'Rata-rata StdDev': '{:.2f}', 'Jml Titik': '{:,.0f}'
            }).background_gradient(cmap='YlGn', subset=['Puncak NDVI', 'Amplitudo']))

            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_comparison(cluster_ts, valid_statuses, selected_year))
            with col2:
                st.pyplot(plot_individual_clusters(cluster_ts, valid_statuses))
                
            st.pyplot(plot_spatial_map(df_year, nr, nc, valid_statuses))
        else:
            st.warning("⚠️ Tidak ada cluster valid yang terbentuk. Coba turunkan `Min Cluster Size` atau sesuaikan `Epsilon`.")
    else:
        st.info("💡 Klik tombol **Jalankan Clustering** di sidebar untuk memulai analisis.")
else:
    st.info("👆 Silakan upload kedua file CSV (2023 & 2024) di sidebar untuk melanjutkan.")