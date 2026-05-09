import pandas as pd
import numpy as np
from tslearn.metrics import cdist_dtw
import hdbscan
import streamlit as st

def run_clustering(df_year, min_cluster_size=3, min_samples=2, cluster_selection_epsilon=0.05):
    """Menjalankan DTW + HDBSCAN pada data tahun terpilih."""
    pivot_df = df_year.pivot(index='id_lokasi', columns='tanggal', values='NDVI_smooth')
    pivot_df = pivot_df.ffill(axis=1).bfill(axis=1)
    
    data_3d = pivot_df.values[:, :, np.newaxis].astype(np.float32)
    
    progress_bar = st.progress(0, text="🔄 Menghitung Matriks DTW...")
    dist_matrix = cdist_dtw(data_3d, n_jobs=-1, verbose=0)
    progress_bar.progress(0.5, text="🔄 Menjalankan HDBSCAN...")

    clusterer = hdbscan.HDBSCAN(
        metric='precomputed',
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        gen_min_span_tree=True,
        prediction_data=True
    )
    labels = clusterer.fit_predict(dist_matrix.astype(np.float64))
    progress_bar.progress(1.0, text="✅ Clustering Selesai!")
    progress_bar.empty()

    pivot_df['cluster'] = labels
    return pivot_df