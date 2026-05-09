import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tslearn.metrics import cdist_dtw
import hdbscan

def preprocess_ts(df):
    """Mengubah format data menjadi matriks time series dan melakukan smoothing."""
    # Pivot data: baris adalah lokasi, kolom adalah tanggal
    ts_pivot = df.pivot(index='id_lokasi', columns='tanggal', values='NDVI_smooth').dropna()
    
    # Smoothing menggunakan Savitzky-Golay
    ts_smooth = savgol_filter(ts_pivot.values, window_length=5, polyorder=2)
    return ts_pivot.index, ts_smooth

def run_clustering(ts_data, min_cluster_size=5):
    """Menjalankan clustering HDBSCAN dengan metrik jarak DTW."""
    # Kalkulasi matriks jarak menggunakan Dynamic Time Warping
    distance_matrix = cdist_dtw(ts_data)
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='precomputed'
    )
    cluster_labels = clusterer.fit_predict(distance_matrix)
    return cluster_labels
