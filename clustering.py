from tslearn.metrics import cdist_dtw
import hdbscan

def compute_dtw_matrix(df_sampled):
    """Step 5: Hanya hitung matrix DTW."""
    pivot_df = df_sampled.pivot(index='id_lokasi', columns='tanggal', values='NDVI_smooth')
    pivot_df = pivot_df.ffill(axis=1).bfill(axis=1)
    
    data_3d = pivot_df.values[:, :, np.newaxis].astype('float32')
    dist_matrix = cdist_dtw(data_3d, n_jobs=-1)
    return dist_matrix, pivot_df.index

def run_hdbscan_only(dist_matrix, min_cluster, epsilon):
    """Step 6: Hanya ganti-ganti parameter clustering."""
    clusterer = hdbscan.HDBSCAN(
        metric='precomputed',
        min_cluster_size=int(min_cluster),
        cluster_selection_epsilon=float(epsilon),
        gen_min_span_tree=True
    )
    labels = clusterer.fit_predict(dist_matrix.astype('float64'))
    return labels
