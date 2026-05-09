import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import ListedColormap

def plot_grid_general(df_all, df_overlay=None, mode='sampling', nr=0, nc=0):
    """
    Fungsi general untuk peta grid.
    - df_all: seluruh data lokasi (background).
    - df_overlay: data yang terpilih/terfilter.
    """
    # Ukuran diperbesar sesuai window browser
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Base layer: Semua data (Warna hijau lebih transparan/pudar)
    grid_base = np.zeros((nr, nc))
    for _, row in df_all[['grid_row', 'grid_col']].drop_duplicates().iterrows():
        grid_base[int(row['grid_row']), int(row['grid_col'])] = 1
    
    ax.pcolormesh(grid_base, cmap=ListedColormap(['white', '#a1d99b']), alpha=0.3, edgecolors='white', linewidth=0.1)

    if df_overlay is not None:
        if mode == 'sampling':
            # Yang tersampling warna biru
            grid_sel = np.zeros((nr, nc))
            for _, row in df_overlay[['grid_row', 'grid_col']].drop_duplicates().iterrows():
                grid_sel[int(row['grid_row']), int(row['grid_col'])] = 1
            mask = np.ma.masked_where(grid_sel == 0, grid_sel)
            ax.pcolormesh(mask, cmap=ListedColormap(['#3498db']), edgecolors='white', linewidth=0.1)
            
        elif mode == 'cluster':
            # Warna sesuai cluster, sisanya (noise) abu-abu
            grid_cluster = np.full((nr, nc), -1.0)
            cluster_ids = df_overlay['cluster'].unique()
            # Mapping cluster ke warna
            palette = sns.color_palette('tab10', len([c for c in cluster_ids if c >= 0]))
            
            for _, row in df_overlay.iterrows():
                grid_cluster[int(row['grid_row']), int(row['grid_col'])] = row['cluster']
            
            # Plot Noise (Abu-abu)
            mask_noise = np.ma.masked_where(grid_cluster != -1, grid_cluster)
            ax.pcolormesh(mask_noise, cmap=ListedColormap(['#95a5a6']), edgecolors='white', linewidth=0.1)
            
            # Plot Clusters
            for i, c_id in enumerate([c for c in cluster_ids if c >= 0]):
                mask_c = np.ma.masked_where(grid_cluster != c_id, grid_cluster)
                ax.pcolormesh(mask_c, cmap=ListedColormap([mpl.colors.rgb2hex(palette[i])]), edgecolors='white', linewidth=0.1)

    ax.invert_yaxis()
    ax.set_aspect('equal')
    plt.axis('off')
    return fig

def plot_smoothing_comparison(df_raw, id_lokasi):
    """Grafik Savitzky vs Asli dengan beda opacity."""
    data = df_raw[df_raw['id_lokasi'] == id_lokasi].sort_values('tanggal')
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Grafik asli (lebih transparan)
    ax.plot(data['tanggal'], data['NDVI'], label='Asli', color='gray', alpha=0.3, linewidth=1)
    # Grafik Savitzky (lebih jelas)
    ax.plot(data['tanggal'], data['NDVI_smooth'], label='Savitzky-Golay', color='green', alpha=1.0, linewidth=2)
    
    ax.set_title(f"Perbandingan Smoothing - Lokasi {id_lokasi}")
    ax.legend()
    return fig
