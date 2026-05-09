import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import re

def _natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split('([0-9]+)', s)]

def assign_status_column(df, pivot_df):
    cluster_mapping = pivot_df['cluster'].to_dict()
    df = df.copy()
    df['cluster_id'] = df['id_lokasi'].map(cluster_mapping).fillna(-2)
    def _label(row):
        if row['cluster_id'] == -2: return 'Unselected'
        if row['cluster_id'] == -1: return 'Noise'
        return f"Cluster {int(row['cluster_id'])}"
    df['status'] = df.apply(_label, axis=1)
    return df

def get_valid_statuses(df):
    return sorted(
        [s for s in df['status'].unique() if s.startswith('Cluster')],
        key=_natural_sort_key
    )

def plot_grid_preview(df, nr, nc):
    grid = np.zeros((nr, nc), dtype=int)
    for _, row in df[['grid_row', 'grid_col']].drop_duplicates().iterrows():
        grid[int(row['grid_row']), int(row['grid_col'])] = 1
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.pcolormesh(np.arange(nc + 1), np.arange(nr + 1), grid,
                  cmap=ListedColormap(['#D3D3D3', '#4CAF50']),
                  edgecolors='white', linewidth=0.4, shading='auto')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title('Sebaran Lokasi — Seluruh Data', fontweight='bold', fontsize=13)
    ax.legend(handles=[
        Patch(facecolor='#4CAF50', edgecolor='white', label='Ada Data'),
        Patch(facecolor='#D3D3D3', edgecolor='white', label='Kosong'),
    ], loc='upper right')
    plt.tight_layout()
    return fig

# BARU: Preview Smoothing
def plot_smoothing_preview(df_year, n=3):
    sample_ids = df_year['id_lokasi'].unique()[:n]
    fig, axes = plt.subplots(1, len(sample_ids), figsize=(5 * len(sample_ids), 4), sharey=True)
    if len(sample_ids) == 1: axes = [axes]
    
    for ax, loc_id in zip(axes, sample_ids):
        sub = df_year[df_year['id_lokasi'] == loc_id].sort_values('tanggal')
        # NDVI Asli (transparan)
        ax.plot(sub['tanggal'], sub['NDVI'], color='gray', alpha=0.4, linewidth=1, label='Raw NDVI')
        # NDVI Smooth (tegas)
        ax.plot(sub['tanggal'], sub['NDVI_smooth'], color='#1b5e20', linewidth=2, label='Smoothed')
        
        ax.set_title(loc_id, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=30, labelsize=7)
        if ax == axes[0]:
            ax.set_ylabel('NDVI')
            ax.legend(fontsize=8)
    plt.tight_layout()
    return fig

def plot_sample_grid(df_year, sampled_ids, nr, nc):
    sampled_set = set(sampled_ids)
    id_to_pos = df_year[['id_lokasi', 'grid_row', 'grid_col']].drop_duplicates('id_lokasi')
    
    # 0: Kosong, 1: Available (Hijau Pudar), 2: Sampling (Biru)
    grid = np.zeros((nr, nc), dtype=int)
    for _, row in id_to_pos.iterrows():
        r, c = int(row['grid_row']), int(row['grid_col'])
        grid[r, c] = 2 if row['id_lokasi'] in sampled_set else 1

    fig, ax = plt.subplots(figsize=(9, 
