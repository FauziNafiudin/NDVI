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

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.pcolormesh(np.arange(nc + 1), np.arange(nr + 1), grid,
                  cmap=ListedColormap(['#D3D3D3', '#C8E6C9', '#2196F3']),
                  vmin=0, vmax=2, edgecolors='white', linewidth=0.3, shading='auto')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(f'Lokasi Tersampling: {len(sampled_ids):,}', fontweight='bold', fontsize=13)
    ax.legend(handles=[
        Patch(facecolor='#2196F3', edgecolor='white', label=f'Tersampling ({len(sampled_ids):,})'),
        Patch(facecolor='#C8E6C9', edgecolor='white', label='Tidak Tersampling'),
        Patch(facecolor='#D3D3D3', edgecolor='white', label='Kosong'),
    ], loc='upper right')
    plt.tight_layout()
    return fig

def plot_sample_ts_preview(df_year, sampled_ids, n=3):
    import random
    preview = random.sample(list(sampled_ids), k=min(n, len(sampled_ids)))
    fig, axes = plt.subplots(1, len(preview), figsize=(5 * len(preview), 4), sharey=True)
    if len(preview) == 1: axes = [axes]
    colors = sns.color_palette('tab10', len(preview))
    for ax, loc_id, color in zip(axes, preview, colors):
        sub = df_year[df_year['id_lokasi'] == loc_id].sort_values('tanggal')
        ax.plot(sub['tanggal'], sub['NDVI_smooth'], color=color, linewidth=1.8)
        ax.fill_between(sub['tanggal'], sub['NDVI_smooth'], alpha=0.15, color=color)
        ax.set_title(loc_id, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=30, labelsize=7)
    axes[0].set_ylabel('NDVI Smoothed')
    plt.tight_layout()
    return fig

def calculate_metrics(df, pivot_df):
    df = assign_status_column(df, pivot_df)
    valid_statuses = get_valid_statuses(df)
    df_clusters = df[df['cluster_id'] >= 0].copy()
    if df_clusters.empty:
        return pd.DataFrame(), pd.DataFrame(), [], df

    cluster_ts = df_clusters.groupby(['status', 'tanggal'])['NDVI_smooth'].agg(
        ['mean', 'std']
    ).reset_index()

    rows = []
    for s in valid_statuses:
        sub = cluster_ts[cluster_ts['status'] == s]
        if sub.empty: continue
        peak_idx = sub['mean'].idxmax()
        rows.append({
            'Status': s,
            'Jml Titik': df_clusters[df_clusters['status'] == s]['id_lokasi'].nunique(),
            'Puncak NDVI': sub['mean'].max(),
            'Waktu Puncak': sub.loc[peak_idx, 'tanggal'].date(),
            'Min NDVI': sub['mean'].min(),
            'Amplitudo': sub['mean'].max() - sub['mean'].min(),
            'Rata-rata NDVI': sub['mean'].mean(),
            'Rata-rata StdDev': sub['std'].mean(),
        })
    df_fenologi = pd.DataFrame(rows).set_index('Status') if rows else pd.DataFrame()
    return df_fenologi, cluster_ts, valid_statuses, df

def plot_comparison(cluster_ts, valid_statuses, tahun):
    palette = sns.color_palette('tab10', len(valid_statuses))
    colors = {s: palette[i] for i, s in enumerate(valid_statuses)}
    y_min = (cluster_ts['mean'] - cluster_ts['std']).min() - 0.05
    y_max = (cluster_ts['mean'] + cluster_ts['std']).max() + 0.05
    fig, ax = plt.subplots(figsize=(14, 6))
    for s in valid_statuses:
        sub = cluster_ts[cluster_ts['status'] == s]
        ax.plot(sub['tanggal'], sub['mean'], label=s, color=colors[s], linewidth=2.5)
        ax.fill_between(sub['tanggal'], sub['mean'] - sub['std'], sub['mean'] + sub['std'],
                        color=colors[s], alpha=0.1)
    ax.set_title(f'Perbandingan Tren NDVI Antar Cluster ({tahun})', fontsize=14, fontweight='bold')
    ax.set_ylabel('NDVI Smoothed')
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.subplots_adjust(right=0.82)
    return fig

def plot_individual_clusters(cluster_ts, valid_statuses):
    palette = sns.color_palette('tab10', len(valid_statuses))
    colors = {s: palette[i] for i, s in enumerate(valid_statuses)}
    n_cols = 2
    n_rows = max(1, (len(valid_statuses) + 1) // n_cols)
    y_min = (cluster_ts['mean'] - cluster_ts['std']).min() - 0.05
    y_max = (cluster_ts['mean'] + cluster_ts['std']).max() + 0.05
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for i, s in enumerate(valid_statuses):
        sub = cluster_ts[cluster_ts['status'] == s]
        axes[i].plot(sub['tanggal'], sub['mean'], color=colors[s], linewidth=2.5)
        axes[i].fill_between(sub['tanggal'], sub['mean'] - sub['std'], sub['mean'] + sub['std'],
                             color=colors[s], alpha=0.15)
        axes[i].set_title(f'Detail: {s}', fontsize=12, fontweight='bold')
        axes[i].set_ylim(y_min, y_max)
        axes[i].grid(True, alpha=0.2, linestyle='--')
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout(pad=3.5)
    return fig

def plot_spatial_map(df_labeled, df_all_available, nr, nc, valid_statuses):
    NOISE_COLOR = '#E74C3C'
    EMPTY_COLOR = '#D3D3D3'
    UNSEL_COLOR = '#C8E6C9' # Hijau pudar
    
    palette = sns.color_palette('tab10', len(valid_statuses))
    
    # Inisialisasi grid: 0=Kosong, 1=Available(Unselected/Noise/Cluster), 2=Khusus mapping nanti
    # Strategi: Isi semua available dulu dengan 1, lalu override dengan cluster/noise
    grid = np.zeros((nr, nc), dtype=int)
    
    # Tandai semua lokasi yang punya data
    avail_ids = set(df_all_available[['grid_row', 'grid_col']].drop_duplicates().values.tolist())
    for r, c in avail_ids:
        grid[r, c] = 1

    status_int = {'Noise': 3} # 3 agar tidak bentrok dengan 1 (available) dan 0 (empty)
    for i, s in enumerate(valid_statuses):
        status_int[s] = i + 10

    gmap = df_labeled[['grid_row', 'grid_col', 'status']].drop_duplicates()
    for _, row in gmap.iterrows():
        r, c = int(row['grid_row']), int(row['grid_col'])
        grid[r, c] = status_int.get(row['status'], 1)

    # Mapping warna: 0->Empty, 1->Unselected(GreenPudar), 2->?, 3->Noise, 10+->Cluster
    # Kita buat colormap dinamis
    max_val = max(status_int.values()) if status_int else 3
    cmap_colors = [EMPTY_COLOR] * (max_val + 1)
    cmap_colors[1] = UNSEL_COLOR # Available
    if 3 in cmap_colors: cmap_colors[3] = NOISE_COLOR # Noise
    for i, s in enumerate(valid_statuses):
        idx = i + 10
        if idx < len(cmap_colors):
            cmap_colors[idx] = mpl.colors.rgb2hex(palette[i])

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.pcolormesh(np.arange(nc + 1), np.arange(nr + 1), grid,
                  cmap=ListedColormap(cmap_colors), edgecolors='white', linewidth=0.3, shading='auto')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title('Peta Sebaran Cluster', fontweight='bold', fontsize=14)
    
    legend_els = [
        Patch(facecolor=NOISE_COLOR, edgecolor='white', label='Noise') if (df_labeled['status']=='Noise').any() else None,
    ]
    legend_els = [el for el in legend_els if el is not None]
    # Tambahkan Unselected jika ada
    if (grid == 1).any():
        legend_els.append(Patch(facecolor=UNSEL_COLOR, edgecolor='white', label='Data Tersedia (Tidak Dipilih)'))
        
    for i, s in enumerate(valid_statuses):
        legend_els.append(Patch(facecolor=mpl.colors.rgb2hex(palette[i]), edgecolor='white', label=s))
        
    ax.legend(handles=legend_els, title='Keterangan', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.75)
    return fig
