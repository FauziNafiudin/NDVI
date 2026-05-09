import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import re

def calculate_metrics(df, pivot_df):
    cluster_mapping = pivot_df['cluster'].to_dict()
    df['cluster_id'] = df['id_lokasi'].map(cluster_mapping).fillna(-2)

    def assign_full_status(row):
        if row['cluster_id'] == -2: return 'Unselected'
        if row['cluster_id'] == -1: return 'Noise'
        return f"Cluster {int(row['cluster_id'])}"

    df['status'] = df.apply(assign_full_status, axis=1)
    
    valid_statuses = sorted(
        [s for s in df['status'].unique() if s.startswith('Cluster')],
        key=lambda s: [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    )

    df_only_clusters = df[df['cluster_id'] >= 0].copy()
    cluster_ts = df_only_clusters.groupby(['status', 'tanggal'])['NDVI_smooth'].agg(['mean', 'std']).reset_index()

    fenologi_list = []
    for s in valid_statuses:
        sub = cluster_ts[cluster_ts['status'] == s]
        if sub.empty: continue
        
        peak_idx = sub['mean'].idxmax()
        peak_val = sub['mean'].max()
        peak_date = sub.loc[peak_idx, 'tanggal']
        min_val = sub['mean'].min()
        amplitudo = peak_val - min_val

        fenologi_list.append({
            'Status': s,
            'Jml Titik': df_only_clusters[df_only_clusters['status'] == s]['id_lokasi'].nunique(),
            'Puncak NDVI': peak_val,
            'Waktu Puncak': peak_date.date(),
            'Min NDVI': min_val,
            'Amplitudo': amplitudo,
            'Rata-rata NDVI': sub['mean'].mean(),
            'Rata-rata StdDev': sub['std'].mean()
        })

    df_fenologi = pd.DataFrame(fenologi_list).set_index('Status') if fenologi_list else pd.DataFrame()
    return df_fenologi, cluster_ts, valid_statuses

def _get_status_colors(valid_statuses):
    valid_palette = sns.color_palette('tab10', len(valid_statuses))
    return {s: valid_palette[i % len(valid_statuses)] for i, s in enumerate(valid_statuses)}

def plot_comparison(cluster_ts, valid_statuses, target_year):
    STATUS_COLORS = _get_status_colors(valid_statuses)
    y_min = (cluster_ts['mean'] - cluster_ts['std']).min() - 0.05
    y_max = (cluster_ts['mean'] + cluster_ts['std']).max() + 0.05

    fig, ax = plt.subplots(figsize=(14, 7))
    for s in valid_statuses:
        sub = cluster_ts[cluster_ts['status'] == s]
        color = STATUS_COLORS[s]
        ax.plot(sub['tanggal'], sub['mean'], label=s, color=color, linewidth=2.5)
        ax.fill_between(sub['tanggal'], sub['mean'] - sub['std'], sub['mean'] + sub['std'],
                        color=color, alpha=0.1)
    ax.set_title(f'Perbandingan Tren NDVI & Variabilitas Antar Cluster ({target_year})', fontsize=14, fontweight='bold')
    ax.set_ylabel('NDVI Smoothed')
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(title='Daftar Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.subplots_adjust(right=0.8)
    return fig

def plot_individual_clusters(cluster_ts, valid_statuses):
    n_cols = 2
    n_rows = (len(valid_statuses) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    y_min = (cluster_ts['mean'] - cluster_ts['std']).min() - 0.05
    y_max = (cluster_ts['mean'] + cluster_ts['std']).max() + 0.05
    STATUS_COLORS = _get_status_colors(valid_statuses)

    for i, s in enumerate(valid_statuses):
        sub = cluster_ts[cluster_ts['status'] == s]
        ax = axes[i]
        color = STATUS_COLORS[s]
        ax.plot(sub['tanggal'], sub['mean'], color=color, linewidth=2.5)
        ax.fill_between(sub['tanggal'], sub['mean'] - sub['std'], sub['mean'] + sub['std'],
                        color=color, alpha=0.15)
        ax.set_title(f'Karakteristik Detail: {s}', fontsize=12, fontweight='bold')
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.2, linestyle='--')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout(pad=3.5)
    return fig

def plot_spatial_map(df, nr, nc, valid_statuses):
    NOISE_COLOR = '#E74C3C'
    UNSELECTED_COLOR = '#D3D3D3'
    STATUS_COLORS = {'Noise': NOISE_COLOR}
    STATUS_COLORS.update(_get_status_colors(valid_statuses))

    status_int_map = {'Noise': 1}
    for i, s in enumerate(valid_statuses):
        status_int_map[s] = i + 2

    grid_matrix_full = np.full((nr, nc), 0)
    gmap = df[['grid_row', 'grid_col', 'status']].drop_duplicates()
    for _, row in gmap.iterrows():
        grid_matrix_full[int(row['grid_row']), int(row['grid_col'])] = status_int_map.get(row['status'], 0)

    valid_palette = sns.color_palette('tab10', len(valid_statuses))
    all_colors = [UNSELECTED_COLOR, NOISE_COLOR] + [mpl.colors.rgb2hex(c) for c in valid_palette[:len(valid_statuses)]]
    cmap_full = ListedColormap(all_colors)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.pcolormesh(np.arange(nc + 1), np.arange(nr + 1), grid_matrix_full,
                  cmap=cmap_full, edgecolors='white', linewidth=0.3, shading='auto')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title('Peta Sebaran Grid Lengkap', fontweight='bold', fontsize=14)

    legend_elements = [Patch(facecolor=NOISE_COLOR, edgecolor='white', label='Noise')] + \
                      [Patch(facecolor=STATUS_COLORS[s], edgecolor='white', label=s) for s in valid_statuses]
    ax.legend(handles=legend_elements, title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    plt.subplots_adjust(right=0.75)
    return fig