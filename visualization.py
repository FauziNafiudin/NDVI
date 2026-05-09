import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from scipy.ndimage import zoom
import io
from PIL import Image
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
    if df_only_clusters.empty:
        return pd.DataFrame(), pd.DataFrame(), []

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

def get_background_image(df, nr, nc):
    fig, ax = plt.subplots(figsize=(8, 8))
    mask = np.zeros((nr, nc))
    if not df.empty:
        valid_rows = df['grid_row'].values
        valid_cols = df['grid_col'].values
        mask_bounds = (valid_rows >= 0) & (valid_rows < nr) & (valid_cols >= 0) & (valid_cols < nc)
        mask[valid_rows[mask_bounds], valid_cols[mask_bounds]] = 1
    cmap = ListedColormap(['#2C3E50', '#FFFFFF'])
    ax.pcolormesh(np.arange(nc + 1), np.arange(nr + 1), mask,
                  cmap=cmap, edgecolors='#CCCCCC', linewidth=0.5, shading='auto')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def show_manual_selection_tool(df, nr, nc):
    st.markdown("### 🎨 Alat Seleksi Manual")
    st.markdown("Gunakan mouse untuk **menggambar (mewarnai)** area grid yang ingin dianalisis (garis oranye).")
    bg_image = get_background_image(df, nr, nc)
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0)", 
        stroke_width=20, 
        stroke_color="#FFA500", 
        background_image=bg_image,
        update_streamlit=True,
        height=600,
        width=600, 
        drawing_mode="freedraw",
        key="draw_canvas_selection",
    )
    if st.button("✅ Konfirmasi & Filter Data", type="primary"):
        if canvas_result.image_data is not None:
            img = canvas_result.image_data
            r_channel = img[:, :, 0].astype(float)
            g_channel = img[:, :, 1].astype(float)
            b_channel = img[:, :, 2].astype(float)
            stroke_mask = (r_channel >= 200) & (g_channel >= 100) & (g_channel <= 190) & (b_channel <= 50)
            h, w = stroke_mask.shape
            if h > 0 and w > 0:
                zoom_y = nr / h
                zoom_x = nc / w
                try:
                    resized_mask = zoom(stroke_mask.astype(float), (zoom_y, zoom_x), order=1) > 0.5
                    rows, cols = np.where(resized_mask)
                    selected_coords = set(zip(rows, cols))
                    df['coord_tuple'] = list(zip(df['grid_row'], df['grid_col']))
                    filtered_df = df[df['coord_tuple'].isin(selected_coords)].copy()
                    filtered_df.drop(columns=['coord_tuple'], inplace=True)
                    if not filtered_df.empty:
                        st.session_state['selected_df'] = filtered_df
                        st.success(f"✅ Berhasil menyeleksi {filtered_df['id_lokasi'].nunique()} lokasi unik.")
                        st.rerun()
                    else:
                        st.warning("Tidak ada data valid di area yang digambar.")
                except Exception as e:
                    st.error(f"Error memproses gambar: {e}")
            else:
                st.warning("Silakan gambar area terlebih dahulu.")
        else:
            st.warning("Silakan gambar area terlebih dahulu.")

def plot_comparison(cluster_ts, valid_statuses, target_year):
    STATUS_COLORS = {s: sns.color_palette('tab10', len(valid_statuses))[i % len(valid_statuses)] for i, s in enumerate(valid_statuses)}
    y_min = (cluster_ts['mean'] - cluster_ts['std']).min() - 0.05
    y_max = (cluster_ts['mean'] + cluster_ts['std']).max() + 0.05
    fig, ax = plt.subplots(figsize=(14, 7))
    for s in valid_statuses:
        sub = cluster_ts[cluster_ts['status'] == s]
        color = STATUS_COLORS[s]
        ax.plot(sub['tanggal'], sub['mean'], label=s, color=color, linewidth=2.5)
        ax.fill_between(sub['tanggal'], sub['mean'] - sub['std'], sub['mean'] + sub['std'], color=color, alpha=0.1)
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
    STATUS_COLORS = {s: sns.color_palette('tab10', len(valid_statuses))[i % len(valid_statuses)] for i, s in enumerate(valid_statuses)}
    y_min = (cluster_ts['mean'] - cluster_ts['std']).min() - 0.05
    y_max = (cluster_ts['mean'] + cluster_ts['std']).max() + 0.05
    for i, s in enumerate(valid_statuses):
        sub = cluster_ts[cluster_ts['status'] == s]
        ax = axes[i]
        color = STATUS_COLORS[s]
        ax.plot(sub['tanggal'], sub['mean'], color=color, linewidth=2.5)
        ax.fill_between(sub['tanggal'], sub['mean'] - sub['std'], sub['mean'] + sub['std'], color=color, alpha=0.15)
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
    STATUS_COLORS.update({s: sns.color_palette('tab10', len(valid_statuses))[i % len(valid_statuses)] for i, s in enumerate(valid_statuses)})
    status_int_map = {'Noise': 1}
    for i, s in enumerate(valid_statuses): status_int_map[s] = i + 2
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