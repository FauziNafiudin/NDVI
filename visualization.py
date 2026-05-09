import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def plot_spatial_grid(df, nr, nc, status_colors, valid_statuses, target_year):
    """Membuat peta grid spasial yang bersih dan profesional."""
    status_int_map = {'Unselected': 0, 'Noise': 1}
    for i, s in enumerate(valid_statuses):
        status_int_map[s] = i + 2
        
    grid_matrix = np.full((nr, nc), np.nan)
    gmap = df[['grid_row', 'grid_col', 'status']].drop_duplicates()
    grid_matrix[gmap['grid_row'].astype(int), gmap['grid_col'].astype(int)] = gmap['status'].map(status_int_map).values
    
    all_colors = ['#D3D3D3', '#E74C3C'] + [mpl.colors.rgb2hex(status_colors[s]) for s in valid_statuses]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pcolormesh(np.arange(nc + 1), np.arange(nr + 1), grid_matrix, 
                  cmap=ListedColormap(all_colors), edgecolors='white', linewidth=0.3)
    
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(f'Peta Sebaran Lokasi per Cluster ({target_year})', fontweight='bold')
    
    legend_els = [Patch(facecolor='#E74C3C', label='Noise')] + \
                 [Patch(facecolor=status_colors[s], label=s) for s in valid_statuses]
    ax.legend(handles=legend_els, title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.8)
    return fig

def plot_interactive_trends(df):
    """Grafik tren NDVI interaktif yang bisa di-zoom."""
    fig = px.line(df, x='tanggal', y='NDVI_smooth', color='status', 
                 line_group='id_lokasi', hover_name='id_lokasi',
                 title="Eksplorasi Tren NDVI (Interaktif)")
    return fig
