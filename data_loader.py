import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import streamlit as st

def load_and_preprocess_data(file_2023, file_2024, window_size=31, poly_order=2):
    """Membaca file upload, menggabungkan, membuat grid, dan smoothing."""
    df_2023 = pd.read_csv(file_2023, parse_dates=['tanggal'])
    df_2024 = pd.read_csv(file_2024, parse_dates=['tanggal'])

    df_2023['tahun'] = '2023'
    df_2024['tahun'] = '2024'
    df_combined = pd.concat([df_2023, df_2024], ignore_index=True)

    # Clean & Sort
    df_clean = (df_combined
                .drop_duplicates(subset=['lat_y', 'lon_x', 'tanggal'], keep='first')
                .sort_values(['lat_y', 'lon_x', 'tanggal'])
                .reset_index(drop=True).copy())

    # ID Lokasi
    df_clean['id_lokasi'] = 'LOC_' + df_clean.groupby(['lat_y', 'lon_x']).ngroup().astype(str).str.zfill(4)
    df_clean['cluster_id'] = -1

    # Grid Creation
    DECIMALS = 5
    df_clean['lat_grid'] = df_clean['lat_y'].round(DECIMALS)
    df_clean['lon_grid'] = df_clean['lon_x'].round(DECIMALS)
    unique_lats = np.sort(df_clean['lat_grid'].unique())[::-1]
    unique_lons = np.sort(df_clean['lon_grid'].unique())
    lat_map = {v: i for i, v in enumerate(unique_lats)}
    lon_map = {v: i for i, v in enumerate(unique_lons)}
    
    df_clean['grid_row'] = df_clean['lat_grid'].map(lat_map).astype('int16')
    df_clean['grid_col'] = df_clean['lon_grid'].map(lon_map).astype('int16')
    n_rows, n_cols = len(unique_lats), len(unique_lons)

    # Batch Interpolation & Smoothing
    results = []
    for loc, group in df_clean.groupby('id_lokasi'):
        series = group.set_index('tanggal')['NDVI']
        daily = series.resample('D').interpolate(method='linear').reset_index()
        
        if len(daily) >= window_size:
            daily['NDVI_smooth'] = savgol_filter(
                daily['NDVI'].values, window_length=window_size, polyorder=poly_order
            ).astype('float32')
        else:
            daily['NDVI_smooth'] = daily['NDVI'].astype('float32')

        daily['id_lokasi'] = loc
        daily['lat_y'] = group['lat_y'].iloc[0]
        daily['lon_x'] = group['lon_x'].iloc[0]
        daily['tahun'] = group['tahun'].iloc[0]
        daily['grid_row'] = group['grid_row'].iloc[0]
        daily['grid_col'] = group['grid_col'].iloc[0]
        results.append(daily[['id_lokasi', 'tanggal', 'NDVI', 'NDVI_smooth', 'lat_y', 'lon_x', 'tahun', 'grid_row', 'grid_col']])

    df_final = pd.concat(results, ignore_index=True)
    return df_final, n_rows, n_cols