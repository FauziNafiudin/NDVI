import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_base_data(file_path):
    """Memuat data NDVI dari file CSV yang diunggah."""
    try:
        df = pd.read_csv(file_path)
        # Pastikan kolom tanggal terbaca sebagai datetime
        if 'tanggal' in df.columns:
            df['tanggal'] = pd.to_datetime(df['tanggal'])
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return None

def filter_by_year(df, year):
    """Menyaring data berdasarkan tahun yang dipilih."""
    return df[df['tanggal'].dt.year == year].copy()
