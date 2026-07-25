"""
Processar ERA5 → grade do GPCP (agregação)
"""
import numpy as np
import netCDF4 as nc
from pathlib import Path

DATA_DIR = Path(r"C:\Users\axnva\tup_data")

def aggregate_era5_to_gpcp():
    # Carregar grade do GPCP
    gpcp = np.load(DATA_DIR / "gpcp_sa.npz", allow_pickle=True)
    gpcp_lats = gpcp['lats']
    gpcp_lons = gpcp['lons']
    
    # Carregar ERA5
    era5 = np.load(DATA_DIR / "era5_sa.npz", allow_pickle=True)
    era5_lats = era5['lats']
    era5_lons = era5['lons']
    
    n_time = era5['u850'].shape[0]
    n_gpcp_lat = len(gpcp_lats)
    n_gpcp_lon = len(gpcp_lons)
    
    print(f"GPCP grid: {n_gpcp_lat}x{n_gpcp_lon}")
    print(f"ERA5 grid: {len(era5_lats)}x{len(era5_lons)}")
    
    # Agregar ERA5 para grade do GPCP
    result = {}
    for var in ['u850', 'u500', 'v850', 'v500', 'q850', 't2m']:
        era5_var = era5[var]
        gpcp_var = np.zeros((n_time, n_gpcp_lat, n_gpcp_lon))
        
        for i, glat in enumerate(gpcp_lats):
            # Encontrar indices ERA5 mais proximos
            lat_idx = np.argmin(np.abs(era5_lats - glat))
            for j, glon in enumerate(gpcp_lons):
                lon_idx = np.argmin(np.abs(era5_lons - glon))
                # Media 3x3 ao redor
                lat_slice = slice(max(0, lat_idx-1), min(len(era5_lats), lat_idx+2))
                lon_slice = slice(max(0, lon_idx-1), min(len(era5_lons), lon_idx+2))
                gpcp_var[:, i, j] = np.mean(era5_var[:, lat_slice, lon_slice], axis=(1, 2))
        
        result[var] = gpcp_var
        print(f"  {var}: std={np.std(gpcp_var):.4g}")
    
    # Salvar
    np.savez(DATA_DIR / "era5_gpcp_grid.npz", **result,
             lats=gpcp_lats, lons=gpcp_lons)
    print("Salvo: era5_gpcp_grid.npz")

if __name__ == "__main__":
    aggregate_era5_to_gpcp()
