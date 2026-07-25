"""
Processar ERA5 e integrar no pipeline Tupã
"""
import numpy as np
import netCDF4 as nc
import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"C:\Users\axnva\tup_data")

def process_era5():
    # Carregar dados
    ds_pl = nc.Dataset(DATA_DIR / "era5_pressure_levels.nc")
    ds_sl = nc.Dataset(DATA_DIR / "era5_single_levels.nc")
    
    # Variáveis
    u = ds_pl.variables['u'][:]  # (570, 2, 269, 201)
    v = ds_pl.variables['v'][:]
    q = ds_pl.variables['q'][:]
    t2m = ds_sl.variables['t2m'][:]  # (570, 269, 201)
    
    lats = ds_pl.variables['latitude'][:]
    lons = ds_pl.variables['longitude'][:]
    times = nc.num2date(ds_pl.variables['valid_time'][:], ds_pl.variables['valid_time'].units)
    
    ds_pl.close()
    ds_sl.close()
    
    # Filtro América do Sul
    lat_mask = (lats >= -55) & (lats <= 12)
    lon_mask = (lons >= -85) & (lons <= -35)
    
    u_sa = u[:, :, lat_mask, :][:, :, :, lon_mask]
    v_sa = v[:, :, lat_mask, :][:, :, :, lon_mask]
    q_sa = q[:, :, lat_mask, :][:, :, :, lon_mask]
    t2m_sa = t2m[:, lat_mask, :][:, :, lon_mask]
    
    print(f"ERA5 SA: {u_sa.shape}")
    print(f"  u850: {u_sa[:, 0, :, :].shape}")
    print(f"  u500: {u_sa[:, 1, :, :].shape}")
    print(f"  t2m: {t2m_sa.shape}")
    
    # Salvar processado
    np.savez(DATA_DIR / "era5_sa.npz",
             u850=u_sa[:, 0, :, :],
             u500=u_sa[:, 1, :, :],
             v850=v_sa[:, 0, :, :],
             v500=v_sa[:, 1, :, :],
             q850=q_sa[:, 0, :, :],
             t2m=t2m_sa,
             lats=lats[lat_mask],
             lons=lons[lon_mask],
             dates=np.array([str(d)[:7] for d in times]))
    
    print("Salvo: era5_sa.npz")

if __name__ == "__main__":
    process_era5()
