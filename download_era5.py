"""
Baixar preditores ERA5 para América do Sul
Variáveis: u,v @850/500 hPa, T2m, q850
"""
import cdsapi
import os

DATA_DIR = r"C:\Users\axnva\tup_data"
os.makedirs(DATA_DIR, exist_ok=True)

c = cdsapi.Client()

# 1. Pressure levels: u, v @850 e 500 hPa
print("Baixando u,v @850/500 hPa...")
c.retrieve(
    'reanalysis-era5-pressure-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': [
            'u_component_of_wind',
            'v_component_of_wind',
            'specific_humidity',
        ],
        'pressure_level': ['850', '500'],
        'year': [str(y) for y in range(1979, 2027)],
        'month': [f'{m:02d}' for m in range(1, 13)],
        'time': '00:00',
        'area': [12, -85, -55, -35],  # N, W, S, E
        'format': 'netcdf',
    },
    os.path.join(DATA_DIR, 'era5_pressure_levels.nc')
)
print("OK: era5_pressure_levels.nc")

# 2. Single levels: T2m
print("Baixando T2m...")
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': [
            '2m_temperature',
        ],
        'year': [str(y) for y in range(1979, 2027)],
        'month': [f'{m:02d}' for m in range(1, 13)],
        'time': '00:00',
        'area': [12, -85, -55, -35],  # N, W, S, E
        'format': 'netcdf',
    },
    os.path.join(DATA_DIR, 'era5_single_levels.nc')
)
print("OK: era5_single_levels.nc")

print("\nDownload completo!")
