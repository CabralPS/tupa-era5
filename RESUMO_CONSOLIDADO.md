# Tupã — Resumo Consolidado (Para Colab)

## Resultados no GPCP Real (1979–2026)

| Métrica | Climatologia | Persistência | XGBoost |
|---------|-------------|-------------|---------|
| RMSE (mm/mês) | 45.22 | 66.58 | 45.05 |
| Skill vs Clim | --- | -0.472 | **+0.017** |
| ACC médio | 0.000 | --- | **+0.104** |
| ACC > 0 (%) | 0% | --- | **69%** |
| ACC máximo | 0.00 | --- | **+0.63** |

## Teleconexões Implementadas (Índices Reais)

| Índice | Fonte | Std |
|--------|-------|-----|
| Niño 3.4 | NOAA PSL | 0.87 |
| Niño 3 | NOAA PSL | 0.90 |
| Niño 4 | NOAA PSL | 0.66 |
| SOI | NOAA PSL | 1.58 |
| PNA | NOAA PSL | 1.01 |
| PDO | NOAA PSL | 1.43 |
| AMM | NOAA PSL | 2.66 |
| TNI | NOAA PSL | 1.31 |
| NAO | NOAA PSL | 1.05 |
| AAO/SAM | CPC | 0.98 |

## Top 5 Features (Importância XGBoost)

1. Niño 3 (0.08)
2. Niño 4 (0.07)
3. Niño 3.4 (0.07)
4. TNI (0.07)
5. Anomalia(t) (0.07)

## Interpretação dos Resultados

### O que funciona:
- **ACC positivo em 69% dos pontos** — o modelo captura sinal de anomalia na maioria da grade
- **Niño 3 domina** — ENSO é o driver principal da previsibilidade mensal
- **Skill geral positivo (+0.017)** — pequeno mas real, primeiro resultado honesto

### Limitações:
- **Skill per-célula baixo (-0.136)** — muitos pontos com skill negativo
- **RMSE ≈ climatologia** — o modelo basicamente "aprende a climatologia"
- **Dados limitados** — GPCP apenas, sem preditores atmosféricos ERA5

### O que falta (para Top-3):
- **ERA5 predictors** — u,v @850/500 hPa, T2m, q850 (magnitude do sinal)
- **CDS key** — precisa de verificação de email
- **Ponderação por latitude** — corrigir oversampling de altas latitudes

## Arquivos

```
tupã-era5/
├── tupã_pipeline.py      # Pipeline completo (rodar no Colab)
├── tupã_results.npz      # Resultados numéricos
├── tupã_skill_maps.png   # Mapas ACC + Skill
├── baseline.py           # Baseline standalone
├── testes_tupa.py        # Suite de testes (7/7)
├── requirements.txt      # Dependências
├── PROJECT.md            # Memória do projeto
├── figures/              # Figuras para apresentação
│   └── tupã_skill_maps.png
└── RESUMO_CONSOLIDADO.md # Este arquivo
```

## Como Rodar no Colab

```python
# 1. Setup
!pip install xgboost netCDF4
import os, urllib.request, numpy as np, netCDF4 as nc
os.makedirs('/content/tup_data', exist_ok=True)

# 2. Baixar GPCP do NOAA
!wget -q --tries=3 -O /content/tup_data/precip.mon.mean.nc \
  'https://psl.noaa.gov/thredds/fileServer/Datasets/gpcp/precip.mon.mean.nc'

# 3. Extrair America do Sul e salvar .npz
ds = nc.Dataset('/content/tup_data/precip.mon.mean.nc')
precip = ds.variables['precip'][:]
lats = ds.variables['lat'][:]
lons = ds.variables['lon'][:]
times = nc.num2date(ds.variables['time'][:], ds.variables['time'].units)
ds.close()
lat_m = (lats >= -55) & (lats <= 12)
lon_m = (lons >= -85) & (lons <= -35)
np.savez('/content/tup_data/gpcp_sa.npz',
         precipitation=precip[:, lat_m, :][:, :, lon_m],
         lats=lats[lat_m], lons=lons[lon_m],
         years=np.array([d.year for d in times]),
         months=np.array([d.month for d in times]),
         dates=np.array([str(d)[:7] for d in times]))
print('GPCP OK')

# 4. Clonar repo e rodar
!git clone https://github.com/CabralPS/tupa-era5.git
%cd tupa-era5
%run tupa_pipeline.py

# 5. Ver mapas
from IPython.display import Image
Image('figures/tupa_skill_maps.png')
```

## Próximos Passos (antes do Top-3)

1. **CDS key** → Baixar ERA5 pressure-level predictors
2. **Ponderação latitude** → Corrigir oversampling
3. **Ensemble** → Média de múltiplos modelos
4. **Feature engineering** → Anomalias de TMI/TRMM
