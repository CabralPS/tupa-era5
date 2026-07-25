"""
Tupã — Pipeline completo para o Hackathon WorCAP 2026
=====================================================
Previsão MENSAL de precipitação sobre a América do Sul.
  1. Dados (sintéticos, .npz real GPCP, ou .csv/.parquet do Kaggle)
  2. Baselines honestas (climatologia + persistência)
  3. Teleconexões via ÍNDICES PRONTOS (Niño3/3.4/4, TNA, TSA, AAO/SAM)
  4. Features sem vazamento (prever t+1 a partir de t e t-1)
  5. XGBoost com skill vs climatologia PER-CÉLULA (honesto)
  6. Salvamento

Rodar:
  - Sintético:        python tupã_pipeline.py
  - Dado real (GPCP): USE_SYNTHETIC=False; DATA_PATH='../tup_data/gpcp_sa.npz'
  - Colab:            %run tupã_pipeline.py
"""

import numpy as np
import pandas as pd
import xarray as xr  # noqa: F401
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
USE_SYNTHETIC = True
DATA_PATH = '../tup_data/gpcp_sa.npz'
TRAIN_YEARS = (1983, 2021)
TEST_YEARS = (2022, 2025)
LAT_RANGE = (-55, 12)
LON_RANGE = (-85, -35)


# =============================================================================
# 1. DADOS
# =============================================================================
def load_data():
    if USE_SYNTHETIC:
        return generate_synthetic_data()
    p = (DATA_PATH or '').lower()
    if p.endswith('.npz'):
        return load_npz(DATA_PATH)
    return load_era5_real(DATA_PATH)


def generate_synthetic_data(n_years=41, start_year=1983, seed=42):
    rng = np.random.default_rng(seed)
    lats = np.arange(12, -55.1, -2.5)
    lons = np.arange(-85, -34.9, 2.5)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    nlat, nlon = len(lats), len(lons)
    T = 12 * n_years
    months = np.tile(np.arange(1, 13), n_years)
    years = np.repeat(np.arange(start_year, start_year + n_years), 12)
    dates = pd.date_range(f'{start_year}-01', periods=T, freq='MS')
    north_wet = np.exp(-((LAT) / 12.0) ** 2)
    monsoon = np.exp(-((LAT + 18) / 14.0) ** 2) * np.exp(-((LON + 55) / 18.0) ** 2)
    enso_env = np.exp(-((LAT + 10) / 20.0) ** 2)
    prec = np.zeros((T, nlat, nlon), dtype=np.float64)
    for t in range(T):
        season = np.cos(2 * np.pi * (months[t] - 1 - 11) / 12.0)
        field = (3.5 * north_wet + 6.0 * monsoon * (0.5 + 0.5 * season)
                 + 1.5 * enso_env * np.sin(t / 18.0))
        prec[t] = np.clip(field + rng.normal(0, 1.0, size=field.shape), 0, None)
    return {'precipitation': prec, 'lats': lats, 'lons': lons,
            'months': months, 'years': years, 'dates': dates}


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    prec = np.ascontiguousarray(d['precipitation']).astype(np.float64)
    dates = pd.to_datetime(d['dates'])
    return {'precipitation': prec, 'lats': np.asarray(d['lats']).astype(np.float64),
            'lons': np.asarray(d['lons']).astype(np.float64),
            'months': dates.month.values.astype(int),
            'years': dates.year.values.astype(int), 'dates': dates}


def load_era5_real(path):
    """Carrega .csv/.parquet longo (time, lat, lon, precip). Para .nc use load_npz
    ou adapte ao spec do Kaggle (~15/09)."""
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        raise NotImplementedError("Para .nc converta p/ .npz/.parquet (spec Kaggle).")
    col_map = {'valid_time': 'time', 'latitude': 'lat', 'longitude': 'lon',
               'tp': 'precip', 'precipitation': 'precip'}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df = df[(df['lat'] >= LAT_RANGE[0]) & (df['lat'] <= LAT_RANGE[1]) &
            (df['lon'] >= LON_RANGE[0]) & (df['lon'] <= LON_RANGE[1])].copy()
    g = df.pivot_table(index='time', columns=['lat', 'lon'], values='precip')
    times = pd.to_datetime(g.index.values)
    lat_idx = np.array(sorted(g.columns.get_level_values('lat').unique()))
    lon_idx = np.array(sorted(g.columns.get_level_values('lon').unique()))
    precip = np.full((len(times), len(lat_idx), len(lon_idx)), np.nan)
    pla = {v: i for i, v in enumerate(lat_idx)}; plo = {v: i for i, v in enumerate(lon_idx)}
    for (la, lo), col in g.items():
        precip[:, pla[la], plo[lo]] = col.values
    return {'precipitation': precip, 'lats': lat_idx, 'lons': lon_idx,
            'months': times.month.values, 'years': times.year.values, 'dates': times}


def load_era5_cds(years, area=(12, -35, -55, -85), api_key_path=None):
    """Baixa preditores ERA5 mensais (u,v @850/500; T2m; q850) via CDS.
    Requer: pip install cdsapi  e  chave em ~/.cdsapirc. NÃO TESTADO sem a key —
    ajustar o request ao spec oficial do WorCAP quando publicado (~15/09)."""
    try:
        import cdsapi  # noqa
    except ImportError as e:
        raise ImportError("pip install cdsapi (e config ~/.cdsapirc)") from e
    raise NotImplementedError(
        "Implementar request 'reanalysis-era5-single-levels-monthly-means' "
        "+ 'reanalysis-era5-pressure-levels-monthly-means' p/ u/v/q/T2m sobre a "
        "bbox SA. Precisa da CDS key; alinhar ao spec WorCAP.")


# =============================================================================
# 2. TELECONEXÕES (índices PRONTOS globais)
# =============================================================================
def _load_monthly_index(url, dates):
    """Baixa índice mensal (formato 'ano + 12 valores', NOAA PSL / CPC) e alinha
    a `dates`. Retorna None se indisponível. Sentinelas (|v|>50) viram NaN."""
    import urllib.request
    try:
        raw = urllib.request.urlopen(url, timeout=15).read().decode()
    except Exception:
        return None
    table = {}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) != 13:
            continue
        try:
            yr = int(parts[0]); vals = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        if 1900 <= yr <= 2100:
            table[yr] = [np.nan if abs(v) > 50 else v for v in vals]
    out = np.full(len(dates), np.nan)
    for i, dt in enumerate(pd.to_datetime(dates)):
        if dt.year in table:
            out[i] = table[dt.year][dt.month - 1]
    return out


def _load_cpc_monthly(url, dates):
    """Índice mensal no formato CPC 'ano mes valor' (1 linha por mês). P/ AAO/SAM."""
    import urllib.request
    try:
        raw = urllib.request.urlopen(url, timeout=15).read().decode(errors='ignore')
    except Exception:
        return None
    table = {}
    for line in raw.splitlines():
        p = line.split()
        if len(p) != 3:
            continue
        try:
            yr, mo, v = int(p[0]), int(p[1]), float(p[2])
        except ValueError:
            continue
        if 1900 <= yr <= 2100 and 1 <= mo <= 12 and abs(v) < 50:
            table.setdefault(yr, {})[mo] = v
    out = np.full(len(dates), np.nan)
    for i, dt in enumerate(pd.to_datetime(dates)):
        if dt.year in table and dt.month in table[dt.year]:
            out[i] = table[dt.year][dt.month]
    return out


def load_teleconnections(dates):
    """Índices mensais autoritativos. ENSO/Pacífico/Atlântico via NOAA PSL
    (ano+12); SAM/AAO via CPC (ano/mês/valor). Falhas são descartadas."""
    psl = {
        'nino34': 'https://psl.noaa.gov/data/correlation/nina34.anom.data',
        'nino3':  'https://psl.noaa.gov/data/correlation/nina3.anom.data',
        'nino4':  'https://psl.noaa.gov/data/correlation/nina4.anom.data',
        'soi':    'https://psl.noaa.gov/data/correlation/soi.data',
        'pna':    'https://psl.noaa.gov/data/correlation/pna.data',
        'pdo':    'https://psl.noaa.gov/data/correlation/pdo.data',
        'amm':    'https://psl.noaa.gov/data/correlation/amm.data',
        'tni':    'https://psl.noaa.gov/data/correlation/tni.data',
        'nao':    'https://psl.noaa.gov/data/correlation/nao.data',
    }
    cpc = {
        'aao': 'https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/aao/monthly.aao.index.b79.current.ascii',
    }
    tele = {}
    for key, url in psl.items():
        s = _load_monthly_index(url, dates)
        if s is None or np.isnan(s).all():
            print(f"  AVISO: {key} indisponivel -> descartado."); continue
        s = np.where(np.isfinite(s), s, 0.0); tele[key] = s - s.mean()
        print(f"  {key}: ok (std={tele[key].std():.2f}).")
    for key, url in cpc.items():
        s = _load_cpc_monthly(url, dates)
        if s is None or np.isnan(s).all():
            print(f"  AVISO: {key} indisponivel -> descartado."); continue
        s = np.where(np.isfinite(s), s, 0.0); tele[key] = s - s.mean()
        print(f"  {key}: ok (std={tele[key].std():.2f}).")
    if not tele:
        tele = {'const': np.zeros(len(dates))}
    return tele


# =============================================================================
# 3. FEATURES (sem vazamento; features de teleconexão dinâmicas)
# =============================================================================
def create_features(prec, clim_fc, tele, months, years, lats, lons, lead=1):
    """Features e ALVO em ANOMALIA (prec - climatologia per-célula) — padrão S2S:
    o modelo não desperdiça capacidade reaprendendo a climatologia, e o sinal das
    teleconexões fica mais saliente."""
    T, nlat, nlon = prec.shape
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    sin_m = np.sin(2 * np.pi * (months - 1) / 12.0)
    cos_m = np.cos(2 * np.pi * (months - 1) / 12.0)
    src = np.arange(lead, T - lead)
    b = (len(src), nlat, nlon)
    feats = [np.broadcast_to(sin_m[src, None, None], b),
             np.broadcast_to(cos_m[src, None, None], b),
             np.broadcast_to(LAT[None, :, :], b),
             np.broadcast_to(LON[None, :, :], b),
             (prec[src] - clim_fc[src]),                  # anomalia atual
             (prec[src - lead] - clim_fc[src - lead])]    # anomalia de lag
    names = ['sin', 'cos', 'lat', 'lon', 'anom(t)', 'anom(t-1)']
    for key, series in tele.items():
        feats.append(np.broadcast_to(series[src, None, None], b)); names.append(key)
    X = np.stack(feats, axis=-1).reshape(-1, len(feats)).astype(np.float32)
    y = (prec[src + lead] - clim_fc[src + lead]).reshape(-1).astype(np.float32)  # ALVO = anomalia
    y_years = np.broadcast_to(years[src + lead, None, None], b).reshape(-1)
    return X, y, y_years, names


# =============================================================================
# 4. BASELINES
# =============================================================================
def monthly_climatology(prec, months, train_mask):
    clim = np.zeros((12, prec.shape[1], prec.shape[2]))
    for m in range(12):
        sel = train_mask & (months == m + 1)
        if sel.sum():
            clim[m] = prec[sel].mean(axis=0)
    return clim

def forecast_climatology(clim, months):
    return clim[months - 1]

def forecast_persistence(prec, lead=1):
    out = np.empty_like(prec, dtype=np.float64)
    out[lead:] = prec[:-lead]; out[:lead] = prec[0:1]
    return out


# =============================================================================
# 5. MÉTRICAS
# =============================================================================
def rmse(pred, obs, mask=None):
    p = pred[mask] if mask is not None else pred
    o = obs[mask] if mask is not None else obs
    v = np.isfinite(p) & np.isfinite(o)
    return 0.0 if v.sum() == 0 else float(np.sqrt(np.mean((p[v] - o[v]) ** 2)))

def skill_score(pred, obs, ref, mask=None):
    r = rmse(ref, obs, mask)
    return 0.0 if r == 0 else 1.0 - rmse(pred, obs, mask) / r


# =============================================================================
# 6. XGBOOST
# =============================================================================
def train_xgboost(X_train, y_train):
    try:
        import xgboost as xgb
    except ImportError:
        print("  AVISO: xgboost ausente -> pulando ML."); return None
    m = xgb.XGBRegressor(objective='reg:squarederror', max_depth=6, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                         n_estimators=300, random_state=42, n_jobs=-1)
    m.fit(X_train, y_train, verbose=False)
    return m


def _plot_skill_maps(acc, skill, lats, lons):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib ausente - mapa pulado; aparece no Colab)"); return
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, field, title, cmap, vmin, vmax in [
            (ax[0], acc, "ACC (corr. anomalia)", "RdBu_r", -0.6, 0.6),
            (ax[1], skill, "Skill vs climatologia", "RdYlGn", -0.5, 0.5)]:
        pcm = a.pcolormesh(LON, LAT, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        a.set_title(title); a.set_xlabel("lon"); a.set_ylabel("lat")
        plt.colorbar(pcm, ax=a)
    fig.tight_layout()
    try:
        fig.savefig('tupã_skill_maps.png', dpi=130, bbox_inches='tight')
        print("  mapa salvo: tupã_skill_maps.png")
    except Exception as e:
        print("  (nao salvou PNG: %s)" % e)
    plt.show()


def evaluate_regional_skill(model, X, y, y_years, lats, lons):
    """ACC e skill PER-CÉLULA no teste — revela skill regional diluída no RMSE médio."""
    nlat, nlon = len(lats), len(lons)
    n_src = X.shape[0] // (nlat * nlon)
    pred = model.predict(X).reshape(n_src, nlat, nlon)
    obs = y.reshape(n_src, nlat, nlon)
    yr = y_years.reshape(n_src, nlat, nlon)[:, 0, 0]
    te = (yr >= TEST_YEARS[0]) & (yr <= TEST_YEARS[1])
    p, o = pred[te], obs[te]
    acc = np.full((nlat, nlon), np.nan); skill = np.full((nlat, nlon), np.nan)
    for i in range(nlat):
        for j in range(nlon):
            a, b = p[:, i, j], o[:, i, j]
            if b.std() > 0:
                if a.std() > 0:
                    acc[i, j] = np.corrcoef(a, b)[0, 1]
                skill[i, j] = 1 - np.sqrt(np.mean((a - b) ** 2)) / b.std()
    print("  ACC medio: %+.3f | frac. ACC>0: %.0f%% | ACC max: %+.2f"
          % (np.nanmean(acc), np.nanmean(acc > 0) * 100, np.nanmax(acc)))
    print("  Skill medio per-celula: %+.3f | frac. skill>0: %.0f%%"
          % (np.nanmean(skill), np.nanmean(skill > 0) * 100))
    _plot_skill_maps(acc, skill, lats, lons)
    return acc, skill


# =============================================================================
# 7. MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("TUPA - Pipeline - WorCAP 2026  [%s]"
          % ("SINTETICO" if USE_SYNTHETIC else "REAL: " + str(DATA_PATH)))
    print("=" * 60)

    print("\n[1/6] Dados...")
    data = load_data()
    prec = data['precipitation']; lats = data['lats']; lons = data['lons']
    months = data['months']; years = data['years']
    print("  shape %s | anos %d-%d | precip mean %.2f max %.2f"
          % (prec.shape, years.min(), years.max(), np.nanmean(prec), np.nanmax(prec)))

    print("\n[2/6] Split temporal...")
    train_mask = (years >= TRAIN_YEARS[0]) & (years <= TRAIN_YEARS[1])
    test_mask = (years >= TEST_YEARS[0]) & (years <= TEST_YEARS[1])
    print("  treino %d meses | teste %d meses" % (int(train_mask.sum()), int(test_mask.sum())))

    print("\n[3/6] Baselines...")
    clim = monthly_climatology(prec, months, train_mask)
    clim_fc = forecast_climatology(clim, months)
    pers_fc = forecast_persistence(prec, lead=1)
    r_clim = rmse(clim_fc, prec, test_mask)
    r_pers = rmse(pers_fc, prec, test_mask)
    s_pers = skill_score(pers_fc, prec, clim_fc, test_mask)
    print("  %-12s RMSE %8.3f  skill    ---" % ("Climatologia", r_clim))
    print("  %-12s RMSE %8.3f  skill %+8.3f" % ("Persistencia", r_pers, s_pers))

    print("\n[4/6] Teleconexoes...")
    tele = load_teleconnections(data['dates'])

    print("\n[5/6] Features + XGBoost (alvo = anomalia)...")
    X, y, y_years, feat_names = create_features(prec, clim_fc, tele, months, years, lats, lons, lead=1)
    tr = y_years <= TRAIN_YEARS[1]
    te = y_years >= TEST_YEARS[0]
    print("  %d features | treino %d | teste %d" % (X.shape[1], int(tr.sum()), int(te.sum())))
    model = train_xgboost(X[tr], y[tr])
    if model is not None:
        y_pred = model.predict(X[te])
        r_xgb = rmse(y_pred, y[te])                          # RMSE na anomalia
        r_clim_anom = rmse(np.zeros_like(y[te]), y[te])      # climatologia = anomalia 0
        s_xgb = 1.0 - r_xgb / r_clim_anom if r_clim_anom > 0 else 0.0
        print("  %-12s anom-RMSE %7.3f | skill %+8.3f  (vs climatologia per-celula)" % ("XGBoost", r_xgb, s_xgb))
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            top = np.argsort(imp)[::-1][:5]
            print("  Top features: %s" % ", ".join(f"{feat_names[i]}({imp[i]:.2f})" for i in top))
        print("\n  Skill regional (ACC por ponto de grade):")
        evaluate_regional_skill(model, X, y, y_years, lats, lons)

    print("\n[6/6] Salvando tupã_results.npz ...")
    res = {'climatology_rmse': r_clim, 'persistence_rmse': r_pers, 'persistence_skill': s_pers}
    if model is not None:
        res['xgboost_rmse'] = r_xgb; res['xgboost_skill'] = s_xgb
    np.savez('tupã_results.npz', **res)
    print("=" * 60)


if __name__ == "__main__":
    main()
