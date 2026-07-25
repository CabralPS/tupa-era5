"""
Tupã — Pipeline com ERA5 predictors
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
USE_SYNTHETIC = False
DATA_PATH = r'C:\Users\axnva\tup_data\gpcp_sa.npz'
ERA5_PATH = r'C:\Users\axnva\tup_data\era5_gpcp_grid.npz'
TRAIN_YEARS = (1983, 2021)
TEST_YEARS = (2022, 2025)

# =============================================================================
# 1. DADOS
# =============================================================================
def load_data():
    d = np.load(DATA_PATH, allow_pickle=True)
    prec = np.ascontiguousarray(d['precipitation']).astype(np.float64)
    dates = pd.to_datetime(d['dates'])
    return {'precipitation': prec, 'lats': np.asarray(d['lats']).astype(np.float64),
            'lons': np.asarray(d['lons']).astype(np.float64),
            'months': dates.month.values.astype(int),
            'years': dates.year.values.astype(int), 'dates': dates}

def load_era5():
    d = np.load(ERA5_PATH, allow_pickle=True)
    return {k: d[k] for k in d.files if k in ['u850','u500','v850','v500','q850','t2m']}

# =============================================================================
# 2. TELECONEXÕES
# =============================================================================
def _load_monthly_index(url, dates):
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
    return tele

# =============================================================================
# 3. FEATURES (com ERA5)
# =============================================================================
def create_features(prec, clim_fc, tele, months, years, lats, lons, era5,
                    era5_clim=None, lead=1):
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
             (prec[src] - clim_fc[src]),
             (prec[src - lead] - clim_fc[src - lead])]
    names = ['sin', 'cos', 'lat', 'lon', 'anom(t)', 'anom(t-1)']
    
    # Ponderacao por latitude (cos(lat)) como feature para reduzir viés de
    # oversampling das altas latitudes no pool espacial. Incluida tambem como
    # peso de amostra no fit (ver train_xgboost).
    lat_w = np.cos(np.deg2rad(LAT))
    
    # ERA5 predictors (spatially varying) -- usar ANOMALIAS (vs climatologia mensal)
    # + lag t-lead, conforme template do PROJECT.md.
    for var_name in ['u850', 'u500', 'v850', 'v500', 'q850', 't2m']:
        if var_name not in era5:
            continue
        era5_var = era5[var_name]
        # Interpolar para grade do GPCP se necessário
        if era5_var.shape != (T, nlat, nlon) and era5_var.shape[1:] != (nlat, nlon):
            from scipy.ndimage import zoom
            zoom_factors = (1, nlat / era5_var.shape[1], nlon / era5_var.shape[2])
            era5_var = zoom(era5_var, zoom_factors, order=1)
        if era5_clim is not None and var_name in era5_clim:
            var_clim = era5_clim[var_name]                      # (12, nlat, nlon)
            anom_t = era5_var[src] - var_clim[months[src] - 1]
            anom_l = era5_var[src - lead] - var_clim[months[src - lead] - 1]
        else:
            # fallback degradado: usa valor bruto se nao houver climatologia
            anom_t = era5_var[src] - era5_var[src].mean(axis=0, keepdims=True)
            anom_l = era5_var[src - lead] - era5_var[src - lead].mean(axis=0, keepdims=True)
        feats.append(anom_t);   names.append(f'{var_name}_anom')
        feats.append(anom_l);   names.append(f'{var_name}_anom_l{lead}')
    
    # Teleconexões (ja centralizadas). Incluir lag para capturar precedencia.
    for key, series in tele.items():
        feats.append(np.broadcast_to(series[src, None, None], b))
        names.append(key)
        feats.append(np.broadcast_to(series[src - lead, None, None], b))
        names.append(f'{key}_l{lead}')
    
    X = np.stack(feats, axis=-1).reshape(-1, len(feats)).astype(np.float32)
    y = (prec[src + lead] - clim_fc[src + lead]).reshape(-1).astype(np.float32)
    y_years = np.broadcast_to(years[src + lead, None, None], b).reshape(-1)
    # Peso por lat para cada amostra (copia cos(lat) do ponto). Usado no fit.
    lat_w_flat = np.broadcast_to(lat_w[None, :, :], b).reshape(-1).astype(np.float32)
    return X, y, y_years, names, lat_w_flat, len(src)


def era5_monthly_climatology(era5, months, train_mask):
    """Climatologia mensal (12, nlat, nlon) por variavel ERA5, so no treino."""
    clims = {}
    for k, v in era5.items():
        if v.ndim != 3:
            continue
        clim = np.zeros((12, *v.shape[1:]), dtype=np.float64)
        for m in range(12):
            sel = train_mask & (months == m + 1)
            if sel.sum():
                clim[m] = v[sel].mean(axis=0)
        clims[k] = clim
    return clims

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
def train_xgboost(X_train, y_train, X_val=None, y_val=None,
                 sample_weight=None, feat_names=None):
    try:
        import xgboost as xgb
    except ImportError:
        print("  AVISO: xgboost ausente -> pulando ML."); return None
    params = dict(objective='reg:squarederror', max_depth=6,
                  learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=10,
                  reg_lambda=2.0, n_estimators=800, random_state=42, n_jobs=-1)
    if X_val is not None and y_val is not None:
        params['early_stopping_rounds'] = 40
    m = xgb.XGBRegressor(**params)
    fit_kwargs = {'verbose': False}
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight
    if X_val is not None and y_val is not None:
        fit_kwargs['eval_set'] = [(X_val, y_val)]
    m.fit(X_train, y_train, **fit_kwargs)
    return m

def _plot_skill_maps(acc, skill, lats, lons, save_path='figures/tupa_skill_maps.png'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib ausente - mapa pulado)"); return
    import os
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, field, title, cmap, vmin, vmax in [
            (ax[0], acc, "ACC (corr. anomalia)", "RdBu_r", -0.6, 0.6),
            (ax[1], skill, "Skill vs climatologia", "RdYlGn", -0.5, 0.5)]:
        pcm = a.pcolormesh(LON, LAT, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        a.set_title(title); a.set_xlabel("lon"); a.set_ylabel("lat")
        plt.colorbar(pcm, ax=a)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print("  Mapa salvo: %s" % save_path)
    plt.show()

def evaluate_regional_skill(model, X, y, y_years, lats, lons):
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
    print("TUPA - Pipeline com ERA5 - WorCAP 2026")
    print("=" * 60)

    print("\n[1/6] Dados...")
    data = load_data()
    prec = data['precipitation']; lats = data['lats']; lons = data['lons']
    months = data['months']; years = data['years']
    print("  GPCP: %s | %d-%d" % (prec.shape, years.min(), years.max()))

    print("\n[2/6] ERA5 predictors...")
    era5 = load_era5()
    era5_vars = {k: v for k, v in era5.items() if k in ['u850','u500','v850','v500','q850','t2m']}
    print("  Variaveis: %s" % list(era5_vars.keys()))
    for k, v in era5_vars.items():
        print("    %s: %s, std=%.4g" % (k, v.shape, np.std(v)))

    print("\n[3/6] Split temporal...")
    train_mask = (years >= TRAIN_YEARS[0]) & (years <= TRAIN_YEARS[1])
    test_mask = (years >= TEST_YEARS[0]) & (years <= TEST_YEARS[1])
    val_mask = (years >= 2017) & (years <= 2021)
    # treino p/ features exclui a janela de validacao (early-stop sem vazamento)
    train_feat_mask = (years >= TRAIN_YEARS[0]) & (years <= 2016)
    print("  treino %d meses | val %d meses | teste %d meses"
          % (int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum())))

    print("\n[4/6] Baselines...")
    clim = monthly_climatology(prec, months, train_mask)
    clim_fc = forecast_climatology(clim, months)
    pers_fc = forecast_persistence(prec, lead=1)
    r_clim = rmse(clim_fc, prec, test_mask)
    r_pers = rmse(pers_fc, prec, test_mask)
    s_pers = skill_score(pers_fc, prec, clim_fc, test_mask)
    print("  Climatologia RMSE: %.3f" % r_clim)
    print("  Persistencia RMSE: %.3f | skill: %+.3f" % (r_pers, s_pers))

    print("\n[4b/6] Climatologia dos preditores ERA5...")
    era5_clim = era5_monthly_climatology(era5_vars, months, train_mask)
    print("  clim por variavel: %s" % list(era5_clim.keys()))

    print("\n[5/6] Teleconexoes...")
    tele = load_teleconnections(data['dates'])

    print("\n[6/6] Features + XGBoost (com ERA5 anomalias + lags)...")
    LEAD = 1
    X, y, y_years, feat_names, lat_w_flat, n_src = create_features(
        prec, clim_fc, tele, months, years, lats, lons, era5_vars,
        era5_clim=era5_clim, lead=LEAD)
    tr = y_years <= 2016
    va = (y_years >= 2017) & (y_years <= 2021)
    te = (y_years >= TEST_YEARS[0]) & (y_years <= TEST_YEARS[1])
    print("  %d features" % X.shape[1])
    print("  treino %d | val %d | teste %d" % (int(tr.sum()), int(va.sum()), int(te.sum())))

    model = train_xgboost(X[tr], y[tr], X[va], y[va],
                          sample_weight=lat_w_flat[tr], feat_names=feat_names)
    if model is not None:
        nlat, nlon = len(lats), len(lons)
        P = nlat * nlon
        yr_pool = y_years.reshape(-1, nlat, nlon)[:, 0, 0]
        te_step_pool = (yr_pool >= TEST_YEARS[0]) & (yr_pool <= TEST_YEARS[1])
        y_pred = model.predict(X[te])
        r_xgb = rmse(y_pred, y[te])
        r_clim_anom = rmse(np.zeros_like(y[te]), y[te])
        s_xgb = 1.0 - r_xgb / r_clim_anom if r_clim_anom > 0 else 0.0
        # Reconstrução total. Cada "linha" do pool => alvo = src+lead. As
        # linhas do pool estão empilhadas por (passo, lat, lon), logo o idx
        # de passo alvo de cada linha é o reshape[-1, P] por coluna.
        te_passos = np.where(te_step_pool)[0]                  # idx no pool (por passo fonte)
        alvo_steps = te_passos + 2 * LEAD                      # alvo = src + lead = i + 2*lead
        clim_te = clim_fc[alvo_steps]                          # (n_te, lat, lon)
        pred_anom_te = y_pred.reshape(-1, nlat, nlon)
        pred_total = clim_te + pred_anom_te
        obs_total = prec[alvo_steps]
        # RMSE na escala total (todos os pontos, todas as celulas).
        def _rmse_flat(a, b):
            d = a - b
            return float(np.sqrt(np.mean(d ** 2)))
        r_clim_tot = _rmse_flat(clim_te, obs_total)
        r_xgb_tot = _rmse_flat(pred_total, obs_total)
        s_xgb_tot = 1.0 - r_xgb_tot / r_clim_tot if r_clim_tot > 0 else 0.0
        print("\n  === RESULTADOS COM ERA5 (anomalias + lags) ===")
        print("  Anomalia  | XGBoost RMSE: %.3f | skill: %+.3f" % (r_xgb, s_xgb))
        print("  Total     | XGBoost RMSE: %.3f | skill vs clim: %+.3f (clim RMSE=%.3f)"
              % (r_xgb_tot, s_xgb_tot, r_clim_tot))
        if hasattr(model, 'best_iteration'):
            print("  best_iteration: %s" % getattr(model, 'best_iteration', '?'))
        
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            top = np.argsort(imp)[::-1][:12]
            print("  Top 12 features:")
            for i in top:
                print("    %s: %.4f" % (feat_names[i], imp[i]))
        
        print("\n  Skill regional:")
        evaluate_regional_skill(model, X, y, y_years, lats, lons)

    print("\nSalvando resultados...")
    res = {'climatology_rmse': r_clim, 'persistence_rmse': r_pers,
           'persistence_skill': s_pers}
    if model is not None:
        res['xgboost_rmse'] = r_xgb; res['xgboost_skill'] = s_xgb
        res['xgboost_rmse_total'] = r_xgb_tot; res['xgboost_skill_total'] = s_xgb_tot
    np.savez('tupa_era5_results.npz', **res)
    print("=" * 60)

if __name__ == "__main__":
    main()
