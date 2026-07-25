"""
Tupã — baseline de previsão mensal de precipitação sobre a América do Sul
WorCAP 2026 Hackathon (ERA5).

Este modulo e um SCAFFOLD: roda ponta-a-ponta com dados SINTETICOS (so precisa de
numpy) e tem ganchos prontos para plugar ERA5 real via:
  - NetCDF/xarray (caminho principal: o dataset ERA5 fornecido no Kaggle), ou
  - cdsapi (download direto do Climate Data Store do Copernicus).

Contem as baselines OBRIGATORIAS que qualquer modelo nosso precisa bater:
  (0) Climatologia mensal  -> media historica por mes de calendario, por ponto.
  (1) Persistencia         -> mes atual como previsao do proximo.
E um ML de referencia (truque do "pool espacial" do Anochi 2021):
  (2) Ridge "pooled"       -> um unico modelo global, cada ponto de grade vira
                              uma amostra (contorna o N pequeno).

Metricas honestas (skill real, nao so acerto bruto):
  - Skill score (RMSE) vs climatologia   -> 0 = tao bom quanto a media historica.
  - Correlacao de anomalia (ACC)         -> mede so o sinal alem da climatologia.

Demo:
    python baseline.py
"""

import numpy as np


# ----------------------------------------------------------------------
# 1. Dados
# ----------------------------------------------------------------------
def make_synthetic_sa_precip(n_years=41, start_year=1983, seed=0):
    """Precipitacao mensal SINTETICA plausivel sobre a America do Sul.

    Inclui: nucleo sempre umido no norte (Amazônia/ITCZ), monsao com pico no
    verao austral (DJF), modulacao interanual tipo ENSO e ruido. So existe para
    exercitar o pipeline inteiro sem precisar do ERA5.
    """
    rng = np.random.default_rng(seed)
    lats = np.arange(12, -55.1, -2.5)          # 12N .. 55S
    lons = np.arange(-85, -34.9, 2.5)          # 85W .. 35W
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    nlat, nlon = LAT.shape

    T = 12 * n_years
    months = np.tile(np.arange(12), n_years)
    years = np.repeat(np.arange(start_year, start_year + n_years), 12)

    north_wet = np.exp(-((LAT - 0) / 12.0) ** 2)
    monsoon = np.exp(-((LAT + 18) / 14.0) ** 2) * np.exp(-((LON + 55) / 18.0) ** 2)
    enso_env = np.exp(-((LAT + 10) / 20.0) ** 2)   # envelope espacial do sinal ENSO

    prec = np.zeros((T, nlat, nlon), dtype=np.float32)
    for t in range(T):
        season = np.cos(2 * np.pi * (months[t] - 11) / 12.0)   # +1 em dezembro
        field = (3.5 * north_wet
                 + 6.0 * monsoon * (0.5 + 0.5 * season)
                 + 1.5 * enso_env * np.sin(t / 18.0))
        prec[t] = np.clip(field + rng.normal(0, 1.0, size=field.shape), 0, None)
    return prec, lats, lons, months, years


def load_era5_netcdf(path, precip_var="tp"):
    """Carrega ERA5 mensal de um NetCDF (formato tipico da competicao Kaggle).

    Requer: pip install xarray netCDF4
    Converte para mm/mes. ERA5 'tp' vem em metros acumulados -> *1000 e ajusta
    pelos dias do mes. ATENCAO: a definicao exata do alvo (total mensal vs media
    diaria, unidade) deve bater com o spec do Kaggle quando publicado (~14/09).
    """
    import xarray as xr  # noqa
    raise NotImplementedError(
        "Hook ERA5/NetCDF: preencher quando o dataset Kaggle estiver disponivel. "
        "Retornar (prec[T,lat,lon], lats, lons, months, years)."
    )


def load_era5_cds(years, area=(12, -35, -55, -85), api_key_path=None):
    """Baixa ERA5 mensal direto do Climate Data Store (Copernicus).

    Requer: pip install cdsapi  e  chave em ~/.cdsapirc (ou api_key_path).
    Secundario: o Kaggle provavelmente ja fornecera o ERA5 — use load_era5_netcdf.
    """
    try:
        import cdsapi  # noqa
    except ImportError as e:
        raise ImportError("Instale cdsapi: pip install cdsapi") from e
    raise NotImplementedError(
        "Hook ERA5/CDS: implementar o request (reanalysis-era5-single-levels, "
        "monthly) quando tiver a chave e o spec do Kaggle."
    )


# ----------------------------------------------------------------------
# 2. Baselines
# ----------------------------------------------------------------------
def monthly_climatology(prec, months, train_mask):
    """Media por mes de calendario em cada ponto de grade (calculada so no treino)."""
    clim = np.zeros((12, *prec.shape[1:]), dtype=np.float64)
    for m in range(12):
        sel = train_mask & (months == m)
        clim[m] = prec[sel].mean(axis=0)
    return clim


def forecast_climatology(clim, months):
    """Previsao da climatologia para cada passo (indexada pelo mes de calendario)."""
    return clim[months]


def forecast_persistence(prec, lead=1):
    """Persistencia: previsao do mes-alvo = observado em (alvo - lead).

    Sem vazamento de futuro: para prever o mes t, usa-se o observado em t-lead.
    """
    out = np.empty_like(prec, dtype=np.float64)
    out[lead:] = prec[:-lead]
    out[:lead] = prec[0:1]               # borda (nao avaliada no teste)
    return out


def ridge_pooled_forecast(prec, months, years, lats, lons,
                          train_years_end=2021, lead=1, lam=10.0):
    """Ridge por POOL ESPACIAL (forma fechada, so numpy).

    Um UNICO modelo global: cada ponto de grade vira uma linha (truque do Anochi
    2021 p/ contornar o N mensal pequeno).
    Features = [sin(mes), cos(mes), lat, lon, prec[t-lead], prec[t]] -> prec[t+lead].
    """
    nT, nlat, nlon = prec.shape
    P = nlat * nlon
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")

    src_t = np.arange(lead, nT - lead)     # passo "atual"
    tgt_t = src_t + lead                    # alvo previsto
    F = 6
    X = np.zeros((len(src_t), P, F), dtype=np.float64)
    for i, t in enumerate(src_t):
        m = months[t]
        X[i, :, 0] = np.sin(2 * np.pi * m / 12.0)
        X[i, :, 1] = np.cos(2 * np.pi * m / 12.0)
        X[i, :, 2] = LAT.ravel()
        X[i, :, 3] = LON.ravel()
        X[i, :, 4] = prec[t - lead].ravel()
        X[i, :, 5] = prec[t].ravel()
    X = X.reshape(len(src_t) * P, F)
    y = prec[tgt_t].reshape(len(tgt_t) * P)
    tgt_years = np.repeat(years[tgt_t], P)

    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    tr = tgt_years <= train_years_end
    A = Xb[tr].T @ Xb[tr] + lam * np.eye(F + 1)
    w = np.linalg.solve(A, Xb[tr].T @ y[tr])
    yhat = Xb @ w

    fc = np.full(prec.shape, np.nan, dtype=np.float64)
    for i, tt in enumerate(tgt_t):
        fc[tt] = yhat[i * P:(i + 1) * P].reshape(nlat, nlon)
    return fc


# ----------------------------------------------------------------------
# 3. Metricas honestas
# ----------------------------------------------------------------------
def rmse(pred, obs, mask):
    d = pred[mask] - obs[mask]
    return float(np.sqrt(np.mean(d ** 2)))


def skill_score(pred, obs, ref, mask):
    """1 - RMSE(pred)/RMSE(ref).  >0 significa melhor que a climatologia."""
    return 1.0 - rmse(pred, obs, mask) / rmse(ref, obs, mask)


def acc(pred, obs, clim_fc, mask):
    """Correlacao de anomalia: corr(pred - clim, obs - clim).
    Retorna nan se a anomalia prevista tiver variancia nula (ex.: climatologia)."""
    a = (pred[mask] - clim_fc[mask]).ravel()
    b = (obs[mask] - clim_fc[mask]).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spi(prec, months, train_mask):
    """SPI-1 (McKee) via ajuste gamma por (ponto, mes). Opcional (requer scipy)."""
    try:
        from scipy.stats import gamma, norm
    except ImportError:
        raise ImportError("SPI requer scipy: pip install scipy")
    spi_field = np.full_like(prec, np.nan, dtype=np.float64)
    for m in range(12):
        tr = train_mask & (months == m)
        if tr.sum() < 5:
            continue
        data = prec[tr]                          # (n_train, lat, lon)
        for it in range(prec.shape[1]):
            for jt in range(prec.shape[2]):
                x = data[:, it, jt]
                x = x[x > 0]
                if x.size < 5:
                    continue
                a, loc, scale = gamma.fit(x, floc=0)
                idx = np.where(months == m)[0]
                cdf = gamma.cdf(prec[idx, it, jt], a, loc=loc, scale=scale)
                spi_field[idx, it, jt] = norm.ppf(np.clip(cdf, 1e-3, 1 - 1e-3))
    return spi_field


# ----------------------------------------------------------------------
# 4. Visualizacao (opcional - so plota se houver matplotlib; ex.: Colab)
# ----------------------------------------------------------------------
def plot_demo(prec, lats, lons, months, clim, results, test_mask):
    """Plota climatologia, sinal interanual e comparativo de metricas.
    No Colab os graficos aparecem inline; sem matplotlib, apenas pula."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib ausente - graficos pulados; aparecem no Colab)")
        return

    LAT, LON = np.meshgrid(lats, lons, indexing="ij")

    # Fig 1: climatologia - verao (Dez) vs inverno (Jun)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, m, titulo in [(ax[0], 11, "Climatologia - Dez (verao)"),
                         (ax[1], 5, "Climatologia - Jun (inverno)")]:
        pcm = a.pcolormesh(LON, LAT, clim[m], cmap="YlGnBu", shading="auto")
        a.set_title(titulo); a.set_xlabel("lon"); a.set_ylabel("lat")
        plt.colorbar(pcm, ax=a)
    fig.tight_layout(); plt.show()

    # Fig 2: anomalia media do dominio (sinal interanual que o ridge explora)
    clim_fc = clim[months]
    serie = (prec - clim_fc).mean(axis=(1, 2))
    fig, ax = plt.subplots(figsize=(12, 3))
    n_test = int(test_mask.sum())
    ax.axvspan(len(serie) - n_test, len(serie), color="orange", alpha=0.2,
               label="teste 2022-23")
    ax.plot(serie, lw=1)
    ax.set_title("Anomalia de precipitacao (media do dominio) - sinal interanual")
    ax.set_xlabel("passo mensal"); ax.set_ylabel("mm/dia (sint.)"); ax.legend()
    fig.tight_layout(); plt.show()

    # Fig 3: RMSE e skill score por baseline
    nomes = list(results.keys())
    cores = ["#999999", "#dd8888", "#88bb88"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
    a1.bar(nomes, [results[n]["rmse"] for n in nomes], color=cores)
    a1.set_title("RMSE (menor = melhor)")
    a2.bar(nomes, [results[n]["skill"] for n in nomes], color=cores)
    a2.axhline(0, color="k", lw=0.8)
    a2.set_title("Skill vs climatologia (>0 = melhor)")
    for a in (a1, a2):
        a.tick_params(axis="x", rotation=15)
    fig.tight_layout(); plt.show()


# ----------------------------------------------------------------------
# 5. Demo
# ----------------------------------------------------------------------
def main():
    prec, lats, lons, months, years = make_synthetic_sa_precip()
    train_mask = years <= 2021
    test_mask = years >= 2022

    clim = monthly_climatology(prec, months, train_mask)
    clim_fc = forecast_climatology(clim, months)
    pers_fc = forecast_persistence(prec, lead=1)
    ridge_fc = ridge_pooled_forecast(prec, months, years, lats, lons)
    eval_mask = test_mask & np.isfinite(ridge_fc[:, 0, 0])

    print("== Tupã - baselines sobre 2022-2023 (dados sinteticos) ==")
    results = {}
    for nome, pred in [("climatologia", clim_fc),
                       ("persistencia", pers_fc),
                       ("ridge(pooled)", ridge_fc)]:
        r = rmse(pred, prec, eval_mask)
        s = skill_score(pred, prec, clim_fc, eval_mask)
        a = acc(pred, prec, clim_fc, eval_mask)
        results[nome] = {"rmse": r, "skill": s, "acc": a}
        print(f"{nome:14s}  RMSE={r:6.3f}  skill(vs clim)={s:+.3f}  ACC={a:+.3f}")

    plot_demo(prec, lats, lons, months, clim, results, test_mask)


if __name__ == "__main__":
    main()
