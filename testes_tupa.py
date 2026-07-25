"""
Tupã — Script de testes
=======================

Execute no Colab:
  %run testes_tupa.py

Ou localmente:
  python testes_tupa.py
"""

import numpy as np
import pandas as pd
import sys
import os


def _gpcp():
    """Caminho do GPCP-SA: busca em locais comuns (Windows local + Colab)."""
    for c in (os.environ.get('GPCP_SA_NPZ'), 'gpcp_sa.npz',
              '../tup_data/gpcp_sa.npz', 'tup_data/gpcp_sa.npz',
              'C:/Users/axnva/tup_data/gpcp_sa.npz'):
        if c and os.path.exists(c):
            return c
    return 'gpcp_sa.npz'


# =============================================================================
# Testes automatizados
# =============================================================================
def test_load_gpcp():
    """Testa carregamento do GPCP."""
    print("[TESTE 1] Carregando GPCP...")
    try:
        data = np.load(_gpcp())
        prec = data['precipitation']
        lats = data['lats']
        lons = data['lons']
        dates = data['dates']
        
        T, nlat, nlon = prec.shape
        assert T > 500, "Deveria ter >500 meses"
        assert nlat > 20, "Deveria ter >20 lats"
        assert nlon > 15, "Deveria ter >15 lons"
        assert lats.min() < -50, "Lat min deveria ser < -50"
        assert lats.max() > 10, "Lat max deveria ser > 10"
        
        print("  OK: Shape %s, Periodo %s a %s" % (
            str(prec.shape),
            str(dates[0])[:7],
            str(dates[-1])[:7]
        ))
        return True
    except Exception as e:
        print("  FALHOU: %s" % str(e))
        return False


def test_climatology():
    """Testa cálculo da climatologia."""
    print("[TESTE 2] Calculando climatologia...")
    try:
        data = np.load(_gpcp())
        prec = data['precipitation']
        months = np.array([pd.Timestamp(d).month for d in data['dates']])
        years = np.array([pd.Timestamp(d).year for d in data['dates']])
        
        train_mask = years <= 2021
        nlat, nlon = prec.shape[1], prec.shape[2]
        
        clim = np.zeros((12, nlat, nlon), dtype=np.float64)
        for m in range(12):
            sel = train_mask & (months == m + 1)
            clim[m] = prec[sel].mean(axis=0)
        
        # Verificar
        assert clim.shape == (12, nlat, nlon), "Shape errado"
        assert np.all(clim > 0), "Climatologia deveria ser > 0"
        assert clim.mean() > 50, "Media deveria ser > 50 mm/mes"
        assert clim.mean() < 200, "Media deveria ser < 200 mm/mes"
        
        print("  OK: Climatologia media = %.1f mm/mes" % clim.mean())
        return True
    except Exception as e:
        print("  FALHOU: %s" % str(e))
        return False


def test_persistence():
    """Testa persistência."""
    print("[TESTE 3] Calculando persistência...")
    try:
        data = np.load(_gpcp())
        prec = data['precipitation']
        
        pers = np.empty_like(prec, dtype=np.float64)
        pers[1:] = prec[:-1]
        pers[0] = prec[0]
        
        # Verificar
        assert pers.shape == prec.shape, "Shape errado"
        assert np.allclose(pers[1:], prec[:-1]), "Persistência incorreta"
        
        print("  OK: Persistência calculada")
        return True
    except Exception as e:
        print("  FALHOU: %s" % str(e))
        return False


def test_rmse():
    """Testa cálculo de RMSE."""
    print("[TESTE 4] Testando RMSE...")
    try:
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        obs = np.array([1.1, 2.1, 3.1, 4.1])
        
        rmse_val = float(np.sqrt(np.mean((pred - obs) ** 2)))
        
        assert abs(rmse_val - 0.1) < 0.01, "RMSE deveria ser ~0.1"
        
        print("  OK: RMSE = %.4f (esperado ~0.1)" % rmse_val)
        return True
    except Exception as e:
        print("  FALHOU: %s" % str(e))
        return False


def test_skill_score():
    """Testa skill score."""
    print("[TESTE 5] Testando skill score...")
    try:
        # Predição perfeita
        pred_perfect = np.array([1.0, 2.0, 3.0])
        obs = np.array([1.0, 2.0, 3.0])
        ref = np.array([0.5, 2.5, 2.5])  # climatologia
        
        r_pred = np.sqrt(np.mean((pred_perfect - obs) ** 2))
        r_ref = np.sqrt(np.mean((ref - obs) ** 2))
        skill = 1.0 - r_pred / r_ref
        
        assert skill == 1.0, "Skill perfeito deveria ser 1.0"
        
        # Predição igual à referência
        pred_ref = ref.copy()
        r_pred2 = np.sqrt(np.mean((pred_ref - obs) ** 2))
        skill2 = 1.0 - r_pred2 / r_ref
        
        assert abs(skill2) < 0.01, "Skill igual à ref deveria ser ~0"
        
        print("  OK: Skill perfeito = %.2f, Skill ref = %.2f" % (skill, skill2))
        return True
    except Exception as e:
        print("  FALHOU: %s" % str(e))
        return False


def test_baseline_values():
    """Verifica se os valores do baseline batem com o esperado."""
    print("[TESTE 6] Verificando valores do baseline...")
    try:
        data = np.load(_gpcp())
        prec = data['precipitation']
        months = np.array([pd.Timestamp(d).month for d in data['dates']])
        years = np.array([pd.Timestamp(d).year for d in data['dates']])
        
        train_mask = years <= 2021
        test_mask = years >= 2022
        nlat, nlon = prec.shape[1], prec.shape[2]
        
        # Climatologia
        clim = np.zeros((12, nlat, nlon), dtype=np.float64)
        for m in range(12):
            sel = train_mask & (months == m + 1)
            clim[m] = prec[sel].mean(axis=0)
        clim_fc = clim[months - 1]
        
        # Persistência
        pers_fc = np.empty_like(prec, dtype=np.float64)
        pers_fc[1:] = prec[:-1]
        pers_fc[0] = prec[0]
        
        # RMSE
        r_clim = float(np.sqrt(np.mean((clim_fc[test_mask] - prec[test_mask]) ** 2)))
        r_pers = float(np.sqrt(np.mean((pers_fc[test_mask] - prec[test_mask]) ** 2)))
        
        # Verificar valores esperados (GPCP SA)
        assert 40 < r_clim < 55, "RMSE clim deveria ser 40-55, got %.1f" % r_clim
        assert 60 < r_pers < 80, "RMSE pers deveria ser 60-80, got %.1f" % r_pers
        
        print("  OK: RMSE clim=%.1f, RMSE pers=%.1f" % (r_clim, r_pers))
        return True
    except Exception as e:
        print("  FALHOU: %s" % str(e))
        return False


def test_synthetic_pipeline():
    """Testa pipeline com dados sintéticos."""
    print("[TESTE 7] Pipeline sintético...")
    try:
        # Gerar dados sintéticos
        rng = np.random.default_rng(42)
        lats = np.arange(12, -55.1, -5.0)
        lons = np.arange(-85, -34.9, 5.0)
        LAT, LON = np.meshgrid(lats, lons, indexing="ij")
        nlat, nlon = len(lats), len(lons)
        
        T = 12 * 10  # 10 anos
        months = np.tile(np.arange(1, 13), 10)  # 1-12, não 0-11
        years = np.repeat(np.arange(2010, 2020), 12)
        
        north_wet = np.exp(-((LAT - 0) / 12.0) ** 2)
        monsoon = np.exp(-((LAT + 18) / 14.0) ** 2) * np.exp(-((LON + 55) / 18.0) ** 2)
        
        prec = np.zeros((T, nlat, nlon), dtype=np.float32)
        for t in range(T):
            season = np.cos(2 * np.pi * (months[t] - 11) / 12.0)
            field = 3.5 * north_wet + 6.0 * monsoon * (0.5 + 0.5 * season)
            prec[t] = np.clip(field + rng.normal(0, 1.0, size=field.shape), 0, None)
        
        # Baselines
        train_mask = years <= 2017
        test_mask = years >= 2018
        
        clim = np.zeros((12, nlat, nlon), dtype=np.float64)
        for m in range(12):
            sel = train_mask & (months == m + 1)
            if sel.sum() > 0:
                clim[m] = prec[sel].mean(axis=0)
            else:
                clim[m] = prec[months == m + 1].mean(axis=0) if (months == m + 1).sum() > 0 else 0
        clim_fc = clim[months - 1]
        
        pers_fc = np.empty_like(prec, dtype=np.float64)
        pers_fc[1:] = prec[:-1]
        pers_fc[0] = prec[0]
        
        r_clim = float(np.sqrt(np.mean((clim_fc[test_mask] - prec[test_mask]) ** 2)))
        r_pers = float(np.sqrt(np.mean((pers_fc[test_mask] - prec[test_mask]) ** 2)))
        s_pers = 1.0 - r_pers / r_clim
        
        assert r_clim >= 0, "RMSE clim deveria ser >= 0"
        assert s_pers <= 0, "Skill pers deveria ser <= 0"
        
        print("  OK: RMSE clim=%.2f, RMSE pers=%.2f, Skill pers=%+.3f" % (r_clim, r_pers, s_pers))
        return True
    except Exception as e:
        print("  FALHOU: %s" % str(e))
        return False


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("TUPA - Suite de Testes")
    print("=" * 60)
    print()
    
    tests = [
        test_load_gpcp,
        test_climatology,
        test_persistence,
        test_rmse,
        test_skill_score,
        test_baseline_values,
        test_synthetic_pipeline,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    # Resumo
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print("RESULTADO: %d/%d testes passaram" % (passed, total))
    
    if passed == total:
        print("TODOS OS TESTES PASSARAM!")
    else:
        print("ALGUNS TESTES FALHARAM")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
