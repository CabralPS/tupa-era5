# Tupã ⚡

Previsão mensal de precipitação sobre a América do Sul com ML sobre ERA5.

## Hackathon WorCAP 2026

- **Evento:** 15-22/09/2026 (Kaggle)
- **Inscrição:** https://forms.gle/yzJpeXQQK1km1uSn6
- **Site:** https://www.gov.br/inpe/pt-br/eventos/worcap-2026

## Estrutura

```
tupã-era5/
├── PROJECT.md          # Memória do projeto (design, decisões, referências)
├── baseline.py         # Baseline: Climatologia + Persistência
├── requirements.txt    # Dependências
└── README.md           # Este arquivo
```

## Quick Start

```bash
# Instalar dependências
pip install -r requirements.txt

# Rodar baseline
python baseline.py
```

## Pipeline

1. **Fase 0** — Pesquisa bibliográfica ✓
2. **Fase 1** — Dados + baseline (climatologia + persistência)
3. **Fase 2** — Features + XGBoost/LightGBM
4. **Fase 3** — DL (CNN/UNet, ConvLSTM)
5. **Fase 4** — Ensemble + Kaggle

## Referências

- Anochi et al. (2021) — Remote Sensing, 13(13):2468
- Anochi & Shimizu (2024) — Meteorology, 4(1):1
- Monego et al. (2022) — Atmosphere, 13(2):243

## Licença

Uso interno para o Hackathon WorCAP 2026.
