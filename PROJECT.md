# Tupã — Previsão Mensal de Precipitação sobre a América do Sul

## Hackathon WorCAP 2026

- **Evento:** Hackathon WorCAP 2026 (INPE)
- **Período:** 15-22/09/2026 (Kaggle)
- **Inscrição:** https://forms.gle/yzJpeXQQK1km1uSn6
- **Site:** https://www.gov.br/inpe/pt-br/eventos/worcap-2026

## Desafio

Desenvolver modelo de ML para prever a precipitação média do mês seguinte sobre a América do Sul usando dados ERA5.

## Equipe

- Ana Rita Cabral
- Yasmin Medeiros
- Paulo Sergio Cabral

## Referências Bibliográficas

### Papers-chave

1. **Anochi, de Almeida & Campos Velho (2021)** — "Machine Learning for Climate Precipitation Prediction Modeling over South America"
   - Remote Sensing, 13(13):2468
   - https://www.mdpi.com/2072-4292/13/13/2468
   - MLP: 2 camadas × 25 ReLU, Adam 1e-3, 1000 epochs
   - Preditores: NCEP R1 (2.5°): u,v @850/500 hPa, T2m, q850
   - Split: treino 1980-2016, teste 2017-2019
   - Resultado: MLP > BAM operacional

2. **Anochi & Shimizu (2024)** — "Precipitation Forecasting and Drought Monitoring in South America Using a Machine Learning Approach"
   - Meteorology, 4(1):1
   - https://www.mdpi.com/2674-0494/4/1/1
   - MLP supervisionada, dados GPCP v3.2
   - Métrica: SPI-1 (seca)
   - NN > NMME em todos os meses

3. **Monego, Anochi & Campos Velho (2022)** — "South America Seasonal Precipitation Prediction by Gradient-Boosting Machine-Learning Approach"
   - Atmosphere, 13(2):243
   - XGBoost para previsão sazonal

4. **Domingos et al. (2025)** — "Exploring Deep Learning Techniques for Seasonal Prediction of Autumn Precipitation in South America"
   - WCAMA/SBC
   - CNN 1D, LSTM, GRU, GConvLSTM

### Prior art

- Hackathon WorCAP 2022 (Kaggle): https://www.kaggle.com/competitions/hackathon-worcap-2022

## Template de Design

### Preditores (ERA5)

| Variável | Nível | Lag |
|----------|-------|-----|
| u (vento zonal) | 850 hPa, 500 hPa | t, t-1 |
| v (vento meridional) | 850 hPa, 500 hPa | t, t-1 |
| T2m (temperatura 2m) | superfície | t, t-1 |
| q (umidade específica) | 850 hPa | t, t-1 |
| precipitação total | superfície | t, t-1 |

### Teleconexões

- Niño 3.4 (ENSO)
- SAM/AAO (Antártico)
- MJO (Onda de Madden-Julian)
- ATL3 (Atlântico tropical)

### Features temporais

- Lat, Lon
- sin(mês), cos(mês)

### Alvo

- Precipitação do mês seguinte (anomalia = precip - climatologia mensal)

### Baselines

0. **Climatologia** — média histórica do mês naquele ponto
1. **Persistência** — precipitação do mês atual

### Modelos

- **Fase 1:** XGBoost/LightGBM por ponto de grade
- **Fase 2:** MLP regularizado
- **Fase 3:** CNN/UNet ou ConvLSTM
- **Fase 4:** Ensemble (climatologia + ML + DL)

### Métricas

- RMSE
- Skill Score vs. climatologia
- Correlação de anomalia

### Validação

- Temporal: rolling/block CV (NUNCA aleatória)
- Split: treino até ~2016, validação 2017-2019, teste 2020+

## Decisões Pendentes

- [ ] Time ou solo?
- [ ] API key CDS disponível?
- [ ] Nível com ERA5/xarray?

## Status

- [x] Fase 0: Pesquisa bibliográfica
- [ ] Fase 1: Dados + baseline
- [ ] Fase 2: Features + modelo simples
- [ ] Fase 3: DL (se sobrar capacidade)
- [ ] Fase 4: 15-22/09 Kaggle
