# Household Electricity Consumption — Analysis & Forecasting (LSTM from scratch)

**Author:** Aytakin Mardani · aytakin57@gmail.com

Next-day household electricity forecasting from 4 years of minute-level data
(UCI *Individual Household Electric Power Consumption*, ~2.07M rows, Dec 2006 – Nov 2010).

## Highlights
- Full EDA: seasonality, daily load curves, sub-metering shares, power quality, correlations (see PDF report, in Persian)
- **LSTM implemented from scratch in NumPy** (forward pass, backpropagation through time, Adam, gradient clipping, early stopping), verified with a **numerical gradient check** (max relative error ≈ 1.4e-8)
- **Walk-forward evaluation** on 3 consecutive test years (2008, 2009, 2010); hyper-parameters tuned on validation data only
- 9 models compared: naive baselines, Ridge, Gradient Boosting, Extra Trees, MLP, LSTM, Ensemble
- Paired Wilcoxon tests, **90% prediction intervals** (split-conformal and quantile regression), residual-based **anomaly detection**

## Results (MAE, kWh/day, mean over 3 test years — lower is better)
| Model | MAE |
|---|---|
| Ensemble (GB + LSTM + Ridge) | 4.69 |
| Ridge | 4.73 |
| LSTM (NumPy) | 4.78 |
| Gradient Boosting | 5.18 |
| Naive (yesterday) | 5.76 |

Honest reading: the Ensemble beats the naive baseline significantly (2010, p ≈ 0.0004), but the differences between
LSTM, Gradient Boosting and the Ensemble are **not** statistically significant. With ~1,000 training days, simple and
ensembled models are as good as deep ones. Weather data would be the most valuable addition.

## Reproduce
```bash
pip install -r requirements.txt
# optional: rebuild the daily file from the raw UCI data
python src/prepare_data.py household_power_consumption.txt
python src/experiments.py      # ~1 min on CPU -> data/results_v2.json
python src/make_figures.py     # -> figures/
```
Interactive walkthrough: `notebooks/forecasting_lstm.ipynb`.

## Layout
```
src/lstm_np.py       LSTM from scratch + gradient check
src/common.py        loading, features, windowing, metrics
src/experiments.py   walk-forward evaluation, tests, intervals, anomalies
src/prepare_data.py  raw minute data -> daily aggregates
data/                daily_aggregated.csv, results, predictions
report/              full report (PDF, Persian)
```
The raw 133 MB file is not included; download it from the UCI Machine Learning Repository.
