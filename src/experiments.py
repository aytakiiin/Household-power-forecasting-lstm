import sys, json, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from common import *
from lstm_np import LSTMRegressor, gradient_check

t0 = time.time()
s = load_daily(); T = tabular(s); SX, SC, SY, SI = sequences(s, 28)
common = T.index.intersection(SI); T = T.loc[common]
keep = np.isin(SI, common); SX, SC, SY, SI = SX[keep], SC[keep], SY[keep], SI[keep]
feats = [c for c in T.columns if c != 'y']
FOLDS = [('2007-12-31', '2008-01-01', '2008-12-31'), ('2008-12-31', '2009-01-01', '2009-12-31'), ('2009-12-31', '2010-01-01', '2010-12-31')]
R = {'gradient_check': gradient_check(), 'n_days': int(len(T))}

def fit_lstm(tr_mask, hidden, lr, seeds, L_epochs=80):
    Xtr, Ctr, ytr = SX[tr_mask], SC[tr_mask], SY[tr_mask]; mu, sd = ytr.mean(), ytr.std()
    def sc(X): X = X.copy(); X[:, :, 0] = (X[:, :, 0] - mu) / sd; return X
    va = len(ytr) - 90; models = []; hists = []
    for sd_ in seeds:
        m = LSTMRegressor(6, 5, hidden=hidden, seed=sd_)
        h = m.fit(sc(Xtr[:va]), Ctr[:va], (ytr[:va] - mu) / sd, sc(Xtr[va:]), Ctr[va:], (ytr[va:] - mu) / sd, epochs=L_epochs, lr=lr, seed=sd_)
        models.append(m); hists.append(h)
    def pred(mask):
        return np.mean([m.predict(sc(SX[mask]), SC[mask]) for m in models], axis=0) * sd + mu
    return pred, hists

# ---- hyper-parameter search (validation = last 90 days of the FINAL training set only; no test leakage)
tr_final = SI <= '2009-12-31'; Xv = np.where(tr_final)[0]; val_mask = np.zeros(len(SI), bool); val_mask[Xv[-90:]] = True
tr_for_tune = tr_final & ~val_mask
grid = []
for hidden in (16, 32):
    for lr in (3e-3, 1e-2):
        pr, _ = fit_lstm(tr_for_tune, hidden, lr, seeds=[0, 1])
        grid.append(dict(hidden=hidden, lr=lr, val_MAE=metrics(SY[val_mask], pr(val_mask))['MAE']))
best = min(grid, key=lambda g: g['val_MAE']); R['lstm_grid'] = grid; R['lstm_best'] = best
print('grid', grid, time.time() - t0, flush=True)

SEEDS = [0, 1, 2, 3, 4]
def run_fold(trend, ts, te):
    tr = T.index <= trend; ts_m = (T.index >= ts) & (T.index <= te)
    Ttr, Tte = T[tr], T[ts_m]; y = Tte['y'].values; out = {}
    out['Naive (روز قبل)'] = Tte['lag1'].values; out['میانگین ۷ روزه'] = Tte['r7'].values; out['Seasonal Naive (هفته قبل)'] = Tte['lag7'].values
    out['Ridge'] = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Ttr[feats], Ttr['y']).predict(Tte[feats])
    gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, max_depth=4, random_state=0).fit(Ttr[feats], Ttr['y']); out['Gradient Boosting'] = gb.predict(Tte[feats])
    out['Extra Trees'] = ExtraTreesRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=1).fit(Ttr[feats], Ttr['y']).predict(Tte[feats])
    ss = StandardScaler().fit(Ttr[feats]); mp = []
    for sd_ in range(3):
        mp.append(MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True, max_iter=800, random_state=sd_).fit(ss.transform(Ttr[feats]), Ttr['y']).predict(ss.transform(Tte[feats])))
    out['MLP'] = np.mean(mp, axis=0)
    trm = SI <= trend; tem = (SI >= ts) & (SI <= te)
    pr, hists = fit_lstm(trm, best['hidden'], best['lr'], SEEDS); out['LSTM (NumPy)'] = pr(tem)
    out['Ensemble (GB+LSTM+Ridge)'] = (out['Gradient Boosting'] + out['LSTM (NumPy)'] + out['Ridge']) / 3
    return Tte.index, y, out, hists, gb

res = {}; keepfinal = None
for k, (a, b, c) in enumerate(FOLDS):
    idx, y, out, hists, gb = run_fold(a, b, c)
    res[c[:4]] = {m: metrics(y, p) for m, p in out.items()}; res[c[:4]]['_n'] = int(len(y))
    print('fold', c[:4], {m: round(v['MAE'], 2) for m, v in res[c[:4]].items() if m != '_n'}, round(time.time() - t0), flush=True)
    if k == 2: keepfinal = (idx, y, out, hists, gb)
R['folds'] = res
names = [m for m in res['2010'] if m != '_n']
R['avg_mae'] = {m: float(np.mean([res[f][m]['MAE'] for f in res])) for m in names}
idx, y, out, hists, gb = keepfinal
# paired Wilcoxon on 2010 abs errors
ae = {m: np.abs(out[m] - y) for m in out}
def wil(a, b):
    st, p = wilcoxon(ae[a], ae[b]); return dict(a=a, b=b, mae_a=float(ae[a].mean()), mae_b=float(ae[b].mean()), p=float(p))
R['tests'] = [wil('Ensemble (GB+LSTM+Ridge)', 'Naive (روز قبل)'), wil('LSTM (NumPy)', 'Gradient Boosting'), wil('Ensemble (GB+LSTM+Ridge)', 'Gradient Boosting'), wil('Gradient Boosting', 'Naive (روز قبل)')]
# ---- prediction intervals: split-conformal on ensemble, calibrated on last 90 training days (models refit without them)
trc = (T.index <= '2009-12-31'); Tcal = T[trc]; cal_idx = Tcal.index[-90:]
Tfit = Tcal.iloc[:-90]
gbc = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, max_depth=4, random_state=0).fit(Tfit[feats], Tfit['y'])
rg = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Tfit[feats], Tfit['y'])
trm = (SI <= cal_idx[0] - pd.Timedelta(days=1)); calm = np.isin(SI, cal_idx)
prc, _ = fit_lstm(trm, best['hidden'], best['lr'], SEEDS)
pcal = (gbc.predict(Tcal.loc[cal_idx, feats]) + rg.predict(Tcal.loc[cal_idx, feats]) + prc(calm)) / 3
resid = np.abs(Tcal.loc[cal_idx, 'y'].values - pcal); q = float(np.quantile(resid, 0.9 * (1 + 1 / len(resid))))
ens = out['Ensemble (GB+LSTM+Ridge)']; lo, hi = ens - q, ens + q
R['interval'] = dict(q=q, coverage=float(np.mean((y >= lo) & (y <= hi)) * 100), nominal=90.0, width=float(2 * q))
# quantile GB
qlo = HistGradientBoostingRegressor(loss='quantile', quantile=.05, max_iter=200, learning_rate=.05, max_depth=3, random_state=0).fit(T[T.index <= '2009-12-31'][feats], T[T.index <= '2009-12-31']['y'])
qhi = HistGradientBoostingRegressor(loss='quantile', quantile=.95, max_iter=200, learning_rate=.05, max_depth=3, random_state=0).fit(T[T.index <= '2009-12-31'][feats], T[T.index <= '2009-12-31']['y'])
Tte = T.loc[idx]; l2, h2 = qlo.predict(Tte[feats]), qhi.predict(Tte[feats])
R['interval_qgb'] = dict(coverage=float(np.mean((y >= l2) & (y <= h2)) * 100), width=float(np.mean(h2 - l2)))
# anomalies: residual z-score of ensemble on 2010
err = y - ens; z = (err - err.mean()) / err.std(); an = pd.DataFrame({'date': idx, 'actual': y, 'pred': ens, 'z': z})
R['anomalies'] = [dict(date=str(r.date.date()), actual=float(r.actual), pred=float(r.pred), z=float(r.z)) for r in an.reindex(an.z.abs().sort_values(ascending=False).index).head(8).itertuples()]
R['n_anom'] = int((np.abs(z) > 2.5).sum())
R['lstm_epochs'] = [len(h) for h in hists]
# save arrays for plotting
pd.DataFrame({'y': y, **{m: p for m, p in out.items()}, 'lo': lo, 'hi': hi, 'z': z}, index=idx).to_pickle('data/final_preds.pkl')
json.dump({'train_curve_mean': [[float(np.mean([h[e][0] for h in hists if len(h) > e])), float(np.mean([h[e][1] for h in hists if len(h) > e]))] for e in range(min(len(h) for h in hists))]}, open('data/curve.json', 'w'))
json.dump(R, open('data/results_v2.json', 'w'), indent=1, ensure_ascii=False)
print('DONE', round(time.time() - t0))
print(json.dumps({k: R[k] for k in ['gradient_check', 'lstm_best', 'avg_mae', 'tests', 'interval', 'interval_qgb', 'n_anom', 'anomalies', 'lstm_epochs']}, indent=1, ensure_ascii=False))
