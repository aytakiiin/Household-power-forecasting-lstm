import numpy as np, pandas as pd

def load_daily(path='data/daily_aggregated.csv'):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    s = df['kwh_adj'].copy(); s[df['cov'] < .5] = np.nan
    return s.asfreq('D').interpolate(limit=14)

def cal_feats(idx):
    doy = idx.dayofyear.values; dw = idx.dayofweek.values
    return np.column_stack([np.sin(2*np.pi*dw/7), np.cos(2*np.pi*dw/7), np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25), (dw >= 5).astype(float)])

def tabular(s):
    X = pd.DataFrame({'y': s})
    for L in (1, 2, 7, 14): X[f'lag{L}'] = s.shift(L)
    X['r7'] = s.shift(1).rolling(7).mean(); X['r30'] = s.shift(1).rolling(30).mean()
    cf = cal_feats(s.index); X['dow'] = s.index.dayofweek; X['wknd'] = cf[:, 4]
    X['s1'] = cf[:, 2]; X['c1'] = cf[:, 3]
    return X.dropna()

def sequences(s, L=28):
    """windows of last L days (value + calendar of that day) -> next day; returns X, C(target-day calendar), y, target index"""
    v = s.values; cf = cal_feats(s.index); Xs, Cs, ys, ix = [], [], [], []
    for t in range(L, len(s)):
        if np.isnan(v[t-L:t+1]).any(): continue
        Xs.append(np.column_stack([v[t-L:t], cf[t-L:t]])); Cs.append(cf[t]); ys.append(v[t]); ix.append(s.index[t])
    return np.array(Xs), np.array(Cs), np.array(ys), pd.DatetimeIndex(ix)

def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p); e = p - y
    return dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e**2))), R2=float(1 - np.sum(e**2)/np.sum((y-y.mean())**2)), MAPE=float(np.mean(np.abs(e/y))*100))
