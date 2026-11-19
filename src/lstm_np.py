"""Minimal LSTM regressor in pure NumPy (single layer, many-to-one) with Adam + BPTT.
Written from scratch because no deep-learning framework is installed in this environment.
Architecture: LSTM(H) over a window of L days -> concat(h_T, calendar features of target day) -> Dense(1)
"""
import numpy as np

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

class LSTMRegressor:
    def __init__(self, n_in, n_cal, hidden=24, seed=0):
        r = np.random.RandomState(seed); H = hidden; self.H, self.n_in, self.n_cal = H, n_in, n_cal
        s = 1.0 / np.sqrt(H)
        self.p = {
            'W': r.uniform(-s, s, (n_in + H, 4 * H)),
            'b': np.zeros(4 * H),
            'Wy': r.uniform(-s, s, (H + n_cal, 1)),
            'by': np.zeros(1)}
        self.p['b'][H:2 * H] = 1.0   # forget-gate bias = 1
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, X, C, keep=False):
        N, L, _ = X.shape; H = self.H; W, b = self.p['W'], self.p['b']
        h = np.zeros((N, H)); c = np.zeros((N, H)); cache = []
        for t in range(L):
            z = np.concatenate([X[:, t], h], axis=1)
            a = z @ W + b
            i = sigmoid(a[:, :H]); f = sigmoid(a[:, H:2 * H]); g = np.tanh(a[:, 2 * H:3 * H]); o = sigmoid(a[:, 3 * H:])
            c_new = f * c + i * g; tc = np.tanh(c_new); h_new = o * tc
            if keep: cache.append((z, i, f, g, o, c, tc))
            h, c = h_new, c_new
        hc = np.concatenate([h, C], axis=1)
        y = hc @ self.p['Wy'] + self.p['by']
        return y[:, 0], (cache, hc)

    def loss_grad(self, X, C, y):
        N = len(y); pred, (cache, hc) = self.forward(X, C, keep=True)
        err = pred - y; loss = np.mean(err ** 2)
        dy = (2.0 / N) * err[:, None]
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        g['Wy'] = hc.T @ dy; g['by'] = dy.sum(0)
        dhc = dy @ self.p['Wy'].T; dh = dhc[:, :self.H]; dc = np.zeros_like(dh)
        H = self.H; W = self.p['W']
        for t in reversed(range(X.shape[1])):
            z, i, f, g_, o, c_prev, tc = cache[t]
            do = dh * tc; dc = dc + dh * o * (1 - tc ** 2)
            di = dc * g_; dg = dc * i; df = dc * c_prev; dc_prev = dc * f
            da = np.concatenate([di * i * (1 - i), df * f * (1 - f), dg * (1 - g_ ** 2), do * o * (1 - o)], axis=1)
            g['W'] += z.T @ da; g['b'] += da.sum(0)
            dz = da @ W.T; dh = dz[:, self.n_in:]; dc = dc_prev
        return loss, g

    def step(self, grads, lr=3e-3, clip=1.0, b1=0.9, b2=0.999, wd=0.0):
        tot = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
        sc = min(1.0, clip / (tot + 1e-12)); self.t += 1
        for k in self.p:
            gk = grads[k] * sc + wd * self.p[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk; self.v[k] = b2 * self.v[k] + (1 - b2) * gk ** 2
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)

    def predict(self, X, C): return self.forward(X, C)[0]

    def fit(self, Xtr, Ctr, ytr, Xva, Cva, yva, epochs=60, bs=32, lr=3e-3, patience=10, wd=1e-4, seed=0, verbose=False):
        r = np.random.RandomState(seed); best = (1e18, None); bad = 0; hist = []
        for ep in range(epochs):
            idx = r.permutation(len(ytr))
            for s in range(0, len(idx), bs):
                j = idx[s:s + bs]; _, g = self.loss_grad(Xtr[j], Ctr[j], ytr[j]); self.step(g, lr=lr, wd=wd)
            tl = np.mean((self.predict(Xtr, Ctr) - ytr) ** 2); vl = np.mean((self.predict(Xva, Cva) - yva) ** 2)
            hist.append((tl, vl))
            if verbose: print(ep, round(tl, 4), round(vl, 4))
            if vl < best[0] - 1e-5: best = (vl, {k: v.copy() for k, v in self.p.items()}); bad = 0
            else:
                bad += 1
                if bad >= patience: break
        self.p = best[1]; return hist

def gradient_check(seed=0, eps=1e-5):
    r = np.random.RandomState(seed); m = LSTMRegressor(3, 2, hidden=5, seed=1)
    X = r.randn(4, 6, 3); C = r.randn(4, 2); y = r.randn(4)
    _, g = m.loss_grad(X, C, y); worst = 0
    for k in m.p:
        flat = m.p[k].reshape(-1); gf = g[k].reshape(-1)
        for idx in r.choice(len(flat), min(8, len(flat)), replace=False):
            old = flat[idx]; flat[idx] = old + eps; lp, _ = m.loss_grad(X, C, y)
            flat[idx] = old - eps; lm, _ = m.loss_grad(X, C, y); flat[idx] = old
            num = (lp - lm) / (2 * eps); rel = abs(num - gf[idx]) / max(1e-8, abs(num) + abs(gf[idx])); worst = max(worst, rel)
    return worst

if __name__ == '__main__':
    print('max relative gradient error:', gradient_check())
