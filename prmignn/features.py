import numpy as np

def compute_features(U0, W, L, deg, X):
    n_snaps, n = U0.shape
    agg = -(L @ U0.T).T
    degn = np.tile(deg / deg.max(), (n_snaps, 1))
    xs = np.tile(X[:, 0], (n_snaps, 1))
    ys = np.tile(X[:, 1], (n_snaps, 1))
    X_feat = np.stack([U0, agg, degn, xs, ys], axis=-1).reshape(-1, 5)
    return X_feat, agg

def normalize(X, mu, sd):
    return (X - mu) / sd