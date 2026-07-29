import numpy as np

def poisson_like_points(n, seed, min_dist_factor=0.6):
    r = np.random.default_rng(seed)
    pts = []
    target_min = min_dist_factor / np.sqrt(n)
    tries = 0
    while len(pts) < n and tries < 200000:
        p = r.random(2)
        if not pts:
            pts.append(p)
        else:
            d = np.min(np.linalg.norm(np.array(pts) - p, axis=1))
            if d > target_min:
                pts.append(p)
        tries += 1
    while len(pts) < n:
        pts.append(r.random(2))
    return np.array(pts)

def knn_graph(X, k=6):
    n = X.shape[0]
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    idx = np.argsort(D, axis=1)[:, 1:k + 1]
    W = np.zeros((n, n))
    edges = []
    for i in range(n):
        for j in idx[i]:
            w = 1.0 / (D[i, j] + 1e-8)
            W[i, j] = w
            W[j, i] = w
            edges.append((i, j))
    edges_unique = list(set(tuple(sorted(e)) for e in edges))
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    Wn = W / rs
    deg = (W > 0).sum(axis=1).astype(float)
    return Wn, deg, edges_unique