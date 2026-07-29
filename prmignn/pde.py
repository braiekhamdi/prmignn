import numpy as np
from .config import ALPHA, DT, WAVENUM_RANGE

def integrate(u0, L, n_sub):
    u = u0.copy()
    dt_sub = DT / n_sub
    for _ in range(n_sub):
        u = u - ALPHA * dt_sub * (L @ u)
    return u

def random_field(X, seed):
    r = np.random.default_rng(seed)
    n_modes = r.integers(2, 5)
    f = np.zeros(X.shape[0])
    for _ in range(n_modes):
        p, q = r.uniform(*WAVENUM_RANGE, size=2)
        phi = r.uniform(0, 2 * np.pi)
        amp = r.uniform(0.5, 1.0)
        f += amp * np.sin(2 * np.pi * (p * X[:, 0] + q * X[:, 1]) + phi)
    return f / max(1.0, np.max(np.abs(f)))

def make_snapshots(X, L, n_snaps, n_sub, seed0):
    U0, U1 = [], []
    for s in range(n_snaps):
        u0 = random_field(X, seed=seed0 + s)
        u1 = integrate(u0, L, n_sub)
        U0.append(u0); U1.append(u1)
    return np.array(U0), np.array(U1)