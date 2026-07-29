import numpy as np

class TinyNet:
    def __init__(self, in_dim=5, hidden=32, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 0.5, (in_dim, hidden)) / np.sqrt(in_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = r.normal(0, 0.5, (hidden, 1)) / np.sqrt(hidden)
        self.b2 = np.zeros(1)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        h = np.tanh(z1)
        out = h @ self.W2 + self.b2
        return out.reshape(-1), h, z1

    def params(self, decoder_only=False):
        if decoder_only:
            return [self.W2, self.b2]
        return [self.W1, self.b1, self.W2, self.b2]


class Adam:
    def __init__(self, params, lr=0.02):
        self.lr = lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
        self.b1, self.b2, self.eps = 0.9, 0.999, 1e-8

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)