import numpy as np
from .model import TinyNet, Adam
from .features import normalize
from .config import ALPHA, DT

def train(net, Xraw, agg_raw, y, mu, sd, epochs, lr, lambda_pde,
          decoder_only=False, batch=None, seed=0, record_every=0, tag="train"):
    Xn = normalize(Xraw, mu, sd)
    n = Xn.shape[0]
    params = net.params(decoder_only)
    opt = Adam(params, lr=lr)
    r = np.random.default_rng(seed)
    history = {"total": [], "data": [], "pde": [], "epochs": []} if record_every > 0 else None
    
    for ep in range(epochs):
        idx = np.arange(n) if batch is None else r.choice(n, size=min(batch, n), replace=False)
        Xb, yb, aggb = Xn[idx], y[idx], agg_raw[idx]
        out, h, z1 = net.forward(Xb)
        
        res_d = out - yb
        res_p = out - ALPHA * DT * aggb
        dOut = (2.0 / len(idx)) * (res_d + lambda_pde * res_p)
        
        dW2 = h.T @ dOut.reshape(-1, 1)
        db2 = dOut.sum(keepdims=True)
        if decoder_only:
            grads = [dW2, db2]
        else:
            dH = dOut.reshape(-1, 1) @ net.W2.T
            dZ1 = dH * (1 - h ** 2)
            dW1 = Xb.T @ dZ1
            db1 = dZ1.sum(axis=0)
            grads = [dW1, db1, dW2, db2]
            
        opt.step(params, grads)
        
        if history is not None and ep % record_every == 0:
            mse_d = float(np.mean(res_d ** 2))
            mse_p = float(np.mean(res_p ** 2))
            history["epochs"].append(ep)
            history["total"].append(mse_d + lambda_pde * mse_p)
            history["data"].append(mse_d)
            history["pde"].append(mse_p)
            
        if ep % 200 == 0 or ep == epochs - 1:
            print(f"      [{tag}] ep {ep+1:>4d}/{epochs}  mse_data={np.mean(res_d**2):.6f}  mse_pde={np.mean(res_p**2):.6f}", flush=True)
    return net, history

def rel_l2(pred_dU, true_dU, U0):
    pred_u1 = U0.reshape(-1) + pred_dU
    true_u1 = U0.reshape(-1) + true_dU
    return np.linalg.norm(pred_u1 - true_u1) / (np.linalg.norm(true_u1) + 1e-12)

def pde_resid(pred_dU, agg):
    r = pred_dU - ALPHA * DT * agg
    return np.sqrt(np.mean(r ** 2))

def evaluate(net, Xf_test, agg_test, dU_test, U0_test, mu, sd):
    pred, _, _ = net.forward(normalize(Xf_test, mu, sd))
    return rel_l2(pred, dU_test, U0_test), pde_resid(pred, agg_test.reshape(-1))