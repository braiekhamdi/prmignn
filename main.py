# main.py
import numpy as np
import json
import os
import time
import platform
import argparse

from prmignn.config import config
from prmignn.mesh import poisson_like_points, knn_graph
from prmignn.pde import integrate, make_snapshots, random_field
from prmignn.features import compute_features
from prmignn.experiments import run_all_experiments
from prmignn.plotting import generate_figures  

OUT = "results"
os.makedirs(OUT, exist_ok=True)

def log(msg):
    print(msg, flush=True)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def setup_data():
    hw_info = {
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "gpu": "N/A (pure NumPy, CPU only)"
    }
    log(f"  Hardware: {hw_info['processor']} | {hw_info['gpu']}")
    timing = {}

    log("[1/7] Generating irregular point-cloud meshes ...")
    t0 = time.time()
    X_lf = poisson_like_points(config["N_LF"], seed=1)
    X_hf = poisson_like_points(config["N_HF"], seed=2)
    W_lf, deg_lf, edges_lf = knn_graph(X_lf, k=config["K_NN"])
    W_hf, deg_hf, edges_hf = knn_graph(X_hf, k=config["K_NN"])
    L_lf = np.eye(config["N_LF"]) - W_lf
    L_hf = np.eye(config["N_HF"]) - W_hf
    timing["mesh_generation_s"] = time.time() - t0

    log("[2/7] Simulating heat diffusion snapshots ...")
    t0 = time.time()
    U0_lf, U1_lf = make_snapshots(X_lf, L_lf, config["N_LF_SNAPS"], n_sub=config["LF_NSUB"], seed0=10000)
    U0_hf_pool, U1_hf_pool = make_snapshots(X_hf, L_hf, config["N_HF_POOL"], n_sub=config["HF_NSUB"], seed0=20000)
    U0_test, U1_test = make_snapshots(X_hf, L_hf, config["N_TEST"], n_sub=config["HF_NSUB"], seed0=99000)

    fid_gaps = []
    for i in range(min(10, config["N_TEST"])):
        u_lf = integrate(U0_test[i], L_hf, n_sub=config["LF_NSUB"])
        u_hf = U1_test[i]
        fid_gaps.append(np.linalg.norm(u_lf - u_hf) / (np.linalg.norm(u_hf) + 1e-12))
    timing["simulation_s"] = time.time() - t0

    log("[3/7] Computing node features ...")
    t0 = time.time()
    Xf_lf, agg_lf = compute_features(U0_lf, W_lf, L_lf, deg_lf, X_lf)
    dU_lf = (U1_lf - U0_lf).reshape(-1)
    Xf_hf_pool, agg_hf_pool = compute_features(U0_hf_pool, W_hf, L_hf, deg_hf, X_hf)
    dU_hf_pool_clean = (U1_hf_pool - U0_hf_pool).reshape(-1)
    noise_rng = np.random.default_rng(777)
    dU_hf_pool_noisy = dU_hf_pool_clean + noise_rng.normal(0, config["HF_NOISE"], size=dU_hf_pool_clean.shape)
    Xf_test, agg_test = compute_features(U0_test, W_hf, L_hf, deg_hf, X_hf)
    dU_test = (U1_test - U0_test).reshape(-1)
    mu, sd = Xf_lf.mean(0), Xf_lf.std(0) + 1e-8
    timing["features_s"] = time.time() - t0

    data_bundle = {
        "Xf_lf": Xf_lf, "agg_lf": agg_lf, "dU_lf": dU_lf,
        "Xf_hf_pool": Xf_hf_pool, "agg_hf_pool": agg_hf_pool, "dU_hf_pool_noisy": dU_hf_pool_noisy,
        "Xf_test": Xf_test, "agg_test": agg_test, "dU_test": dU_test, "U0_test": U0_test,
        "mu": mu, "sd": sd, "fid_gaps": fid_gaps
    }
    
    mesh_info = {
        "X_lf": X_lf.tolist(), "X_hf": X_hf.tolist(),
        "edges_lf": edges_lf, "edges_hf": edges_hf,
        "sample_field_lf": random_field(X_lf, seed=42).tolist(),
        "sample_field_hf": random_field(X_hf, seed=42).tolist()
    }
    return data_bundle, config, hw_info, timing, mesh_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default='all', choices=['proposed', 'ablation', 'all'],
                        help="'proposed' runs only PR-MF-GNN, 'ablation' runs baselines/ablations, 'all' runs everything sequentially and generates plots.")
    args = parser.parse_args()

    total_t0 = time.time()
    data_bundle, config, hw_info, timing, mesh_info = setup_data()
    
    all_results = run_all_experiments(data_bundle, config, hw_info, timing, run_mode=args.run)
    all_results["mesh"] = mesh_info

    log("\n[6/7] Saving results ...")
    timing["total_s"] = time.time() - total_t0
    all_results["timing"] = timing
    
    fname = f"results_{args.run}.json" if args.run != "all" else "results.json"
    outpath = os.path.join(OUT, fname)
    with open(outpath, "w") as f:
        json.dump(all_results, f, cls=NumpyEncoder, indent=2)
    log(f"    Saved {outpath} ({os.path.getsize(outpath)/1024:.0f} KB)")
    
    # Automatically generate figures if --run all is used
    if args.run == 'all':
        generate_figures(all_results, out_dir=OUT)
        log("\n[7/7] Done. All experiments completed and figures generated.")
    else:
        log("\n[7/7] Done. Run 'python plot_from_json.py' to generate merged figures.")