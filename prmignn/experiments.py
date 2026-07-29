# prmignn/experiments.py
import numpy as np
import time
from .model import TinyNet
from .train import train, evaluate
from .features import normalize

def hf_subset(Xf_hf_pool, agg_hf_pool, dU_hf_pool_noisy, N_HF, n_snaps, pool_size, seed=1):
    r = np.random.default_rng(seed)
    snap_idx = r.choice(pool_size, size=n_snaps, replace=False)
    node_idx = np.concatenate([np.arange(N_HF) + s * N_HF for s in snap_idx])
    return (Xf_hf_pool[node_idx],
            agg_hf_pool.reshape(-1, N_HF)[snap_idx].reshape(-1),
            dU_hf_pool_noisy[node_idx])

def pretrain(net, Xf_lf, agg_lf, dU_lf, mu, sd, config, lambda_pde, seed=0, record_every=0):
    net, hist = train(net, Xf_lf, agg_lf.reshape(-1), dU_lf, mu, sd,
                      epochs=config["PRETRAIN_EPOCHS"], lr=config["PRETRAIN_LR"],
                      lambda_pde=lambda_pde, decoder_only=False, seed=seed,
                      record_every=record_every, tag=f"pretrain_s{seed}")
    return net, hist

def run_all_experiments(data_bundle, config, hw_info, timing, run_mode="all"):
    print(f"[4/7] Running experiments (Mode: {run_mode}) ...", flush=True)
    T_GLOBAL = time.time()
    
    Xf_lf = data_bundle["Xf_lf"]; agg_lf = data_bundle["agg_lf"]; dU_lf = data_bundle["dU_lf"]
    Xf_hf_pool = data_bundle["Xf_hf_pool"]; agg_hf_pool = data_bundle["agg_hf_pool"]
    dU_hf_pool_noisy = data_bundle["dU_hf_pool_noisy"]
    Xf_test = data_bundle["Xf_test"]; agg_test = data_bundle["agg_test"]; dU_test = data_bundle["dU_test"]
    U0_test = data_bundle["U0_test"]
    mu = data_bundle["mu"]; sd = data_bundle["sd"]
    fid_gaps = data_bundle["fid_gaps"]
    
    N_HF = config["N_HF"]; N_HF_SCARCE = config["N_HF_SCARCE"]
    N_SEEDS = config["N_SEEDS"]; N_SEEDS_EFF = config["N_SEEDS_EFF"]
    snap_counts = config["SNAP_COUNTS"]
    
    all_results = {
        "config": config, "hardware": hw_info, "timing": timing,
        "fidelity_gap": {
            "per_snapshot": [float(g) for g in fid_gaps],
            "mean": float(np.mean(fid_gaps)), "std": float(np.std(fid_gaps))
        }
    }

    def run_subset(method_name, lambda_pde_pre, lambda_pde_ft, decoder_only, epochs, lr, tag):
        print(f"  [{method_name}] ...", flush=True)
        t0 = time.time()
        errs, resids = [], []
        for s in range(N_SEEDS):
            net = TinyNet(seed=s)
            if lambda_pde_pre > 0:
                net, _ = pretrain(net, Xf_lf, agg_lf, dU_lf, mu, sd, config, lambda_pde_pre, seed=s)
            Xhf, agghf, dUhf = hf_subset(Xf_hf_pool, agg_hf_pool, dU_hf_pool_noisy, N_HF, N_HF_SCARCE, config["N_HF_POOL"], seed=s)
            train(net, Xhf, agghf, dUhf, mu, sd, epochs=epochs, lr=lr,
                  lambda_pde=lambda_pde_ft, decoder_only=decoder_only, seed=s, tag=f"{tag}_s{s}")
            e, r_ = evaluate(net, Xf_test, agg_test, dU_test, U0_test, mu, sd)
            errs.append(e); resids.append(r_)
            print(f"    seed {s}: L2={e*100:.2f}%  PDE_res={r_:.4f}", flush=True)
        return {
            "per_seed_l2": errs, "per_seed_pde_resid": resids,
            "mean_l2": float(np.mean(errs)), "std_l2": float(np.std(errs)),
            "mean_pde_resid": float(np.mean(resids))
        }, time.time() - t0

    if run_mode in ["proposed", "all"]:
        res, t = run_subset("Proposed", config["LAMBDA_PDE_PRE"], config["LAMBDA_PDE_FT"], True, config["FINETUNE_EPOCHS"], config["FINETUNE_LR"], "prop")
        all_results["proposed"] = res; timing["proposed_s"] = t
        
        print("  [Proposed-extra] LF-pretrained ONLY ...", flush=True)
        t0 = time.time()
        errs, resids = [], []
        for s in range(N_SEEDS):
            net = TinyNet(seed=s)
            net, _ = pretrain(net, Xf_lf, agg_lf, dU_lf, mu, sd, config, config["LAMBDA_PDE_PRE"], seed=s)
            e, r_ = evaluate(net, Xf_test, agg_test, dU_test, U0_test, mu, sd)
            errs.append(e); resids.append(r_)
        all_results["pretrain_only"] = {
            "per_seed_l2": errs, "per_seed_pde_resid": resids,
            "mean_l2": float(np.mean(errs)), "std_l2": float(np.std(errs)),
            "mean_pde_resid": float(np.mean(resids))
        }
        timing["pretrain_only_s"] = time.time() - t0

        print("  [Proposed] Prediction fields & Convergence ...", flush=True)
        t0 = time.time()
        net_pre = TinyNet(seed=0)
        net_pre, _ = pretrain(net_pre, Xf_lf, agg_lf, dU_lf, mu, sd, config, config["LAMBDA_PDE_PRE"], seed=0)
        pred_pre, _, _ = net_pre.forward(normalize(Xf_test, mu, sd))
        Xhf, agghf, dUhf = hf_subset(Xf_hf_pool, agg_hf_pool, dU_hf_pool_noisy, N_HF, N_HF_SCARCE, config["N_HF_POOL"], seed=0)
        train(net_pre, Xhf, agghf, dUhf, mu, sd, epochs=config["FINETUNE_EPOCHS"], lr=config["FINETUNE_LR"],
              lambda_pde=config["LAMBDA_PDE_FT"], decoder_only=True, seed=0, tag="vis_prop")
        pred_prop, _, _ = net_pre.forward(normalize(Xf_test, mu, sd))
        s0, s1 = 0, N_HF
        true_field = U0_test[0] + dU_test[s0:s1]
        pred_pre_field = U0_test[0] + pred_pre[s0:s1]
        pred_prop_field = U0_test[0] + pred_prop[s0:s1]
        all_results["prediction_fields_proposed"] = {
            "true": true_field.tolist(), "lf_pretrained": pred_pre_field.tolist(),
            "proposed": pred_prop_field.tolist(),
            "errors": {
                "lf_pretrained": float(np.linalg.norm(pred_pre_field - true_field) / (np.linalg.norm(true_field) + 1e-12)),
                "proposed": float(np.linalg.norm(pred_prop_field - true_field) / (np.linalg.norm(true_field) + 1e-12))
            }
        }

        net_conv = TinyNet(seed=0)
        net_conv, hist_s1 = pretrain(net_conv, Xf_lf, agg_lf, dU_lf, mu, sd, config, config["LAMBDA_PDE_PRE"], seed=0, record_every=5)
        Xhf_c, agghf_c, dUhf_c = hf_subset(Xf_hf_pool, agg_hf_pool, dU_hf_pool_noisy, N_HF, N_HF_SCARCE, config["N_HF_POOL"], seed=0)
        net_conv, hist_s2 = train(net_conv, Xhf_c, agghf_c, dUhf_c, mu, sd, epochs=config["FINETUNE_EPOCHS"],
                                  lr=config["FINETUNE_LR"], lambda_pde=config["LAMBDA_PDE_FT"],
                                  decoder_only=True, seed=0, record_every=5, tag="conv_s2")
        all_results["training_convergence"] = {"stage1": hist_s1, "stage2": hist_s2}
        timing["proposed_extras_s"] = time.time() - t0

    if run_mode in ["ablation", "all"]:
        res, t = run_subset("Baseline", 0.0, 0.0, False, config["BASELINE_EPOCHS"], config["BASELINE_LR"], "base")
        all_results["baseline"] = res; timing["baseline_s"] = t
        
        res, t = run_subset("No PDE", 0.0, 0.0, True, config["FINETUNE_EPOCHS"], config["FINETUNE_LR"], "nopde")
        all_results["ablation_no_pde"] = res; timing["ablation_nopde_s"] = t
        
        res, t = run_subset("Full FT", config["LAMBDA_PDE_PRE"], config["LAMBDA_PDE_FT"], False, config["FINETUNE_EPOCHS"], config["FINETUNE_LR"], "fullft")
        all_results["ablation_fullft"] = res; timing["ablation_fullft_s"] = t

        print(f"  [Ablation] Data-efficiency ...", flush=True)
        t0 = time.time()
        eff_data = {"snap_counts": snap_counts, "proposed": {"per_seed": []}, "baseline": {"per_seed": []}}
        for ns in snap_counts:
            ep_seeds, eb_seeds = [], []
            for s in range(N_SEEDS_EFF):
                net = TinyNet(seed=s)
                net, _ = pretrain(net, Xf_lf, agg_lf, dU_lf, mu, sd, config, config["LAMBDA_PDE_PRE"], seed=s)
                Xhf, agghf, dUhf = hf_subset(Xf_hf_pool, agg_hf_pool, dU_hf_pool_noisy, N_HF, ns, config["N_HF_POOL"], seed=s)
                train(net, Xhf, agghf, dUhf, mu, sd, epochs=config["FINETUNE_EPOCHS"], lr=config["FINETUNE_LR"],
                      lambda_pde=config["LAMBDA_PDE_FT"], decoder_only=True, seed=s, tag=f"eff_p_n{ns}_s{s}")
                ep_seeds.append(evaluate(net, Xf_test, agg_test, dU_test, U0_test, mu, sd)[0])
                
                netb = TinyNet(seed=300 + s)
                train(netb, Xhf, agghf, dUhf, mu, sd, epochs=config["BASELINE_EPOCHS"], lr=config["BASELINE_LR"],
                      lambda_pde=0.0, seed=s, tag=f"eff_b_n{ns}_s{s}")
                eb_seeds.append(evaluate(netb, Xf_test, agg_test, dU_test, U0_test, mu, sd)[0])
            eff_data["proposed"]["per_seed"].append(ep_seeds)
            eff_data["baseline"]["per_seed"].append(eb_seeds)
            print(f"    n={ns}: prop={np.mean(ep_seeds)*100:.2f}%  base={np.mean(eb_seeds)*100:.2f}%", flush=True)
        
        # FIX: Removed the zip(*) which caused the shape mismatch
        eff_data["proposed"]["mean"] = [float(np.mean(s)) for s in eff_data["proposed"]["per_seed"]]
        eff_data["proposed"]["std"] = [float(np.std(s)) for s in eff_data["proposed"]["per_seed"]]
        eff_data["baseline"]["mean"] = [float(np.mean(s)) for s in eff_data["baseline"]["per_seed"]]
        eff_data["baseline"]["std"] = [float(np.std(s)) for s in eff_data["baseline"]["per_seed"]]
        all_results["data_efficiency"] = eff_data
        timing["data_efficiency_s"] = time.time() - t0

        print("  [Ablation] Baseline Prediction fields ...", flush=True)
        net_base = TinyNet(seed=100)
        Xhf, agghf, dUhf = hf_subset(Xf_hf_pool, agg_hf_pool, dU_hf_pool_noisy, N_HF, N_HF_SCARCE, config["N_HF_POOL"], seed=0)
        train(net_base, Xhf, agghf, dUhf, mu, sd, epochs=config["BASELINE_EPOCHS"], lr=config["BASELINE_LR"],
              lambda_pde=0.0, seed=0, tag="vis_base")
        pred_base, _, _ = net_base.forward(normalize(Xf_test, mu, sd))
        s0, s1 = 0, N_HF
        true_field = U0_test[0] + dU_test[s0:s1]
        pred_base_field = U0_test[0] + pred_base[s0:s1]
        all_results["prediction_fields_baseline"] = {
            "baseline": pred_base_field.tolist(),
            "error_baseline": float(np.linalg.norm(pred_base_field - true_field) / (np.linalg.norm(true_field) + 1e-12))
        }

    timing["total_experiments_s"] = time.time() - T_GLOBAL
    all_results["timing"] = timing
    return all_results