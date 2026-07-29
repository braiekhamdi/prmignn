# tune.py
import numpy as np
import optuna
import json
import os
from main import setup_data
from pimignn.model import TinyNet
from pimignn.train import train, evaluate
from pimignn.experiments import hf_subset, pretrain
from pimignn.features import normalize

def objective(trial, data_bundle, config):
    # 1. Define the hyperparameters to tune
    lr_pre = trial.suggest_float('pretrain_lr', 1e-4, 1e-2, log=True)
    lambda_pde = trial.suggest_float('lambda_pde_pre', 0.1, 2.0)
    lr_ft = trial.suggest_float('finetune_lr', 1e-4, 1e-2, log=True)
    
    # Temporarily update config
    config["PRETRAIN_LR"] = lr_pre
    config["LAMBDA_PDE_PRE"] = lambda_pde
    config["FINETUNE_LR"] = lr_ft
    
    # 2. Run a quick 2-seed evaluation
    errs = []
    N_HF_SCARCE = config["N_HF_SCARCE"]
    N_HF = config["N_HF"]
    
    for s in range(2):  # 2 seeds for speed
        net = TinyNet(seed=s)
        net, _ = pretrain(net, data_bundle["Xf_lf"], data_bundle["agg_lf"], data_bundle["dU_lf"], 
                          data_bundle["mu"], data_bundle["sd"], config, lambda_pde, seed=s)
        
        Xhf, agghf, dUhf = hf_subset(data_bundle["Xf_hf_pool"], data_bundle["agg_hf_pool"], 
                                     data_bundle["dU_hf_pool_noisy"], N_HF, N_HF_SCARCE, 
                                     config["N_HF_POOL"], seed=s)
        
        train(net, Xhf, agghf, dUhf, data_bundle["mu"], data_bundle["sd"], 
              epochs=config["FINETUNE_EPOCHS"], lr=lr_ft, lambda_pde=config["LAMBDA_PDE_FT"], 
              decoder_only=True, seed=s, tag=f"trial_s{s}")
        
        e, _ = evaluate(net, data_bundle["Xf_test"], data_bundle["agg_test"], 
                        data_bundle["dU_test"], data_bundle["U0_test"], 
                        data_bundle["mu"], data_bundle["sd"])
        errs.append(e)
        
    return np.mean(errs)

if __name__ == "__main__":
    print("Setting up data for hyperparameter tuning...")
    data_bundle, config, _, _, _ = setup_data()
    
    # Run Bayesian Optimization
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, data_bundle, config), n_trials=20)

    print("\n=== Tuning Complete ===")
    print(f"Best trial value (Mean L2 error): {study.best_trial.value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save best params to a file
    with open("results/best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("Saved best parameters to results/best_params.json")