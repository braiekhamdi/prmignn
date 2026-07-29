# prmignn/plotting.py
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_mesh(ax, X, edges, field, title):
    for i, j in edges:
        ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 'k-', alpha=0.3, lw=0.5)
    sc = ax.scatter(X[:, 0], X[:, 1], c=field, s=20, cmap='viridis')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return sc

def generate_figures(res, out_dir="results"):
    print("\n[Plotting] Generating figures...", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Mesh visualization
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))
    X_lf = np.array(res["mesh"]["X_lf"])
    X_hf = np.array(res["mesh"]["X_hf"])
    sc1 = plot_mesh(axes[0], X_lf, res["mesh"]["edges_lf"], res["mesh"]["sample_field_lf"], "LF Mesh (64 nodes)")
    sc2 = plot_mesh(axes[1], X_hf, res["mesh"]["edges_hf"], res["mesh"]["sample_field_hf"], "HF Mesh (256 nodes)")
    
    fig.colorbar(sc2, ax=axes, orientation='horizontal', fraction=0.046, pad=0.1, aspect=30)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.2, wspace=0.1)
    fig.savefig(os.path.join(out_dir, "fig_mesh_visualization.pdf"))
    plt.close(fig)

    # 2. Prediction comparison
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    true_f = np.array(res["prediction_fields_proposed"]["true"])
    pre_f = np.array(res["prediction_fields_proposed"]["lf_pretrained"])
    prop_f = np.array(res["prediction_fields_proposed"]["proposed"])
    base_f = np.array(res["prediction_fields_baseline"]["baseline"])
    vmin, vmax = true_f.min(), true_f.max()

    titles = [
        f"Ground Truth",
        f"LF-pretrained\n(L2={res['prediction_fields_proposed']['errors']['lf_pretrained']*100:.2f}%)",
        f"PR-MF-GNN\n(L2={res['prediction_fields_proposed']['errors']['proposed']*100:.2f}%)",
        f"HF-only Baseline\n(L2={res['prediction_fields_baseline']['error_baseline']*100:.2f}%)"
    ]
    fields = [true_f, pre_f, prop_f, base_f]
    
    for i, ax in enumerate(axes.flat):
        sc = ax.scatter(X_hf[:, 0], X_hf[:, 1], c=fields[i], s=10, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(titles[i], fontsize=9)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
    fig.colorbar(sc, ax=axes, orientation='horizontal', fraction=0.046, pad=0.08, aspect=30)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.12, wspace=0.1, hspace=0.2)
    fig.savefig(os.path.join(out_dir, "fig_prediction_comparison.pdf"))
    plt.close(fig)

    # 3. Data efficiency
    fig, ax = plt.subplots(figsize=(5, 4))
    snaps = res["data_efficiency"]["snap_counts"]
    prop_mean = np.array(res["data_efficiency"]["proposed"]["mean"]) * 100
    prop_std = np.array(res["data_efficiency"]["proposed"]["std"]) * 100
    base_mean = np.array(res["data_efficiency"]["baseline"]["mean"]) * 100
    base_std = np.array(res["data_efficiency"]["baseline"]["std"]) * 100

    ax.semilogx(snaps, base_mean, 'o-', label="HF-only baseline")
    ax.fill_between(snaps, base_mean - base_std, base_mean + base_std, alpha=0.2)
    ax.semilogx(snaps, prop_mean, 's-', label="PR-MF-GNN")
    ax.fill_between(snaps, prop_mean - prop_std, prop_mean + prop_std, alpha=0.2)
    ax.set_xlabel("Number of HF training snapshots")
    ax.set_ylabel(r"Relative $L_2$ error (\%)")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_data_efficiency.pdf"))
    plt.close(fig)

    # 4. Ablation study
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["HF-only", "No PDE loss", "Full-net FT", "PR-MF-GNN"]
    means = [
        res["baseline"]["mean_l2"] * 100,
        res["ablation_no_pde"]["mean_l2"] * 100,
        res["ablation_fullft"]["mean_l2"] * 100,
        res["proposed"]["mean_l2"] * 100
    ]
    stds = [
        res["baseline"]["std_l2"] * 100,
        res["ablation_no_pde"]["std_l2"] * 100,
        res["ablation_fullft"]["std_l2"] * 100,
        res["proposed"]["std_l2"] * 100
    ]
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=['gray', 'orange', 'red', 'blue'], alpha=0.7)
    bars[3].set_color('green')
    ax.set_ylabel(r"Relative $L_2$ error (\%)")
    plt.xticks(rotation=15, ha='right')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_ablation.pdf"))
    plt.close(fig)

    # 5. Training analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left: PDE residuals
    res_labels = ["Baseline", "No PDE", "Full FT", "PR-MF-GNN"]
    res_vals = [
        res["baseline"]["mean_pde_resid"],
        res["ablation_no_pde"]["mean_pde_resid"],
        res["ablation_fullft"]["mean_pde_resid"],
        res["proposed"]["mean_pde_resid"]
    ]
    ax1.bar(res_labels, [v*100 for v in res_vals], color=['gray', 'orange', 'red', 'green'], alpha=0.7)
    ax1.set_title("PDE Residual")
    ax1.set_ylabel(r"$\|R(\hat{u})\| \times 10^{-2}$")
    plt.sca(ax1)
    plt.xticks(rotation=15, ha='right')

    # Right: Loss curves
    s1_data = res["training_convergence"]["stage1"]
    s2_data = res["training_convergence"]["stage2"]
    s1_epochs = s1_data["epochs"]
    s1_loss = s1_data["total"]
    s2_epochs = [e + max(s1_epochs) + 1 for e in s2_data["epochs"]]
    s2_loss = s2_data["total"]
    
    ax2.semilogy(s1_epochs, s1_loss, label="Stage 1: LF pre-training")
    ax2.semilogy(s2_epochs, s2_loss, label="Stage 2: HF decoder FT")
    ax2.axvline(max(s1_epochs), color='k', linestyle='--', alpha=0.5)
    ax2.set_title("Training Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Total Loss (log scale)")
    ax2.legend()
    
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_training_analysis.pdf"))
    plt.close(fig)

    print(f"[Plotting] All figures saved to {out_dir}/ directory.", flush=True)