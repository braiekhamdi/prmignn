# plot_from_json.py
import json
import os
from prmignn.plotting import generate_figures

OUT = "results"

def load_results():
    """Load results, merging proposed and ablation if run separately."""
    path_all = os.path.join(OUT, "results.json")
    path_prop = os.path.join(OUT, "results_proposed.json")
    path_abl = os.path.join(OUT, "results_ablation.json")
    
    if os.path.exists(path_all):
        with open(path_all, "r") as f:
            return json.load(f)
    elif os.path.exists(path_prop) and os.path.exists(path_abl):
        print("Merging results_proposed.json and results_ablation.json...")
        with open(path_prop, "r") as f:
            prop = json.load(f)
        with open(path_abl, "r") as f:
            abl = json.load(f)
        prop.update(abl)
        return prop
    else:
        raise FileNotFoundError("No result JSON files found. Run main.py first.")

if __name__ == "__main__":
    results = load_results()
    generate_figures(results, out_dir=OUT)