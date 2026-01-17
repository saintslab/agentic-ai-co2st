import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

def visualize_results(log_root="carbon_logs"):
    """
    Generates a structured grid of subplots where each model variant is assigned
    its own row to isolate absolute and normalized energetic trends across agency tiers.
    Additionally saves the aggregated statistical data to a JSON file.
    """
    session_folders = sorted(glob.glob(os.path.join(log_root, "session_*")))
    if not session_folders:
        print(f"No experimental sessions found in {log_root}.")
        return
    
    target_dir = session_folders[-1]
    print(f"[Analysis] Parsing most recent session: {target_dir}")
    
    meta_files = glob.glob(os.path.join(target_dir, "meta_*.json"))
    if not meta_files:
        print(f"No metadata files found in {target_dir}.")
        return
    
    results = {}
    for meta_path in meta_files:
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            model, agency = meta.get('model'), meta.get('agency_level')
            m = meta.get('metrics', {})
            if not model or not agency: continue
            if model not in results:
                results[model] = {l: {"gpu": [], "cpu": [], "total": [], "co2": []} 
                                 for l in ["none", "low", "medium", "high"]}
            g_e = m.get("gpu_energy_kwh", 0); c_e = m.get("cpu_energy_kwh", 0)
            results[model][agency]["gpu"].append(g_e)
            results[model][agency]["cpu"].append(c_e)
            results[model][agency]["total"].append(m.get("total_energy_kwh", g_e + c_e))
            results[model][agency]["co2"].append(m.get("total_co2eq_g", 0))
        except (json.JSONDecodeError, IOError): continue

    models = sorted(results.keys())
    num_models = len(models)
    levels = ["none", "low", "medium", "high"]
    colors = {'gemma:2b': '#4285F4', 'gemma:7b': '#EA4335'}
    
    # Statistical export structure
    export_data = {}

    fig, axs = plt.subplots(num_models, 3, figsize=(20, 6 * num_models), sharex='col', squeeze=False)
    
    for row_idx, model in enumerate(models):
        data = results[model]
        export_data[model] = {}
        
        # Baselines for normalization (None = 1.0)
        b_gpu = np.mean(data["none"]["gpu"]) if data["none"]["gpu"] else 1e-10
        b_cpu = np.mean(data["none"]["cpu"]) if data["none"]["cpu"] else 1e-10
        b_tot = np.mean(data["none"]["total"]) if data["none"]["total"] else (b_gpu + b_cpu)
        
        # Metrics to iterate through for export and plotting
        metric_keys = ["total", "gpu", "cpu", "co2"]
        baselines = [b_tot, b_gpu, b_cpu, None]
        titles = ["Total System Energy", "Isolated GPU Demand", "CPU Orchestration", "Carbon Footprint"]

        # Populate export data structure with stats
        for l in levels:
            export_data[model][l] = {}
            for key_idx, key in enumerate(metric_keys):
                vals = np.array(data[l][key])
                if len(vals) > 0:
                    mean_val = np.mean(vals)
                    std_val = np.std(vals)
                    export_data[model][l][key] = {
                        "mean": float(mean_val),
                        "std": float(std_val),
                        "unit": "kWh" if key != "co2" else "gCO2eq"
                    }
                    # Add normalized multipliers for energy components
                    if key_idx < 3:
                        base = baselines[key_idx]
                        export_data[model][l][f"{key}_multiplier"] = {
                            "mean": float(mean_val / base),
                            "std": float(std_val / base)
                        }
                else:
                    export_data[model][l][key] = {"mean": 0.0, "std": 0.0}

        # Visualization loop for the 3 energy columns
        for col_idx in range(3):
            key = metric_keys[col_idx]
            base = baselines[col_idx]
            title = titles[col_idx]
            
            means = [export_data[model][l][key]["mean"] for l in levels]
            stds = [export_data[model][l][key]["std"] for l in levels]
            norm_means = [export_data[model][l][f"{key}_multiplier"]["mean"] for l in levels]
            norm_stds = [export_data[model][l][f"{key}_multiplier"]["std"] for l in levels]

            ax = axs[row_idx, col_idx]
            model_color = colors.get(model, '#333333')
            
            # Plot Absolute Values (Primary Y-axis)
            ax.errorbar(levels, means, yerr=stds, label=f"Abs ({model})", 
                        fmt='-o', capsize=5, color=model_color, linewidth=2)
            ax.set_ylabel("Absolute Energy (kWh)", fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.5)
            
            # Create Twin Axis for Normalized Multipliers
            ax_norm = ax.twinx()
            ax_norm.set_ylabel("Multiplier (None=1.0)", fontsize=10, color='gray')
            ax_norm.axhline(y=1.0, color='black', linestyle='-', alpha=0.1)
            
            # Sync the normalized axis scale to the absolute axis
            y_lims = ax.get_ylim()
            ax_norm.set_ylim(y_lims[0] / base, y_lims[1] / base)

            if row_idx == 0:
                ax.set_title(f"{title}", fontsize=12, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"{model}\n\nAbsolute Energy (kWh)", fontsize=11, fontweight='bold')
            if row_idx == num_models - 1:
                ax.set_xlabel("Agency Level", fontsize=10)

    # Serialize the aggregated statistical data to JSON
    data_save_path = os.path.join(target_dir, "aggregated_results.json")
    with open(data_save_path, 'w') as f:
        json.dump(export_data, f, indent=4)
    print(f"Aggregated statistics saved to {data_save_path}")

    plt.suptitle(f"Multi-Model Agentic Energy Audit: Session {os.path.basename(target_dir)}", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(target_dir, "model_separated_analysis.png")
    plt.savefig(save_path, dpi=300)
    print(f"Model-separated analysis saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_results()
