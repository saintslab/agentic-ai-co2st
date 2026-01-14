import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

params = {'font.size': 18,
          'axes.labelsize':18,
          'axes.titlesize':18,
          'axes.titleweight':'bold',
          'legend.fontsize': 18,
         }
matplotlib.rcParams.update(params)

def visualize_results(log_root="carbon_logs"):
    """
    Identifies the most recent session folder within carbon_logs and generates 
    a comparative analysis of absolute and normalized energy metrics.
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
            results[model][agency]["gpu"].append(g_e*1000)
            results[model][agency]["cpu"].append(c_e*1000)
            results[model][agency]["total"].append(m.get("total_energy_kwh", (g_e + c_e)*1000))
            results[model][agency]["co2"].append(m.get("total_co2eq_g", 0))
        except (json.JSONDecodeError, IOError): continue

    fig, axs = plt.subplots(2, 3, figsize=(18, 12), sharex='col')
    levels = ["none", "low", "medium", "high"]
    
    for model, data in results.items():
        b_gpu = np.mean(data["none"]["gpu"]) if data["none"]["gpu"] else 1e-10
        b_cpu = np.mean(data["none"]["cpu"]) if data["none"]["cpu"] else 1e-10
        b_tot = np.mean(data["none"]["total"]) if data["none"]["total"] else (b_gpu + b_cpu)
        
        m_vals = {k: [] for k in ["t", "g", "c", "nt", "ng", "nc"]}
        s_vals = {k: [] for k in ["t", "g", "c", "nt", "ng", "nc"]}
        
        for l in levels:
            g_v, c_v = np.array(data[l]["gpu"]), np.array(data[l]["cpu"])
            t_v = np.array(data[l]["total"]) if data[l]["total"] else (g_v + c_v)
            m_vals["t"].append(np.mean(t_v)); s_vals["t"].append(np.std(t_v))
            m_vals["g"].append(np.mean(g_v)); s_vals["g"].append(np.std(g_v))
            m_vals["c"].append(np.mean(c_v)); s_vals["c"].append(np.std(c_v))
            m_vals["nt"].append(np.mean(t_v) / b_tot); s_vals["nt"].append(np.std(t_v) / b_tot)
            m_vals["ng"].append(np.mean(g_v / b_gpu)); s_vals["ng"].append(np.std(g_v / b_gpu))
            m_vals["nc"].append(np.mean(c_v / b_cpu)); s_vals["nc"].append(np.std(c_v / b_cpu))

        for i, k in enumerate(["t", "g", "c"]):
            axs[0, i].errorbar(levels, m_vals[k], yerr=s_vals[k], label=model, fmt='-o', capsize=4,linewidth=3) 
            axs[1, i].errorbar(levels, m_vals["n"+k], yerr=s_vals["n"+k], label=model, fmt='-o', capsize=4,linewidth=3)

    titles = ["Total Energy (Wh)", "GPU Energy (Wh)", "CPU Energy (Wh)"]
    for i, title in enumerate(titles):
        axs[0, i].set_title(f"Absolute {title}", fontweight='bold')
        axs[1, i].set_title(f"Normalized {title}", fontweight='bold')
        axs[1, i].set_xlabel("Agency Level")
        axs[1, i].axhline(y=1.0, color='black', linestyle='-', alpha=0.2)
    
    for ax in axs.flat:
        ax.grid(True, linestyle=':', alpha=0.5); ax.legend(loc='best', fontsize='small')

    plt.suptitle(f"Agentic Energy Analysis: Session {os.path.basename(target_dir)}", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(target_dir, "comparative_analysis.png"), dpi=300)
    print(f"Analysis saved to {target_dir}/comparative_analysis.png")
    plt.show()

if __name__ == "__main__":
    visualize_results()
