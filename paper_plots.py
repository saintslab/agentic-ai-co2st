import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib 
params = {'font.size': 14,
          'axes.labelsize':14,
          'axes.titlesize':14,
          'axes.titleweight':'bold',
          'legend.fontsize': 18,
         }
matplotlib.rcParams.update(params)


def load_and_plot_stacked(log_root="carbon_logs", selected_models=None, norm_mode="absolute"):
    """
    Generates a comparative stacked bar chart for multiple models.
    norm_mode options: 
    - 'absolute': Raw kWh values.
    - 'per_model': Each model normalized to its own 'low' agency total energy.
    - 'global': All models normalized to the 'low' agency total of the first selected model.
    """
    
    target_dir = './'
    json_path = os.path.join(target_dir, "aggregated_results.json")
    
    if not os.path.exists(json_path):
        print(f"Aggregated results file not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        full_data = json.load(f)

    models_to_plot = selected_models if selected_models else sorted(full_data.keys())
    if not models_to_plot:
        print("No models available to plot.")
        return

    #levels = ["none", "low", "medium", "high"]
    # Define the expanded agency hierarchy
    levels = ["none", "low", "medium", "high", "highest"]
    
    # Pre-process data to inject the 'highest' aggregate tier
    for model in models_to_plot:
        if model in full_data and "medium" in full_data[model] and "high" in full_data[model]:
            med = full_data[model]["medium"]
            hig = full_data[model]["high"]
            
            # Aggregate means and combine variances for standard deviation
            full_data[model]["highest"] = {
                "gpu": {
                    "mean": med["gpu"]["mean"] + hig["gpu"]["mean"],
                    "std": np.sqrt(med["gpu"]["std"]**2 + hig["gpu"]["std"]**2)
                },
                "cpu": {
                    "mean": med["cpu"]["mean"] + hig["cpu"]["mean"],
                    "std": np.sqrt(med["cpu"]["std"]**2 + hig["cpu"]["std"]**2)
                },
                "total": {
                    "mean": med["total"]["mean"] + hig["total"]["mean"],
                    "std": np.sqrt(med["total"]["std"]**2 + hig["total"]["std"]**2)
                }
            }
    
    # Setup plotting aesthetics and grouping
    # We use a distinct color for each model's GPU part, and a lighter version for the CPU
    model_base_colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(models_to_plot)))
    
    fig, ax = plt.subplots(figsize=(16, 8))
    bar_width = 0.8 / len(models_to_plot)
    x = np.arange(len(levels))

    # Determine Global Normalization Factor if required
    global_norm_factor = 1.0
    if norm_mode == "global":
        first_model = models_to_plot[0]
        if "none" in full_data[first_model]:
            low_data = full_data[first_model]["none"]
            global_norm_factor = low_data["total"]["mean"] * 1000
        else:
            print("Warning: Global normalization requested but 'low' agency data missing for first model.")

    # Structures to manage custom x-ticks
    tick_positions = []
    tick_labels = []

    for i, model in enumerate(models_to_plot):
        if model not in full_data:
            continue
        
        # Determine Per-Model Normalization Factor if required
        model_norm_factor = 1.0
        if norm_mode == "per_model":
            if "low" in full_data[model]:
                model_norm_factor = full_data[model]["low"]["total"]["mean"] * 1000
            else:
                print(f"Warning: Normalization for {model} failed; 'low' agency data missing.")

        # Aggregate and normalize hardware means
        norm_factor = global_norm_factor if norm_mode == "global" else model_norm_factor
        
        gpu_means = np.array([full_data[model][l]["gpu"]["mean"]*1000 / norm_factor for l in levels])
        cpu_means = np.array([full_data[model][l]["cpu"]["mean"]*1000 / norm_factor for l in levels])
        
        gpu_stds = np.array([(full_data[model][l]["gpu"]["std"] * 1000.0) / norm_factor for l in levels])
        cpu_stds = np.array([(full_data[model][l]["cpu"]["std"] * 1000.0) / norm_factor for l in levels])
        
        # Total standard deviation for the error bars (combined variance)
        total_stds = np.sqrt(gpu_stds**2 + cpu_stds**2)
        # Calculate x-positions for grouped bars
        pos = x + (i - (len(models_to_plot) - 1) / 2) * bar_width
        
        # Collect positions for per-bar labels
        for p in pos:
            tick_positions.append(p)
            tick_labels.append(model)
        
        base_color = model_base_colors[i]
        
        # Plot stacked segments: GPU as solid base, CPU as lightened top
        ax.bar(pos, gpu_means, bar_width, label=f'{model} (GPU)', color=base_color, alpha=0.9, edgecolor='white', linewidth=0.5)
        ax.bar(pos, cpu_means, bar_width, bottom=gpu_means, label=f'{model} (CPU)', color=base_color, alpha=0.4, edgecolor='white', linewidth=0.5)

        # Apply standard deviation error bars to the total stack
        ax.errorbar(pos, gpu_means + cpu_means, yerr=total_stds, fmt='none', ecolor='black', capsize=3, alpha=0.7)
        # Annotate total values on top of bars
        #for j, p in enumerate(pos):
        #    total = gpu_means[j] + cpu_means[j]
        #    ax.text(p, total, f'{total:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold', rotation=0)

    # Formatting and labeling
    y_label = "Energy Consumption (Wh)" if norm_mode == "absolute" else f"Normalized Energy (None=1.0 via {norm_mode})"
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')

    # Configure multi-level X-axis labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=12)
    
    # Add agency level group labels below the model names
    for j, level in enumerate(levels):
        group_center = x[j]
        ax.text(group_center, -0.175 * ax.get_ylim()[1], level.upper(), 
                ha='center', va='top', fontsize=12, fontweight='black')

    #ax.set_xlabel("Agency Level", fontsize=12, fontweight='bold')
    #ax.set_xticks(x)
    #ax.set_xticklabels([l.capitalize() for l in levels], fontsize=11)
    
    #title_suffix = f" (Normalized: {norm_mode})" if norm_mode != "absolute" else " (Absolute)"
    #ax.set_title(f"Agentic Energy Footprint: Model Comparison{title_suffix}\nSession: {os.path.basename(target_dir)}", 
    #             fontsize=15, fontweight='bold', pad=20)
    
    # Custom Legend for Hardware Components Only
    gpu_patch = mpatches.Patch(color='gray', alpha=0.9, label='GPU Energy')
    cpu_patch = mpatches.Patch(color='gray', alpha=0.4, label='CPU Energy')
    ax.legend(handles=[gpu_patch, cpu_patch], loc='upper left', frameon=True, fontsize=14)
    
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    #ax.legend(title="Hardware Components", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    
    filename = f"stacked_comparison_{norm_mode}.pdf"
    save_path = os.path.join(target_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Analysis plot saved to {save_path}")
    #plt.show()

if __name__ == "__main__":
    # TARGET_MODELS: Specify models to compare (e.g., ["gemma:2b", "gemma:7b"])
    # Leave empty to include all models found in aggregated_results.json
#    TARGET_MODELS = ["qwen2.5:0.5b","qwen2.5:1.5b","qwen2.5:3b","qwen2.5:7b","gemma3:270m","gemma3:1b","gemma3:4b"]
    
    TARGET_MODELS = ["gemma3:270m","qwen2.5:0.5b","gemma3:4b","qwen2.5:7b"]
 
    # NORM_MODE: 'absolute', 'per_model', or 'global'
    NORM_MODE = "absolute" 
    
    load_and_plot_stacked(selected_models=TARGET_MODELS, norm_mode=NORM_MODE)
