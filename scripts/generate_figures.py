"""
Generate all visualizations from probing results and CKA analysis.

Usage:
    python scripts/generate_figures.py
"""

import sys
sys.path.insert(0, ".")

from src.analysis.visualization import (
    load_probing_results,
    plot_layer_curves_by_model,
    plot_information_landscape,
    plot_cross_model_normalized,
    plot_full_summary,
    plot_all_cka,
)


def main():
    print("=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)
    
    # --- Probing figures ---
    print("\nLoading probing results...")
    all_results = load_probing_results("results/probing")
    
    print(f"  Properties found: {list(all_results.keys())}")
    for prop, models in all_results.items():
        for model, probes in models.items():
            print(f"    {prop}/{model}: {list(probes.keys())}")
    
    print("\n[1] Layer-wise curves per property...")
    for prop_name in all_results:
        plot_layer_curves_by_model(
            all_results, prop_name, probe_type="linear",
            save_path=f"results/figures/layer_curves_{prop_name}_linear.png",
        )
    
    print("\n[2] Information landscape heatmaps...")
    all_models = set()
    for prop_results in all_results.values():
        all_models.update(prop_results.keys())
    
    for model_name in all_models:
        plot_information_landscape(
            all_results, model_name, probe_type="linear",
            save_path=f"results/figures/landscape_{model_name}.png",
        )
    
    print("\n[3] Cross-model normalized comparison...")
    for prop_name in all_results:
        plot_cross_model_normalized(
            all_results, prop_name, probe_type="linear",
            save_path=f"results/figures/cross_model_{prop_name}.png",
        )
    
    print("\n[4] Full summary figure...")
    plot_full_summary(
        all_results, probe_type="linear",
        save_path="results/figures/full_summary.png",
    )
    
    # --- CKA figures ---
    print("\n[5] CKA heatmaps...")
    from pathlib import Path
    if Path("results/cka").exists() and list(Path("results/cka").glob("*.npy")):
        plot_all_cka()
    else:
        print("  No CKA results found. Run: python scripts/run_cka.py")
    
    print("\n[DONE] All figures saved to results/figures/")


if __name__ == "__main__":
    main()
