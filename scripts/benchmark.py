"""
scripts/benchmark.py
====================
Generates comparison plots and tables comparing YOLOv8n Baseline and YOLO-DarkWater v1.0.

Plots generated in results/comparison/:
  - map_comparison.png (mAP50 and mAP50-95 comparison bar chart)
  - pr_curves.png (Precision vs. Recall curves from confidence sweep)
  - f1_curves.png (F1 Score vs. Confidence threshold curves)
  - speed_accuracy_scatter.png (Inference latency vs. mAP50 scatter plot)
  - training_loss_curves.png (Loss curve comparison over training epochs)
  - per_class_ap.png (Per-class AP comparison bar chart)
  - comparison_table.csv (Full metrics comparison sheet)

Usage:
    python scripts/benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Setup directories
OUT_DIR = REPO_ROOT / "results" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_loss_data(run_type: str) -> pd.DataFrame | None:
    """Finds and parses the training results.csv file."""
    search_dir = REPO_ROOT / "results" / run_type
    if not search_dir.exists():
        return None
        
    csv_files = list(search_dir.glob("**/results.csv"))
    if csv_files:
        try:
            df = pd.read_csv(csv_files[0])
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception:
            pass
    return None


def plot_loss_curves():
    """Generates training loss curves comparison."""
    baseline_df = get_loss_data("baseline")
    darkwater_df = get_loss_data("darkwater")
    
    if baseline_df is None and darkwater_df is None:
        print("Skipping loss curves: no results.csv files found.")
        return
        
    plt.figure(figsize=(10, 6))
    
    # Check what columns exist (train/box_loss, train/cls_loss etc.)
    if baseline_df is not None:
        box_col = [c for c in baseline_df.columns if "train/box_loss" in c]
        if box_col:
            plt.plot(baseline_df["epoch"], baseline_df[box_col[0]], 'b--', label="Baseline Box Loss")
        cls_col = [c for c in baseline_df.columns if "train/cls_loss" in c]
        if cls_col:
            plt.plot(baseline_df["epoch"], baseline_df[cls_col[0]], 'b-', label="Baseline Cls Loss")
            
    if darkwater_df is not None:
        box_col = [c for c in darkwater_df.columns if "train/box_loss" in c]
        if box_col:
            plt.plot(darkwater_df["epoch"], darkwater_df[box_col[0]], 'r--', label="DarkWater Box Loss")
        cls_col = [c for c in darkwater_df.columns if "train/cls_loss" in c]
        if cls_col:
            plt.plot(darkwater_df["epoch"], darkwater_df[cls_col[0]], 'r-', label="DarkWater Cls Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.title("Training Loss Convergence Comparison")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    loss_plot_path = OUT_DIR / "training_loss_curves.png"
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"Saved loss curves plot to {loss_plot_path}")


def generate_plots(results: dict):
    """Generates accuracy, PR, F1, per-class AP, and latency plots."""
    has_baseline = "baseline" in results
    has_darkwater = "darkwater" in results
    
    if not has_baseline and not has_darkwater:
        print("No evaluation results to plot.")
        return

    # Set styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # ------------------------------------------------------------------
    # 1. mAP Comparison Bar Chart
    # ------------------------------------------------------------------
    models_labels = []
    map50_vals = []
    map95_vals = []
    
    if has_baseline:
        models_labels.append("YOLOv8n Baseline")
        map50_vals.append(results["baseline"]["mAP50"])
        map95_vals.append(results["baseline"]["mAP50-95"])
    if has_darkwater:
        models_labels.append("YOLO-DarkWater")
        map50_vals.append(results["darkwater"]["mAP50"])
        map95_vals.append(results["darkwater"]["mAP50-95"])

    x = np.arange(len(models_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, map50_vals, width, label='mAP50', color='#3498db')
    rects2 = ax.bar(x + width/2, map95_vals, width, label='mAP50-95', color='#2ecc71')
    
    ax.set_ylabel('Score')
    ax.set_title('Detection Performance Comparison (mAP)')
    ax.set_xticks(x)
    ax.set_xticklabels(models_labels)
    ax.set_ylim(0, 1.0)
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    map_plot_path = OUT_DIR / "map_comparison.png"
    plt.savefig(map_plot_path, dpi=300)
    plt.close()
    print(f"Saved mAP comparison bar chart to {map_plot_path}")

    # ------------------------------------------------------------------
    # 2. PR Curves & F1 Curves (from confidence sweeps)
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    baseline_sweep: pd.DataFrame | None = None
    dw_sweep: pd.DataFrame | None = None
    if has_baseline:
        baseline_sweep = pd.DataFrame(results["baseline"]["confidence_sweep"])
        # Append point at (0.0, 1.0) and (1.0, 0.0) for standard PR layout
        r_vals = [1.0] + list(baseline_sweep["recall"]) + [0.0]
        p_vals = [0.0] + list(baseline_sweep["precision"]) + [1.0]
        # Sort by recall
        sorted_indices = np.argsort(r_vals)
        plt.plot(np.array(r_vals)[sorted_indices], np.array(p_vals)[sorted_indices], 
                 label=f"YOLOv8n Baseline (AUC={results['baseline']['mAP50']:.3f})", 
                 color='#3498db', linewidth=2)
                 
    if has_darkwater:
        dw_sweep = pd.DataFrame(results["darkwater"]["confidence_sweep"])
        r_vals = [1.0] + list(dw_sweep["recall"]) + [0.0]
        p_vals = [0.0] + list(dw_sweep["precision"]) + [1.0]
        sorted_indices = np.argsort(r_vals)
        plt.plot(np.array(r_vals)[sorted_indices], np.array(p_vals)[sorted_indices], 
                 label=f"YOLO-DarkWater v1.0 (AUC={results['darkwater']['mAP50']:.3f})", 
                 color='#e74c3c', linewidth=2)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (PR) Curves")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    pr_plot_path = OUT_DIR / "pr_curves.png"
    plt.savefig(pr_plot_path, dpi=300)
    plt.close()
    print(f"Saved Precision-Recall curves to {pr_plot_path}")

    # ------------------------------------------------------------------
    # 3. F1 Curve vs Confidence
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    if has_baseline and baseline_sweep is not None:
        plt.plot(baseline_sweep["threshold"], baseline_sweep["f1"], 
                 label="YOLOv8n Baseline", color='#3498db', linewidth=2)
    if has_darkwater and dw_sweep is not None:
        plt.plot(dw_sweep["threshold"], dw_sweep["f1"], 
                 label="YOLO-DarkWater v1.0", color='#e74c3c', linewidth=2)
                 
    plt.xlabel("Confidence Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs. Confidence Threshold")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    f1_plot_path = OUT_DIR / "f1_curves.png"
    plt.savefig(f1_plot_path, dpi=300)
    plt.close()
    print(f"Saved F1 curves to {f1_plot_path}")

    # ------------------------------------------------------------------
    # 4. Speed vs Accuracy Scatter Plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    if has_baseline:
        plt.scatter(results["baseline"]["inference_latency_ms"], results["baseline"]["mAP50"], 
                    s=150, color='#3498db', marker='o', label="YOLOv8n Baseline")
        plt.annotate("Baseline", (results["baseline"]["inference_latency_ms"] + 0.1, results["baseline"]["mAP50"] + 0.005))
        
    if has_darkwater:
        plt.scatter(results["darkwater"]["inference_latency_ms"], results["darkwater"]["mAP50"], 
                    s=150, color='#e74c3c', marker='^', label="YOLO-DarkWater")
        plt.annotate("YOLO-DarkWater", (results["darkwater"]["inference_latency_ms"] + 0.1, results["darkwater"]["mAP50"] + 0.005))

    plt.xlabel("Inference Latency (ms per image)")
    plt.ylabel("mAP50")
    plt.title("Speed vs. Accuracy Tradeoff")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    scatter_plot_path = OUT_DIR / "speed_accuracy_scatter.png"
    plt.savefig(scatter_plot_path, dpi=300)
    plt.close()
    print(f"Saved speed-accuracy scatter plot to {scatter_plot_path}")

    # ------------------------------------------------------------------
    # 5. Per-class AP Comparison Bar Chart
    # ------------------------------------------------------------------
    if has_baseline and has_darkwater:
        base_classes = set(results["baseline"]["per_class_ap"].keys())
        dw_classes = set(results["darkwater"]["per_class_ap"].keys())
        all_classes = sorted(list(base_classes.union(dw_classes)))
        
        baseline_ap = [results["baseline"]["per_class_ap"].get(c, 0.0) for c in all_classes]
        darkwater_ap = [results["darkwater"]["per_class_ap"].get(c, 0.0) for c in all_classes]
        
        y = np.arange(len(all_classes))
        height = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 8))
        rects1 = ax.barh(y - height/2, baseline_ap, height, label='YOLOv8n Baseline', color='#3498db')
        rects2 = ax.barh(y + height/2, darkwater_ap, height, label='YOLO-DarkWater v1.0', color='#e74c3c')
        
        ax.set_xlabel('mAP50')
        ax.set_title('Per-Class AP Comparison')
        ax.set_yticks(y)
        ax.set_yticklabels(all_classes)
        ax.set_xlim(0, 1.0)
        ax.legend()
        plt.tight_layout()
        
        per_class_plot_path = OUT_DIR / "per_class_ap.png"
        plt.savefig(per_class_plot_path, dpi=300)
        plt.close()
        print(f"Saved per-class AP chart to {per_class_plot_path}")


def generate_comparison_table(results: dict):
    """Generates the main comparison CSV and prints a styled table to stdout."""
    has_baseline = "baseline" in results
    has_darkwater = "darkwater" in results
    
    rows = []
    metrics = [
        ("Precision", "precision", ".4f"),
        ("Recall", "recall", ".4f"),
        ("mAP50", "mAP50", ".4f"),
        ("mAP50-95", "mAP50-95", ".4f"),
        ("F1 Score", "f1", ".4f"),
        ("Latency (ms/img)", "inference_latency_ms", ".2f"),
        ("Throughput (FPS)", "fps", ".1f"),
        ("Parameters (M)", "parameters_m", ".3f"),
        ("GFLOPs", "gflops", ".2f"),
        ("Model Size (MB)", "model_size_mb", ".2f"),
        ("Validation Time (s)", "validation_time_seconds", ".1f")
    ]

    for label, key, fmt in metrics:
        row = {"Metric": label}
        if has_baseline:
            row["YOLOv8n Baseline"] = f"{results['baseline'][key]:{fmt}}"
        else:
            row["YOLOv8n Baseline"] = "N/A"
            
        if has_darkwater:
            row["YOLO-DarkWater v1.0"] = f"{results['darkwater'][key]:{fmt}}"
            
            # Compute improvement delta
            if has_baseline and results["baseline"][key] != 0:
                base_v = results["baseline"][key]
                dw_v = results["darkwater"][key]
                if key in ["inference_latency_ms", "parameters_m", "gflops", "model_size_mb"]:
                    # Lower is better
                    pct = ((base_v - dw_v) / base_v) * 100
                    row["Improvement"] = f"{pct:+.1f}% (reduction)"
                else:
                    # Higher is better
                    pct = ((dw_v - base_v) / base_v) * 100 if base_v > 0 else 0
                    row["Improvement"] = f"{pct:+.1f}%"
            else:
                row["Improvement"] = "N/A"
        else:
            row["YOLO-DarkWater v1.0"] = "N/A"
            row["Improvement"] = "N/A"
            
        rows.append(row)

    df_comp = pd.DataFrame(rows)
    csv_path = OUT_DIR / "comparison_table.csv"
    df_comp.to_csv(csv_path, index=False)
    print(f"Saved comparison spreadsheet to {csv_path}")

    # Generate a markdown string representation
    print("\n" + "=" * 65)
    print("DETAILED PERFORMANCE COMPARISON TABLE")
    print("=" * 65)
    print(df_comp.to_markdown(index=False))
    print("=" * 65)


def main():
    print("=" * 60)
    print("YOLO-DarkWater Benchmarking Script")
    print("=" * 60)
    
    # 1. Read eval_results.json
    results_json = OUT_DIR / "eval_results.json"
    if not results_json.exists():
        print(f"ERROR: {results_json} not found. Run scripts/evaluate.py first.")
        sys.exit(1)
        
    with open(results_json) as f:
        results = json.load(f)

    # 2. Generate plots
    generate_plots(results)
    
    # 3. Plot training loss curve comparisons
    plot_loss_curves()
    
    # 4. Generate comparison table CSV
    generate_comparison_table(results)
    
    print("\nBenchmarking complete! Plots and tables saved to:")
    print(f"  {OUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
