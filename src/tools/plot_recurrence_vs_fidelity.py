# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
"""
Plot recurrence (count) vs maximum fidelity for control and changed cases.
"""

import os

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt


def get_max_fidelity_from_file(fid_loss_path):
    if not os.path.exists(fid_loss_path):
        return None
    try:
        data = np.loadtxt(fid_loss_path)
        if data.ndim == 1:
            fidelities = data
        else:
            fidelities = data[0] if data.shape[0] < data.shape[1] else data[:, 0]
        return np.max(fidelities)
    except Exception:
        return None


def collect_max_fidelities(base_path, pattern):
    max_fids = []
    for root, dirs, files in os.walk(base_path):
        if pattern in root and "log_fidelity_loss.txt" in files:
            fid_loss_path = os.path.join(root, "log_fidelity_loss.txt")
            max_fid = get_max_fidelity_from_file(fid_loss_path)
            if max_fid is not None:
                max_fids.append(max_fid)
    return max_fids


def plot_recurrence_vs_fidelity(base_path, save_path=None):
    control_fids = collect_max_fidelities(base_path, "repeated_control_")
    changed_fids = collect_max_fidelities(base_path, "repeated_changed_")

    bins = np.linspace(0, 1, 21)
    control_hist, _ = np.histogram(control_fids, bins=bins)
    changed_hist, _ = np.histogram(changed_fids, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.figure(figsize=(8, 6))
    width = (bins[1] - bins[0]) * 0.4
    plt.bar(bin_centers - width / 2, control_hist, width=width, label="Control (no change)", alpha=0.7, color="C0")
    plt.bar(
        bin_centers + width / 2, changed_hist, width=width, label="Changed (with config change)", alpha=0.7, color="C1"
    )
    plt.xlabel("Maximum Fidelity Reached")
    plt.ylabel("Recurrence (Count)")
    plt.title("Recurrence vs Maximum Fidelity")
    plt.legend()
    plt.grid(True)
    if save_path is None:
        save_path = os.path.join(base_path, "recurrence_vs_fidelity.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = "./generated_data/MULTIPLE_RUNS/latest"  # Default path
    plot_recurrence_vs_fidelity(base_path)
