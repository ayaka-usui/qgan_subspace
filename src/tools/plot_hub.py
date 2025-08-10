# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The plot tool"""

import os
import re

import matplotlib as mpl
import numpy as np

from tools.data.data_managers import print_and_log

mpl.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt


########################################################################
# MAIN PLOTTING FUNCTION
########################################################################
def generate_all_plots(
    base_path, log_path, n_runs, max_fidelity, common_initial_plateaus=None, run_names=None, x_label: str = "Run"
):
    """Generate all plots.

    Supports backward-compatible boolean `common_initial_plateaus`

    Optional `run_names` allows custom x-axis labels (list or dict mapping run index to string).
    """
    # 1) Histogram-style plots (unchanged)
    for run_idx in range(1, n_runs + 1):
        plot_recurrence_vs_fid(base_path, log_path, run_idx, max_fidelity, common_initial_plateaus)
    plot_comparison_all_runs(base_path, log_path, n_runs, max_fidelity, common_initial_plateaus)

    # 2) Scatter-style plots (refactored)
    if not common_initial_plateaus:
        # from_scratch (and change): single merged scatter
        scatter_plot(base_path, log_path, n_runs, max_fidelity, run_names, x_label)
    else:
        # from_common_plateaus: now three plots (clouds, avg_fidelity, success)
        scatter_plateau_clouds(base_path, log_path, n_runs, max_fidelity, run_names, x_label)
        scatter_plateau_avg_fidelity(base_path, log_path, n_runs, max_fidelity, run_names, x_label)
        scatter_plateau_success(base_path, log_path, n_runs, max_fidelity, run_names, x_label)
        # Add back overall plot (two data points per run)
        scatter_plateau_overall(base_path, log_path, n_runs, max_fidelity, run_names, x_label)


########################################################################
# REAL TIME RUN PLOTTING FUNCTION
########################################################################
def plt_fidelity_vs_iter(fidelities, losses, config, indx=0):
    fig, (axs1, axs2) = plt.subplots(1, 2)
    axs1.plot(range(len(fidelities)), fidelities)
    axs1.set_xlabel("Iteration")
    axs1.set_ylabel("Fidelity")
    axs1.set_title("Fidelity <target|gen> vs Iterations")
    axs2.plot(range(len(losses)), losses)
    axs2.set_xlabel("Iteration")
    axs2.set_ylabel("Loss")
    axs2.set_title("Wasserstein Loss vs Iterations")
    plt.tight_layout()

    # Save the figure
    fig_path = f"{config.figure_path}/{config.system_size}qubit_{config.gen_layers}_{indx}.png"
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)


#########################################################################
# PLOT INDIVIDUAL RUNS HISTOGRAMS
#########################################################################
def plot_recurrence_vs_fid(base_path, log_path, run_idx, max_fidelity, common_initial_plateaus):
    run_colors = plt.cm.tab10.colors  # Consistent palette for control and runs
    control_fids = (
        collect_max_fidelities_nested(base_path, r"repeated_control", None) if common_initial_plateaus else []
    )
    changed_fids = collect_latest_changed_fidelities_nested(base_path, common_initial_plateaus, run_idx)
    # Split last bin at max_fidelity
    bins = [*list(np.linspace(0, max_fidelity, 20)), max_fidelity, 1.0]
    control_hist, _ = np.histogram(control_fids, bins=bins) if control_fids else (np.zeros(len(bins) - 1), bins)
    changed_hist, _ = np.histogram(changed_fids, bins=bins)
    # Renormalize histograms to show distributions
    control_hist = control_hist / control_hist.sum() if control_hist.sum() > 0 else control_hist
    changed_hist = changed_hist / changed_hist.sum() if changed_hist.sum() > 0 else changed_hist
    bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
    plt.figure(figsize=(8, 6))
    width = (bins[1] - bins[0]) * 0.4
    bars = []
    if common_initial_plateaus and np.any(control_hist):
        bars.append(
            plt.bar(
                bin_centers - width / 2,
                control_hist,
                width=width,
                label=f"Control (no change) ({len(control_fids)} tries)",
                alpha=0.7,
                color=run_colors[0],
            )
        )
    if np.any(changed_hist):
        # Use the second color from the palette for the first run, or cycle if run_idx is given
        run_color = run_colors[run_idx % len(run_colors)] if run_idx else run_colors[1]
        run_label = (
            f"Run {run_idx} ({len(changed_fids)} tries)" if run_idx else f"Experiment Runs ({len(changed_fids)} tries)"
        )
        bars.append(
            plt.bar(
                bin_centers + width / 2,
                changed_hist,
                width=width,
                label=run_label,
                alpha=0.7,
                color=run_color,
            )
        )
    plt.xlabel("Maximum Fidelity Reached")
    plt.ylabel("Distribution (Fraction)")
    title = "Distribution vs Maximum Fidelity"
    if run_idx:
        title += f" (run {run_idx})"
    elif not common_initial_plateaus:
        title += " (Experiment Mode)"
    plt.title(title)
    if bars:
        plt.legend()
    plt.grid(True)
    save_path = os.path.join(
        base_path,
        f"comparison_distribution_vs_fidelity_run{run_idx}.png" if run_idx else "distribution_vs_fidelity.png",
    )
    plt.tight_layout()
    plt.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close()


###########################################################################
# PLOT COMPARISON OF ALL HISTOGRAMS TOGETHER
###########################################################################
def plot_comparison_all_runs(base_path, log_path, n_runs, max_fidelity, common_initial_plateaus):
    run_colors = plt.cm.tab10.colors
    control_fids = (
        collect_max_fidelities_nested(base_path, r"repeated_control", None) if common_initial_plateaus else []
    )
    # Split last bin at max_fidelity
    bins = [*list(np.linspace(0, max_fidelity, 20)), max_fidelity, 1.0]
    bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
    plt.figure(figsize=(10, 7))
    all_hists = []
    all_labels = []
    all_colors = []
    # Collect control as first 'run' if present
    if common_initial_plateaus and len(control_fids) > 0:
        control_hist, _ = np.histogram(control_fids, bins=bins)
        control_hist = control_hist / control_hist.sum() if control_hist.sum() > 0 else control_hist
        all_hists.append(control_hist)
        all_labels.append(f"Control (no change) ({len(control_fids)} tries)")
        all_colors.append(run_colors[0])
    # Collect all runs
    for run_idx in range(1, n_runs + 1):
        if common_initial_plateaus:
            changed_fids = collect_latest_changed_fidelities_nested_run(base_path, run_idx)
        else:
            changed_fids = collect_latest_changed_fidelities_nested(base_path, common_initial_plateaus, run_idx)
        changed_hist, _ = np.histogram(changed_fids, bins=bins)
        changed_hist = changed_hist / changed_hist.sum() if changed_hist.sum() > 0 else changed_hist
        all_hists.append(changed_hist)
        all_labels.append(f"Run {run_idx} ({len(changed_fids)} tries)")
        all_colors.append(run_colors[run_idx % len(run_colors)])
    # Plot as grouped bars: each group is a run (control is group 0 if present)
    n_groups = len(all_hists)
    width = (bins[1] - bins[0]) * 0.7 / n_groups
    for i, (hist, label, color) in enumerate(zip(all_hists, all_labels, all_colors)):
        plt.bar(
            bin_centers + width * (i - n_groups / 2 + 0.5),
            hist,
            width=width,
            label=label,
            alpha=0.7,
            color=color,
        )
    plt.xlabel("Maximum Fidelity Reached")
    plt.ylabel("Distribution (Fraction)")
    title = "Comparison: Distribution vs Maximum Fidelity (All Runs)"
    if not common_initial_plateaus:
        title += " (Experiment Mode)"
    plt.title(title)
    if n_groups > 0:
        plt.legend()
    plt.grid(True)
    save_path = os.path.join(base_path, "comparison_distribution_vs_fidelity_all.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close()


##########################################################################
# HELPER FUNCTIONS TO COLLECT MAX FIDELITIES
##########################################################################
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
    except (OSError, IOError, ValueError):
        return None


def collect_max_fidelities_nested(base_path, outer_pattern, inner_pattern):
    """
    Collect max fidelities from all outer_pattern/inner_pattern/fidelities/log_fidelity_loss.txt
    """
    max_fids = []
    for root, dirs, files in os.walk(base_path):
        if (
            re.search(outer_pattern, root)
            and (inner_pattern is None or re.search(inner_pattern, root))
            and os.path.basename(root) == "fidelities"
            and "log_fidelity_loss.txt" in files
        ):
            fid_loss_path = os.path.join(root, "log_fidelity_loss.txt")
            max_fid = get_max_fidelity_from_file(fid_loss_path)
            if max_fid is not None:
                max_fids.append(max_fid)
    return max_fids


def collect_latest_changed_fidelities_nested(base_path, common_initial_plateaus, run_idx=None):
    """
    Collect max fidelities for changed runs, supporting both folder structures.
    common_initial_plateaus: boolean, if True, uses the initial plateaus structure.
    If run_idx is not None, only collect for that run.
    """
    run_dirs = {}
    if common_initial_plateaus:
        pattern = (
            rf"initial_plateau_(\d+)/repeated_changed_run{run_idx}/(\d+)/fidelities$"
            if run_idx is not None
            else r"initial_plateau_(\d+)/repeated_changed_run(\d+)/(\d+)/fidelities$"
        )
    elif run_idx is not None:
        pattern = rf"experiment{run_idx}/(\d+)/fidelities$"
    else:
        pattern = r"experiment(\d+)/(\d+)/fidelities$"
    for root, dirs, files in os.walk(base_path):
        m = re.search(pattern, root)
        if m and "log_fidelity_loss.txt" in files:
            if common_initial_plateaus:
                if run_idx is not None:
                    run_y = run_idx
                    x_num = int(m[2])
                else:
                    run_y = int(m[2])
                    x_num = int(m[3])
                exp_j = int(m[1])
                key = (exp_j, x_num)
            else:
                if run_idx is not None:
                    run_y = run_idx
                    x_num = int(m[1])
                else:
                    run_y = int(m[1])
                    x_num = int(m[2])
                key = (run_y, x_num)
            if key not in run_dirs or run_y > run_dirs[key][0]:
                run_dirs[key] = (run_y, os.path.join(root, "log_fidelity_loss.txt"))
    max_fids = []
    for run_y, fid_loss_path in run_dirs.values():
        max_fid = get_max_fidelity_from_file(fid_loss_path)
        if max_fid is not None:
            max_fids.append(max_fid)
    return max_fids


def collect_latest_changed_fidelities_nested_run(base_path, run_idx):
    run_dirs = {}
    for root, dirs, files in os.walk(base_path):
        m = re.search(
            r"initial_plateau_(\d+)[/\\]repeated_changed_run" + str(run_idx) + r"[/\\](\d+)[/\\]fidelities$", root
        )
        if m and "log_fidelity_loss.txt" in files:
            exp_j = int(m[1])
            x_num = int(m[2])
            key = (exp_j, x_num)
            run_y = run_idx
            if key not in run_dirs or run_y > run_dirs[key][0]:
                run_dirs[key] = (run_y, os.path.join(root, "log_fidelity_loss.txt"))
    max_fids = []
    for run_y, fid_loss_path in run_dirs.values():
        max_fid = get_max_fidelity_from_file(fid_loss_path)
        if max_fid is not None:
            max_fids.append(max_fid)
    return max_fids


def collect_fidelities_by_plateau_for_run(base_path, run_idx):
    """Collect fidelities grouped by plateau for a specific run."""
    plateau_fids = {}
    for root, dirs, files in os.walk(base_path):
        m = re.search(rf"initial_plateau_(\d+)[/\\]repeated_changed_run{run_idx}[/\\](\d+)[/\\]fidelities$", root)
        if m and "log_fidelity_loss.txt" in files:
            plateau_num = int(m[1])
            fid_loss_path = os.path.join(root, "log_fidelity_loss.txt")
            max_fid = get_max_fidelity_from_file(fid_loss_path)
            if max_fid is not None:
                if plateau_num not in plateau_fids:
                    plateau_fids[plateau_num] = []
                plateau_fids[plateau_num].append(max_fid)
    return plateau_fids


def collect_fidelities_by_plateau_control(base_path):
    """Collect control fidelities grouped by plateau."""
    plateau_fids = {}
    for root, dirs, files in os.walk(base_path):
        m = re.search(r"initial_plateau_(\d+)[/\\]repeated_control[/\\]fidelities$", root)
        if m and "log_fidelity_loss.txt" in files:
            plateau_num = int(m[1])
            fid_loss_path = os.path.join(root, "log_fidelity_loss.txt")
            max_fid = get_max_fidelity_from_file(fid_loss_path)
            if max_fid is not None:
                if plateau_num not in plateau_fids:
                    plateau_fids[plateau_num] = []
                plateau_fids[plateau_num].append(max_fid)
    return plateau_fids


# Helpers for labels
def _label_for_run(run_idx, tries, run_names=None):
    if isinstance(run_names, dict):
        base = run_names.get(run_idx, f"Run {run_idx}")
    elif isinstance(run_names, (list, tuple)) and 0 <= run_idx - 1 < len(run_names):
        base = str(run_names[run_idx - 1])
    else:
        base = f"Run {run_idx}"
    return f"{base}\n({tries} tries)" if tries and tries > 0 else base


def _base_label_for_run(run_idx, run_names=None):
    """Return the base label for a run without tries count."""
    if isinstance(run_names, dict):
        return run_names.get(run_idx, f"Run {run_idx}")
    if isinstance(run_names, (list, tuple)) and 0 <= run_idx - 1 < len(run_names):
        return str(run_names[run_idx - 1])
    return f"Run {run_idx}"


def _draw_tries_sublabels(ax, x_positions, tries_counts, fontsize=8, y_offset=-0.045, color="0.35"):
    """Draw smaller '(X tries)' sublabels below the x-axis tick labels.

    Uses the x-axis transform so x is in data coords and y is in axes coords.
    """
    for x, t in zip(x_positions, tries_counts):
        if t and t > 0:
            ax.text(
                x,
                y_offset,
                f"({t} tries)",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=fontsize,
                color=color,
            )


def _make_jittered_xs(base_x: float, count: int, jitter: float = 0.12):
    """Return x positions jittered around base_x to reduce overlap.

    Only generates the xs so callers can choose their own scatter styling.
    """
    if not count:
        return []
    return [base_x + np.random.uniform(-jitter, jitter) for _ in range(count)]


########################################################################
# FROM SCRATCH SCATTER PLOTS
########################################################################
def scatter_plot(base_path, log_path, n_runs, max_fidelity, run_names=None, x_label: str = "Run"):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # Right axis for percentages
    # Threshold line on left axis now in percent
    ax1.axhline(100 * max_fidelity, color="C0", linestyle="--", label=f"max_fidelity={max_fidelity}")

    # Plot clouds and compute averages
    avg_fid_percent = []
    avg_success_percent = []
    x_ticks = []
    base_labels = []
    tries_counts = []

    for run_idx in range(1, n_runs + 1):
        vals = collect_latest_changed_fidelities_nested(base_path, False, run_idx)
        tries = len(vals) if vals else 0
        # Cloud on left axis (percent)
        if vals:
            xs = _make_jittered_xs(run_idx, len(vals))
            ax1.scatter(xs, [v * 100 for v in vals], color="gray", alpha=0.4, s=18, zorder=3)
        avg_f = (np.nanmean(vals) * 100) if vals else 0.0
        avg_s = (100.0 * np.sum(np.array(vals) >= max_fidelity) / tries) if tries > 0 else 0.0
        avg_fid_percent.append(avg_f)
        avg_success_percent.append(avg_s)

        # Centered overlays on right axis (percent)
        ax2.scatter([run_idx], [avg_f], color="green", edgecolors="black", linewidths=0.5, s=60, zorder=3)
        ax2.scatter([run_idx], [avg_s], color="red", marker="D", s=50, zorder=3)
        # Annotations for averages (percent)
        t1 = ax2.text(
            run_idx + 0.07,
            avg_f,
            f"{avg_f:.1f}%",
            ha="left",
            va="center",
            fontsize=10,
            color="green",
        )
        t1.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        t2 = ax2.text(
            run_idx + 0.07,
            avg_s,
            f"{avg_s:.1f}%",
            ha="left",
            va="center",
            fontsize=10,
            color="red",
        )
        t2.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

        x_ticks.append(run_idx)
        base_labels.append(_base_label_for_run(run_idx, run_names))
        tries_counts.append(tries)

    # Axes labels
    ax1.set_ylabel("Best Fidelity Achieved in each Repetition (%)")
    ax2.set_ylabel("Average/Success Rate of each Run (%)")
    ax1.set_xlabel(x_label, labelpad=18)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(base_labels)
    _draw_tries_sublabels(ax1, x_ticks, tries_counts, fontsize=8)
    fig.subplots_adjust(bottom=0.24)

    ax1.set_ylim(0, 105)
    ax2.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)

    # Legend
    handles = [
        plt.Line2D([0], [0], color="C0", linestyle="--", label=f"Max fidelity ({int(max_fidelity * 100)}%)"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            alpha=0.4,
            markersize=6,
            linestyle="None",
            label="Repetitions Best Fidelity (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markeredgecolor="black",
            markersize=7,
            linestyle="None",
            label="Run Avg Best Fidelity (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="red",
            markersize=7,
            linestyle="None",
            label="Run Success Rate (%)",
        ),
    ]
    ax1.legend(handles=handles, loc="best")

    save_path = os.path.join(base_path, "scatter_plot.png")
    fig.tight_layout()
    fig.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close(fig)


########################################################################
# PLATEAUS SCATTERS PLOTS
########################################################################
def _collect_all_plateau_ids(base_path):
    ids = set()
    # look for initial_plateau_X
    for root, dirs, files in os.walk(base_path):
        if m := re.search(r"initial_plateau_(\d+)(?=/|$)", root):
            ids.add(int(m[1]))
    return sorted(ids)


def scatter_plateau_clouds(base_path, log_path, n_runs, max_fidelity, run_names=None, x_label: str = "Run"):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax2 = ax.twinx()  # mirror/right axis for success rate overlays
    ax.axhline(100 * max_fidelity, color="C0", linestyle="-", label=f"max_fidelity={max_fidelity}")

    x_ticks, base_labels, tries_counts = [], [], []

    # Control run on the far left (x=0)
    control_plateau_fids = collect_fidelities_by_plateau_control(base_path)
    control_tries = sum(len(v) for v in control_plateau_fids.values()) if control_plateau_fids else 0
    if control_plateau_fids:
        for vals in control_plateau_fids.values():
            if not vals:
                continue
            xs = _make_jittered_xs(0, len(vals))
            ax.scatter(xs, [v * 100 for v in vals], color="gray", alpha=0.7, s=18, label=None)
        # Control overall averages: blue squares centered at x=0 on both axes
        control_vals_flat = [v for lst in control_plateau_fids.values() for v in lst]
        if control_vals_flat:
            overall_fid = float(np.nanmean(control_vals_flat)) * 100.0
            overall_succ = 100.0 * np.sum(np.array(control_vals_flat) >= max_fidelity) / len(control_vals_flat)
            ax.scatter([0], [overall_fid], color="blue", marker="s", edgecolors="black", linewidths=0.5, s=60, zorder=5)
            ax2.scatter(
                [0], [overall_succ], color="blue", marker="s", edgecolors="black", linewidths=0.5, s=60, zorder=5
            )
            # value tags
            t_f = ax.text(
                0 + 0.1,
                overall_fid,
                f"{overall_fid:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="blue",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t_f.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
            t_s = ax2.text(
                0 + 0.1,
                overall_succ,
                f"{overall_succ:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="blue",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t_s.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        x_ticks.append(0)
        base_labels.append("Control")
        tries_counts.append(control_tries)

    for run_idx in range(1, n_runs + 1):
        plateau_fids = collect_fidelities_by_plateau_for_run(base_path, run_idx)
        tries = sum(len(v) for v in plateau_fids.values()) if plateau_fids else 0
        for vals in plateau_fids.values():
            if not vals:
                continue
            xs = _make_jittered_xs(run_idx, len(vals), jitter=0.15)
            ax.scatter(xs, [v * 100 for v in vals], color="gray", alpha=0.7, s=18, label=None)
        # Overlay run averages (when data exists): Avg Fidelity on left axis, Success on right axis
        all_vals = [v for lst in plateau_fids.values() for v in lst]
        if all_vals:
            overall_fid = float(np.nanmean(all_vals)) * 100.0
            overall_succ = 100.0 * np.sum(np.array(all_vals) >= max_fidelity) / len(all_vals)
            # Avg fidelity (green circle) + tag on left axis
            ax.scatter([run_idx], [overall_fid], color="green", edgecolors="black", linewidths=0.5, s=60, zorder=5)
            t = ax.text(
                run_idx + 0.1,
                overall_fid,
                f"{overall_fid:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="green",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
            # Success rate (red diamond) + tag on right axis
            ax2.scatter([run_idx], [overall_succ], color="red", marker="D", s=55, zorder=5)
            t2 = ax2.text(
                run_idx + 0.1,
                overall_succ,
                f"{overall_succ:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="red",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t2.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        x_ticks.append(run_idx)
        base_labels.append(_base_label_for_run(run_idx, run_names))
        tries_counts.append(tries)

    ax.set_ylabel("Best Fidelity Achieved in each Repetition (%)")
    ax2.set_ylabel("Average/Success Rate of each Run (%)")
    ax.set_xlabel(x_label, labelpad=22)
    plt.xticks(x_ticks, base_labels)
    _draw_tries_sublabels(ax, x_ticks, tries_counts, fontsize=8)
    ax.figure.subplots_adjust(bottom=0.26)

    ax.set_ylim(0, 105)
    ax2.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Legend: threshold + example repetition marker + run averages overlays + control
    handles = [
        plt.Line2D([0], [0], color="C0", linestyle="-", label=f"Max fidelity ({int(max_fidelity * 100)}%)"),
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="gray", markersize=6, linestyle="None", label="Repetition"
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markeredgecolor="black",
            markersize=7,
            linestyle="None",
            label="Run Avg Best Fidelity (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="red",
            markersize=7,
            linestyle="None",
            label="Run Success Rate (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="blue",
            markeredgecolor="black",
            markersize=7,
            linestyle="None",
            label="Control (both)",
        ),
    ]
    plt.legend(handles=handles, loc="best")

    save_path = os.path.join(base_path, "scatter_plateau_clouds.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close()


def _pastelize(color, factor: float = 0.6):
    """Lighten a matplotlib color toward white by the given factor (0..1)."""
    r, g, b, a = mpl.colors.to_rgba(color)
    r = r + (1 - r) * factor
    g = g + (1 - g) * factor
    b = b + (1 - b) * factor
    return (r, g, b, a)


def scatter_plateau_avg_fidelity(base_path, log_path, n_runs, max_fidelity, run_names=None, x_label: str = "Run"):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.axhline(100 * max_fidelity, color="C0", linestyle="-", label=f"max_fidelity={max_fidelity}")
    plateau_ids = _collect_all_plateau_ids(base_path)
    cmap = plt.cm.get_cmap("tab20", max(1, len(plateau_ids)))
    plateau_colors = {pid: cmap(i % cmap.N) for i, pid in enumerate(plateau_ids)}
    line_colors = {pid: _pastelize(plateau_colors[pid], 0.5) for pid in plateau_ids}

    x_ticks, base_labels, tries_counts = [], [], []
    plateau_series = {pid: {} for pid in plateau_ids}

    # Control averages at x=0 (collect and plot jittered gray points)
    control_plateau_fids = collect_fidelities_by_plateau_control(base_path)
    control_tries = sum(len(v) for v in control_plateau_fids.values()) if control_plateau_fids else 0
    if control_plateau_fids:
        control_avgs = []
        for pid in plateau_ids:
            if vals := control_plateau_fids.get(pid, []):
                avg = float(np.nanmean(vals))
                plateau_series[pid][0] = avg
                control_avgs.append(avg)
        xs = _make_jittered_xs(0, len(control_avgs))
        ax.scatter(xs, [v * 100 for v in control_avgs], color="gray", s=45, zorder=3, alpha=0.7)
        if control_vals_flat := [v for lst in control_plateau_fids.values() for v in lst]:
            overall_fid = float(np.nanmean(control_vals_flat))
            ax.scatter(
                [0], [overall_fid * 100], color="blue", marker="s", edgecolors="black", linewidths=0.5, s=60, zorder=4
            )
            t = ax.text(
                0 + 0.1,
                overall_fid * 100,
                f"{overall_fid * 100:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="blue",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        x_ticks.append(0)
        base_labels.append("Control")
        tries_counts.append(control_tries)

    for run_idx in range(1, n_runs + 1):
        plateau_fids = collect_fidelities_by_plateau_for_run(base_path, run_idx)
        tries = sum(len(v) for v in plateau_fids.values()) if plateau_fids else 0
        run_avgs = []
        for pid in plateau_ids:
            if vals := plateau_fids.get(pid, []):
                avg = float(np.nanmean(vals))
                plateau_series[pid][run_idx] = avg
                run_avgs.append(avg)
        xs = _make_jittered_xs(run_idx, len(run_avgs))
        ax.scatter(xs, [v * 100 for v in run_avgs], color="gray", s=45, zorder=3, alpha=0.7)
        if all_vals := [v for lst in plateau_fids.values() for v in lst]:
            overall_fid = float(np.nanmean(all_vals))
            ax.scatter(
                [run_idx], [overall_fid * 100], color="green", edgecolors="black", linewidths=0.5, s=60, zorder=4
            )
            t = ax.text(
                run_idx + 0.1,
                overall_fid * 100,
                f"{overall_fid * 100:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="green",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        x_ticks.append(run_idx)
        base_labels.append(_base_label_for_run(run_idx, run_names))
        tries_counts.append(tries)

    # connect lines per plateau in percent
    for pid, series in plateau_series.items():
        if len(series) > 1:
            xs = sorted(series.keys())
            ys = [series[i] * 100 for i in xs]
            plt.plot(xs, ys, "--", color=line_colors[pid], alpha=0.5, linewidth=1, zorder=1)

    ax.set_ylabel("Average Best Fidelity (%)")
    ax.set_xlabel(x_label, labelpad=22)
    plt.xticks(x_ticks, base_labels)
    _draw_tries_sublabels(ax, x_ticks, tries_counts, fontsize=8)
    ax.figure.subplots_adjust(bottom=0.26)

    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Legend remains the same
    handles = [
        plt.Line2D([0], [0], color="C0", linestyle="-", label=f"Max fidelity ({int(max_fidelity * 100)}%)"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=6,
            linestyle="None",
            label="Plateau Avg Best Fidelity (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markeredgecolor="black",
            markersize=7,
            linestyle="None",
            label="Run Avg Best Fidelity (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="blue",
            markeredgecolor="black",
            markersize=7,
            linestyle="None",
            label="Control Avg Best Fidelity (%)",
        ),
    ]
    plt.legend(handles=handles, loc="best")

    save_path = os.path.join(base_path, "scatter_plateau_avg_fidelity.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close()


def scatter_plateau_success(base_path, log_path, n_runs, max_fidelity, run_names=None, x_label: str = "Run"):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    plateau_ids = _collect_all_plateau_ids(base_path)
    cmap = plt.cm.get_cmap("tab20", max(1, len(plateau_ids)))
    plateau_colors = {pid: cmap(i % cmap.N) for i, pid in enumerate(plateau_ids)}
    line_colors = {pid: _pastelize(plateau_colors[pid], 0.5) for pid in plateau_ids}

    x_ticks, base_labels, tries_counts = [], [], []
    plateau_series = {pid: {} for pid in plateau_ids}

    # Control success at x=0
    control_plateau_fids = collect_fidelities_by_plateau_control(base_path)
    control_tries = sum(len(v) for v in control_plateau_fids.values()) if control_plateau_fids else 0
    if control_plateau_fids:
        control_succ = []
        for pid in plateau_ids:
            if vals := control_plateau_fids.get(pid, []):
                succ = 100.0 * np.sum(np.array(vals) >= max_fidelity) / len(vals)
                plateau_series[pid][0] = succ
                control_succ.append(succ)
        xs = _make_jittered_xs(0, len(control_succ))
        ax.scatter(xs, control_succ, color="gray", s=45, zorder=3, alpha=0.7)
        if control_vals_flat := [v for lst in control_plateau_fids.values() for v in lst]:
            overall_succ = 100.0 * np.sum(np.array(control_vals_flat) >= max_fidelity) / len(control_vals_flat)
            ax.scatter(
                [0], [overall_succ], color="blue", marker="s", edgecolors="black", linewidths=0.5, s=60, zorder=4
            )
            t = ax.text(
                0 + 0.1,
                overall_succ,
                f"{overall_succ:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="blue",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        x_ticks.append(0)
        base_labels.append("Control")
        tries_counts.append(control_tries)

    for run_idx in range(1, n_runs + 1):
        plateau_fids = collect_fidelities_by_plateau_for_run(base_path, run_idx)
        tries = sum(len(v) for v in plateau_fids.values()) if plateau_fids else 0
        run_succ = []
        for pid in plateau_ids:
            if vals := plateau_fids.get(pid, []):
                succ = 100.0 * np.sum(np.array(vals) >= max_fidelity) / len(vals)
                plateau_series[pid][run_idx] = succ
                run_succ.append(succ)
        xs = _make_jittered_xs(run_idx, len(run_succ))
        ax.scatter(xs, run_succ, color="gray", s=45, zorder=3, alpha=0.7)
        if all_vals := [v for lst in plateau_fids.values() for v in lst]:
            overall_succ = 100.0 * np.sum(np.array(all_vals) >= max_fidelity) / len(all_vals)
            ax.scatter([run_idx], [overall_succ], color="red", marker="D", s=55, zorder=4)
            t = ax.text(
                run_idx + 0.1,
                overall_succ,
                f"{overall_succ:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="red",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        x_ticks.append(run_idx)
        base_labels.append(_base_label_for_run(run_idx, run_names))
        tries_counts.append(tries)

    # connect lines per plateau
    for pid, series in plateau_series.items():
        if len(series) > 1:
            xs = sorted(series.keys())
            ys = [series[i] for i in xs]
            plt.plot(xs, ys, "--", color=line_colors[pid], alpha=0.5, linewidth=1, zorder=1)

    ax.set_ylabel("Success Rate (%)")
    ax.set_xlabel(x_label)
    plt.xticks(x_ticks, base_labels)
    _draw_tries_sublabels(ax, x_ticks, tries_counts, fontsize=8)
    ax.figure.subplots_adjust(bottom=0.24)

    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=6,
            linestyle="None",
            label="Plateau Success Rate (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="red",
            markersize=7,
            linestyle="None",
            label="Run Success Rate (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="blue",
            markeredgecolor="black",
            markersize=7,
            linestyle="None",
            label="Control Success Rate (%)",
        ),
    ]
    plt.legend(handles=handles, loc="best")

    save_path = os.path.join(base_path, "scatter_plateau_success.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close()


def scatter_plateau_overall(base_path, log_path, n_runs, max_fidelity, run_names=None, x_label: str = "Run"):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    # Removed threshold line for the overall plot
    # ax1.axhline(100 * max_fidelity, color="C0", linestyle="-", label=f"max_fidelity={max_fidelity}")

    x_ticks, base_labels, tries_counts = [], [], []

    # Control overall at x=0
    control_plateau_fids = collect_fidelities_by_plateau_control(base_path)
    if control_vals := [v for lst in control_plateau_fids.values() for v in lst]:
        control_tries = len(control_vals)
        avg_fid = float(np.nanmean(control_vals)) * 100.0
        avg_succ = 100.0 * np.sum(np.array(control_vals) >= max_fidelity) / control_tries if control_tries > 0 else 0.0
        # Control as blue squares centered at x=0 on both axes
        ax1.scatter([0], [avg_fid], color="blue", marker="s", edgecolors="black", linewidths=0.5, s=60, zorder=4)
        ax2.scatter([0], [avg_succ], color="blue", marker="s", edgecolors="black", linewidths=0.5, s=60, zorder=4)
        # value tags
        t1 = ax1.text(
            0 + 0.1,
            avg_fid,
            f"{avg_fid:.1f}%",
            ha="left",
            va="center",
            fontsize=10,
            color="blue",
            zorder=100,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
        )
        t1.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        t2 = ax2.text(
            0 + 0.1,
            avg_succ,
            f"{avg_succ:.1f}%",
            ha="left",
            va="center",
            fontsize=10,
            color="blue",
            zorder=100,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
        )
        t2.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        x_ticks.append(0)
        base_labels.append("Control")
        tries_counts.append(control_tries)

    # Runs
    for run_idx in range(1, n_runs + 1):
        plateau_fids = collect_fidelities_by_plateau_for_run(base_path, run_idx)
        vals = [v for lst in plateau_fids.values() for v in lst]
        tries = len(vals)
        avg_fid = float(np.nanmean(vals) * 100.0) if vals else 0.0
        avg_succ = (100.0 * np.sum(np.array(vals) >= max_fidelity) / tries) if tries > 0 else 0.0

        # Only if there is data for this run
        if tries > 0:
            # Avg fidelity on left (green circle)
            ax1.scatter([run_idx], [avg_fid], color="green", edgecolors="black", linewidths=0.5, s=60)
            # Avg success on right (red diamond)
            ax2.scatter([run_idx], [avg_succ], color="red", marker="D", s=55)

            # tags
            t1 = ax1.text(
                run_idx + 0.1,
                avg_fid,
                f"{avg_fid:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="green",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t1.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

            t2 = ax2.text(
                run_idx + 0.1,
                avg_succ,
                f"{avg_succ:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="red",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t2.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        x_ticks.append(run_idx)
        base_labels.append(_base_label_for_run(run_idx, run_names))
        tries_counts.append(tries)

    ax1.set_ylabel("Average Best Fidelity (%)")
    ax2.set_ylabel("Success Rate (%)")
    ax1.set_xlabel(x_label, labelpad=22)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(base_labels)
    _draw_tries_sublabels(ax1, x_ticks, tries_counts, fontsize=8)
    fig.subplots_adjust(bottom=0.26)

    ax1.set_ylim(0, 105)
    ax2.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)

    # Updated legend without threshold entry
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markeredgecolor="black",
            markersize=7,
            linestyle="None",
            label="Avg Fidelity (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="red",
            markersize=7,
            linestyle="None",
            label="Success Rate (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="blue",
            markeredgecolor="black",
            markersize=7,
            linestyle="None",
            label="Control (both)",
        ),
    ]
    ax1.legend(handles=handles, loc="best")

    save_path = os.path.join(base_path, "scatter_plateau_overall.png")
    fig.tight_layout()
    fig.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close(fig)


def find_if_common_initial_plateaus(base_path):
    """
    Check if the folder structure indicates common initial plateaus.
    Returns True if 'initial_plateau_X' folders are found, False otherwise.
    """
    # Auto-detect whether this run used common initial plateaus
    try:
        plateau_dirs = [
            d
            for d in os.listdir(base_path)
            if d.startswith("initial_plateau_") and os.path.isdir(os.path.join(base_path, d))
        ]
        return len(plateau_dirs) > 0
    except FileNotFoundError:
        return False
